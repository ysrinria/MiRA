import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F


def info_NCE_pair(z1, z2, tau=0.1):
    """
    z1: [B, D] or [B, N, D]
    z2: [B, D] or [B, N, D]
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits12 = torch.matmul(z1, z2.transpose(-2, -1)) / tau
    logits21 = torch.matmul(z2, z1.transpose(-2, -1)) / tau

    B = z1.size(0)
    labels = torch.arange(B, device=z1.device)

    if logits12.dim() > 2:
        logits12 = logits12.mean(dim=1)
        logits21 = logits21.mean(dim=1)

    loss = 0.5 * (
        F.cross_entropy(logits12, labels) +
        F.cross_entropy(logits21, labels)
    )
    return loss

class UncertaintyWeighter(nn.Module):
    """
    Uncertainty Weighting:
    L = sum(exp(-log_sigma_i) * L_i + log_sigma_i) + reg * ||log_sigma||^2
    """
    def __init__(self, task_names, init_log_sigma=0.0, clamp=(-3.0, 3.0), reg=1e-4):
        super().__init__()
        self.task_names = list(task_names)
        self.log_sigma = nn.Parameter(torch.full((len(task_names),), float(init_log_sigma)))
        self.clamp = clamp
        self.reg = reg

    @torch.no_grad()
    def get_weights(self):
        ls = self.log_sigma
        if self.clamp is not None:
            ls = ls.clamp(self.clamp[0], self.clamp[1])
        w = torch.exp(-ls)
        return {name: float(w[i].item()) for i, name in enumerate(self.task_names)}

    def forward(self, losses_dict):
        loss_vec = torch.stack([losses_dict[name] for name in self.task_names])  # [T]
        ls = self.log_sigma
        if self.clamp is not None:
            ls = ls.clamp(self.clamp[0], self.clamp[1])
        weights = torch.exp(-ls)
        total = (weights * loss_vec).sum() + ls.sum()
        if self.reg and self.reg > 0:
            total = total + self.reg * (self.log_sigma ** 2).sum()
        return total


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None,
                    uw: torch.nn.Module = None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func_pixel = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch   
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs_recon = model(videos, bool_masked_pos)
            loss_total = loss_func_pixel(input=outputs_recon, target=labels) 

        loss_total_value = loss_total.item()

        if not math.isfinite(loss_total_value):
            print("Total-Loss = {}, stopping training".format(loss_total_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss_total, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_total_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_total_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            if uw is not None:
                ws = uw.get_weights()
                log_writer.update(uw_w_feature=ws.get('feature', 0.0), head="opt")
                log_writer.update(uw_w_recon=ws.get('recon', 0.0), head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
