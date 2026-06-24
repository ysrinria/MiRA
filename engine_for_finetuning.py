import ast
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from mixup import Mixup
from scipy.special import softmax
from timm.utils import ModelEma, accuracy
from torch.utils.data.distributed import DistributedSampler

import utils


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================================================
# Train helpers
# =========================================================

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


# =========================================================
# Train
# =========================================================

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
):
    model.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            break

        it = start_steps + step

        if (lr_schedule_values is not None or wd_schedule_values is not None) and data_iter_step % update_freq == 0:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            loss, output = train_class_batch(model, samples, targets, criterion)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0 and model_ema is not None:
                model_ema.update(model)

            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order

            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )

            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        class_acc = None if mixup_fn is not None else (output.max(-1)[-1] == targets).float().mean()

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)

        min_lr = min(group["lr"] for group in optimizer.param_groups)
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if target.ndim == 2:
            target = target.argmax(dim=-1)
        target = target.long()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = model(videos)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if isinstance(getattr(data_loader, "sampler", None), DistributedSampler):
            metric_logger.synchronize_between_processes()

    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss,
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# =========================================================
# Test-time raw prediction dump
# =========================================================

@torch.no_grad()
def final_test(data_loader, model, device, file, criterion=None):
    """
    Write raw per-view predictions.

    Format:
        line 0 : "acc1, acc5"
        line 1+: "<id> <logits(list)> <target> <chunk_nb> <split_nb>"
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()
    final_result: List[str] = []

    last_acc1, last_acc5 = 0.0, 0.0

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if target.ndim == 2:
            target = target.argmax(dim=-1)
        target = target.long()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = model(videos)
            loss = criterion(output, target)

        out_cpu = output.detach().to(torch.float32).cpu()
        tgt_cpu = target.detach().to(torch.int64).cpu()
        chunk_cpu = chunk_nb.detach().to(torch.int64).cpu()
        split_cpu = split_nb.detach().to(torch.int64).cpu()

        for i in range(out_cpu.size(0)):
            final_result.append(
                "{} {} {} {} {}\n".format(
                    str(ids[i]),
                    str(out_cpu[i].tolist()),
                    int(tgt_cpu[i].item()),
                    int(chunk_cpu[i].item()),
                    int(split_cpu[i].item()),
                )
            )

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        last_acc1, last_acc5 = acc1.item(), acc5.item()

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if isinstance(getattr(data_loader, "sampler", None), DistributedSampler):
            metric_logger.synchronize_between_processes()

    rank = torch.distributed.get_rank() if (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ) else 0
    print(f"[DEBUG][rank {rank}] raw prediction lines: {len(final_result)}")

    p = Path(file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("{}, {}\n".format(last_acc1, last_acc5))
        f.writelines(final_result)

    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss,
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def run_evaluation(data_loader, model, device, file, criterion=None):
    return final_test(data_loader, model, device, file, criterion=criterion)


# =========================================================
# Metric helpers
# =========================================================

def _cm_from_pairs(pairs, num_classes: int):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for tgt, pred in pairs:
        if 0 <= tgt < num_classes and 0 <= pred < num_classes:
            cm[tgt, pred] += 1
    return cm


def _uar_war(cm: torch.Tensor):
    cmf = cm.float()
    support_true = cmf.sum(1).clamp_min(1.0)
    recall = cmf.diag() / support_true
    uar = recall.mean().item()
    war = (cmf.diag().sum() / cmf.sum().clamp_min(1.0)).item()
    return uar, war


def _f1_from_cm(cm: torch.Tensor):
    cmf = cm.float()
    tp = torch.diag(cmf)
    support_true = cmf.sum(1)
    support_pred = cmf.sum(0)

    prec_c = torch.where(support_pred > 0, tp / support_pred, torch.zeros_like(tp))
    rec_c = torch.where(support_true > 0, tp / support_true, torch.zeros_like(tp))

    denom = prec_c + rec_c
    f1_c = torch.where(denom > 0, 2 * prec_c * rec_c / denom, torch.zeros_like(denom))

    weighted_f1 = (f1_c * support_true).sum() / support_true.sum().clamp_min(1.0)
    macro_f1 = f1_c.mean()

    tp_sum = tp.sum()
    pred_sum = support_pred.sum()
    true_sum = support_true.sum()

    micro_prec = (tp_sum / pred_sum) if pred_sum > 0 else torch.tensor(0.0, device=cm.device)
    micro_rec = (tp_sum / true_sum) if true_sum > 0 else torch.tensor(0.0, device=cm.device)
    micro_den = micro_prec + micro_rec
    micro_f1 = (2 * micro_prec * micro_rec / micro_den).item() if micro_den.item() > 0 else 0.0

    return float(weighted_f1.item()), float(micro_f1), float(macro_f1.item())


_LOGIT_RE = re.compile(
    r"^(?P<id>\S+)\s+\[(?P<logits>[^\]]+)\]\s+"
    r"(?P<label>-?\d+)\s+(?P<chunk>-?\d+)\s+(?P<split>-?\d+)\s*$"
)

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _empty_metric_dict():
    return {
        "UAR": None,
        "WAR": None,
        "Acc1": None,
        "Acc5": None,
        "WeightedF1": None,
        "MicroF1": None,
        "MacroF1": None,
        "NumVideos": None,
    }


def _parse_logits_from_string(logits_s: str) -> Optional[np.ndarray]:
    try:
        arr = np.fromstring(logits_s, dtype=np.float64, sep=",")
        if arr.size == 0:
            arr = np.fromstring(logits_s.replace(" ", ""), dtype=np.float64, sep=",")
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _aggregate_prediction_lines(
    lines: List[str],
    expected_num_views: Optional[int] = None,
    strict: bool = False,
    debug: bool = False,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, int], Dict[str, set]]:
    """
    Aggregate raw per-view predictions by video id.
    """
    dict_feats: Dict[str, List[np.ndarray]] = {}
    dict_label: Dict[str, int] = {}
    dict_pos: Dict[str, set] = {}

    malformed_count = 0
    duplicate_count = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = _LOGIT_RE.match(line)
        if not m:
            malformed_count += 1
            continue

        vid_id = m.group("id")
        logits_s = m.group("logits")
        label = int(m.group("label"))
        chunk_nb = int(m.group("chunk"))
        split_nb = int(m.group("split"))

        arr = _parse_logits_from_string(logits_s)
        if arr is None:
            malformed_count += 1
            continue

        prob = softmax(arr)
        key = f"{chunk_nb}_{split_nb}"

        if vid_id not in dict_feats:
            dict_feats[vid_id] = []
            dict_label[vid_id] = label
            dict_pos[vid_id] = set()
        elif dict_label[vid_id] != label:
            raise ValueError(
                f"Inconsistent label for video id '{vid_id}': "
                f"{dict_label[vid_id]} vs {label}"
            )

        if key in dict_pos[vid_id]:
            duplicate_count += 1
            continue

        dict_feats[vid_id].append(prob)
        dict_pos[vid_id].add(key)

    if debug:
        print(f"[DEBUG] malformed / skipped lines: {malformed_count}")
        print(f"[DEBUG] duplicate (chunk, split) views removed: {duplicate_count}")
        print(f"[DEBUG] unique aggregated video ids: {len(dict_feats)}")

        view_counts = [len(v) for v in dict_feats.values()]
        if view_counts:
            print(
                f"[DEBUG] min/max/mean #views per video: "
                f"{min(view_counts)} / {max(view_counts)} / {np.mean(view_counts):.2f}"
            )
            print(f"[DEBUG] unique #views values: {sorted(set(view_counts))[:20]}")
        else:
            print("[DEBUG] no valid aggregated videos found")

    if expected_num_views is not None:
        bad = [(vid, len(dict_feats[vid])) for vid in dict_feats if len(dict_feats[vid]) != expected_num_views]
        if bad:
            msg = (
                f"Found {len(bad)} videos with != expected_num_views({expected_num_views}). "
                f"First 10 examples: {bad[:10]}"
            )
            if strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}")

    return dict_feats, dict_label, dict_pos


def _compute_metrics_from_prob_bank(
    dict_feats: Dict[str, List[np.ndarray]],
    dict_label: Dict[str, int],
    num_classes: int,
    debug: bool = False,
):
    video_pairs: List[Tuple[int, int]] = []
    top1_list: List[float] = []
    top5_list: List[float] = []

    for vid_id, feats in dict_feats.items():
        label = int(dict_label[vid_id])
        mean_prob = np.mean(feats, axis=0)
        pred = int(np.argmax(mean_prob))

        video_pairs.append((label, pred))
        top1_list.append(float(pred == label))
        top5_list.append(float(label in np.argsort(-mean_prob)[:5]))

    acc1 = float(np.mean(top1_list) * 100.0) if top1_list else 0.0
    acc5 = float(np.mean(top5_list) * 100.0) if top5_list else 0.0

    cm = _cm_from_pairs(video_pairs, num_classes)
    uar, war = _uar_war(cm)
    weighted_f1, micro_f1, macro_f1 = _f1_from_cm(cm)

    if debug:
        print(f"[DEBUG] aggregated video predictions: {len(video_pairs)}")
        print(f"[DEBUG] confusion matrix sum: {int(cm.sum().item())}")

    assert int(cm.sum().item()) == len(video_pairs), (
        f"CM sum {int(cm.sum().item())} != #video_pairs {len(video_pairs)}"
    )

    return {
        "UAR": uar,
        "WAR": war,
        "Acc1": acc1,
        "Acc5": acc5,
        "WeightedF1": weighted_f1,
        "MicroF1": micro_f1,
        "MacroF1": macro_f1,
        "NumVideos": len(video_pairs),
        "ConfusionMatrix": cm,
    }


def _write_metrics_log(log_txt: str, epoch, metrics: Dict):
    os.makedirs(os.path.dirname(log_txt), exist_ok=True)
    tag = f"[epoch {epoch}]"

    cm = metrics["ConfusionMatrix"]

    with open(log_txt, "a", encoding="utf-8") as lf:
        lf.write(
            f"{tag} Acc1={metrics['Acc1']:.4f}, Acc5={metrics['Acc5']:.4f}, "
            f"UAR={metrics['UAR']:.4f}, WAR={metrics['WAR']:.4f}, "
            f"WeightedF1={metrics['WeightedF1']:.4f}, "
            f"MicroF1={metrics['MicroF1']:.4f}, MacroF1={metrics['MacroF1']:.4f}\n"
        )
        lf.write(f"{tag} ConfusionMatrix:\n")
        for row in cm.tolist():
            lf.write(" ".join(str(int(v)) for v in row) + "\n")
        lf.write("\n")


# =========================================================
# Legacy shard merge helper
# =========================================================

def merge(eval_path, num_tasks):
    """
    Merge shard files and compute aggregated video predictions.
    """
    print("Reading individual output files")

    all_lines: List[str] = []
    for x in range(num_tasks):
        file = os.path.join(eval_path, f"{x}.txt")
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing shard file: {file}")

        with open(file, "r", encoding="utf-8") as fh:
            shard_lines = fh.readlines()
            if len(shard_lines) == 0:
                raise ValueError(f"Empty shard file: {file}")
            all_lines.extend(shard_lines[1:])

    dict_feats, dict_label, _ = _aggregate_prediction_lines(
        all_lines,
        expected_num_views=None,
        strict=False,
        debug=True,
    )

    metrics = _compute_metrics_from_prob_bank(
        dict_feats=dict_feats,
        dict_label=dict_label,
        num_classes=max(dict_label.values()) + 1 if dict_label else 1,
        debug=True,
    )

    pred_dict = {"id": [], "label": [], "pred": []}
    for vid_id, feats in dict_feats.items():
        mean_prob = np.mean(feats, axis=0)
        pred = int(np.argmax(mean_prob))
        label = int(dict_label[vid_id])

        pred_dict["id"].append(vid_id)
        pred_dict["label"].append(label)
        pred_dict["pred"].append(pred)

    return metrics["Acc1"], metrics["Acc5"], pred_dict


# =========================================================
# Main metric computation
# =========================================================

@torch.no_grad()
def compute_metrics_from_preds(
    preds_file: str,
    log_txt: str,
    num_classes: int,
    epoch,
    expected_num_views: Optional[int] = None,
    strict: bool = False,
    debug: bool = False,
):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return _empty_metric_dict()

    print("Reading prediction file:", preds_file)
    with open(preds_file, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    if len(lines) == 0:
        raise ValueError(f"Prediction file is empty: {preds_file}")

    raw_lines = lines[1:]
    if debug:
        print(f"[DEBUG] raw prediction lines (excluding header): {len(raw_lines)}")

    dict_feats, dict_label, _ = _aggregate_prediction_lines(
        raw_lines,
        expected_num_views=expected_num_views,
        strict=strict,
        debug=debug,
    )

    print(f"Computing final metrics on {len(dict_feats)} unique ids")

    metrics = _compute_metrics_from_prob_bank(
        dict_feats=dict_feats,
        dict_label=dict_label,
        num_classes=num_classes,
        debug=debug,
    )

    _write_metrics_log(log_txt, epoch, metrics)

    return {
        "UAR": metrics["UAR"],
        "WAR": metrics["WAR"],
        "Acc1": metrics["Acc1"],
        "Acc5": metrics["Acc5"],
        "WeightedF1": metrics["WeightedF1"],
        "MicroF1": metrics["MicroF1"],
        "MacroF1": metrics["MacroF1"],
        "NumVideos": metrics["NumVideos"],
    }


# =========================================================
# Backward-compatible alternate parser
# =========================================================

@torch.no_grad()
def append_metrics_from_preds(
    preds_file: str,
    log_txt: str,
    num_classes: int,
    epoch,
    aggregate_by_id: bool = True,
):
    """
    Backward-compatible parser.
    Prefer compute_metrics_from_preds() in the main pipeline.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return _empty_metric_dict()

    acc1 = acc5 = None
    pairs = []
    bank = {}

    with open(preds_file, "r", encoding="utf-8") as f:
        first = f.readline()
        if first:
            nums = _FLOAT_RE.findall(first)
            if len(nums) >= 2:
                try:
                    acc1, acc5 = float(nums[0]), float(nums[1])
                except Exception:
                    acc1 = acc5 = None

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            try:
                id_part, rest = line.split(" ", 1)
            except ValueError:
                continue

            lb = rest.find("[")
            rb = rest.find("]", lb + 1)
            if lb < 0 or rb < 0:
                continue

            logits_str = rest[lb:rb + 1]
            tail = rest[rb + 1:].strip()

            try:
                logits = torch.tensor(ast.literal_eval(logits_str), dtype=torch.float32)
            except Exception:
                continue

            toks = tail.split()
            if len(toks) < 3:
                continue

            try:
                tgt = int(toks[0])
            except Exception:
                continue

            if aggregate_by_id:
                probs = torch.softmax(logits, dim=-1)
                entry = bank.get(id_part)
                if entry is None:
                    bank[id_part] = {"sum": probs, "cnt": 1, "tgt": tgt}
                else:
                    entry["sum"] += probs
                    entry["cnt"] += 1
            else:
                pred = int(torch.argmax(logits).item())
                pairs.append((tgt, pred))

    if aggregate_by_id:
        for _, obj in bank.items():
            avg_probs = obj["sum"] / max(obj["cnt"], 1)
            pred = int(torch.argmax(avg_probs).item())
            pairs.append((obj["tgt"], pred))

    cm = _cm_from_pairs(pairs, num_classes)
    uar, war = _uar_war(cm)
    weighted_f1, micro_f1, macro_f1 = _f1_from_cm(cm)

    metrics = {
        "UAR": uar,
        "WAR": war,
        "Acc1": acc1,
        "Acc5": acc5,
        "WeightedF1": weighted_f1,
        "MicroF1": micro_f1,
        "MacroF1": macro_f1,
        "NumVideos": len(pairs),
        "ConfusionMatrix": cm,
    }

    os.makedirs(os.path.dirname(log_txt), exist_ok=True)
    tag = f"[epoch {epoch}]"
    with open(log_txt, "a", encoding="utf-8") as lf:
        if acc1 is not None and acc5 is not None:
            lf.write(f"{tag} Acc1={acc1:.4f}, Acc5={acc5:.4f}, ")
        lf.write(
            f"UAR={uar:.4f}, WAR={war:.4f}, "
            f"WeightedF1={weighted_f1:.4f}, MicroF1={micro_f1:.4f}, MacroF1={macro_f1:.4f}\n"
        )
        lf.write(f"{tag} ConfusionMatrix:\n")
        for row in cm.tolist():
            lf.write(" ".join(str(int(v)) for v in row) + "\n")
        lf.write("\n")

    return {
        "UAR": metrics["UAR"],
        "WAR": metrics["WAR"],
        "Acc1": metrics["Acc1"],
        "Acc5": metrics["Acc5"],
        "WeightedF1": metrics["WeightedF1"],
        "MicroF1": metrics["MicroF1"],
        "MacroF1": metrics["MacroF1"],
        "NumVideos": metrics["NumVideos"],
    }