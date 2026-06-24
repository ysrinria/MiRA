import argparse
import datetime
import json
import os
import shutil
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma

from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from my_datasets import build_dataset
from engine_for_finetuning import (
    train_one_epoch,
    validation_one_epoch,
    run_evaluation,
    compute_metrics_from_preds,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate
import utils
import modeling_finetune


def get_args():
    parser = argparse.ArgumentParser(
        "VideoMAE fine-tuning and evaluation script for video classification",
        add_help=False,
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=100, type=int)

    # Model
    parser.add_argument("--model", default="vit_base_patch16_224", type=str, metavar="MODEL")
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--fc_drop_rate", type=float, default=0.0, metavar="PCT")
    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, metavar="PCT")
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT")

    parser.add_argument("--disable_eval_during_finetuning", action="store_true", default=False)
    parser.add_argument("--model_ema", action="store_true", default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999)
    parser.add_argument("--model_ema_force_cpu", action="store_true", default=False)
    parser.set_defaults(model_ema_force_cpu=True)

    # FMP
    parser.add_argument("--use_st_block", default=False, action="store_true")
    parser.add_argument("--add_intra_attention", default=False, action="store_true")
    parser.add_argument("--add_fmp_attention", default=False, action="store_true")
    parser.add_argument("--fmp_num_last_layers", default=1, type=int)
    parser.add_argument("--fmp_no_use_ema", default=False, action="store_true")
    parser.add_argument("--fmp_use_residual", default=False, action="store_true")
    parser.add_argument("--use_fmp_flashlite", default=False, action="store_true")
    parser.add_argument(
        "--stats_mode",
        default="batch_ema",
        choices=["batch_ema", "batch", "instance"],
        type=str,
    )

    # Optimizer
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER")
    parser.add_argument("--opt_eps", type=float, default=1e-8, metavar="EPSILON")
    parser.add_argument("--opt_betas", type=float, nargs="+", metavar="BETA")
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--weight_decay_end", type=float, default=None)

    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--warmup_lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N")
    parser.add_argument("--warmup_steps", type=int, default=-1, metavar="N")

    # Augmentation
    parser.add_argument("--color_jitter", type=float, default=0.4, metavar="PCT")
    parser.add_argument("--num_sample", type=int, default=2)
    parser.add_argument("--aa", type=str, default="rand-m7-n4-mstd0.5-inc1", metavar="NAME")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--train_interpolation", type=str, default="bicubic")

    # Evaluation
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--short_side_size", type=int, default=224)
    parser.add_argument("--test_num_segment", type=int, default=5)
    parser.add_argument("--test_num_crop", type=int, default=3)
    parser.add_argument("--eval_strict", action="store_true", default=True)
    parser.add_argument("--no_eval_strict", action="store_false", dest="eval_strict")
    parser.add_argument("--eval_debug", action="store_true", default=False)

    # Random erase
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT")
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", action="store_true", default=False)

    # Mixup
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")

    # Fine-tuning
    parser.add_argument("--finetune", default="")
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)
    parser.add_argument("--init_scale", default=0.001, type=float)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument("--use_mean_pooling", action="store_true")
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")
    parser.add_argument("--loss_type", default="auto", choices=["auto", "weighted_ce"], type=str, help="training loss type")
    parser.add_argument("--eval_metric", default="acc1", choices=["acc1", "weighted_f1", "uar", "war", "macro_f1", "micro_f1"], type=str, help="metric used to select best checkpoint")
    # Dataset
    parser.add_argument("--data_path", default="/path/to/list_kinetics-400", type=str)
    parser.add_argument("--eval_data_path", default=None, type=str)
    parser.add_argument("--eval_target_in_train", default="test", choices=["test", "val"])
    parser.add_argument("--nb_classes", default=400, type=int)
    parser.add_argument("--imagenet_default_mean_and_std", default=True, action="store_true")
    parser.add_argument("--num_segments", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument(
        "--data_set",
        default="DFEW",
        choices=["DFEW", "MAFW", "DFEW_crop", "MAFW_crop", "AVCAFFE_V", "AVCAFFE_A", "FERV39k", "image_folder"],
        type=str,
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--enable_deepspeed", action="store_true", default=False)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except Exception:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_class_weights(dataset, args, device):
    labels = dataset.label_array
    mapper = dataset.class_label_map_fn 

    # string → int
    label_ids = np.array([mapper(x) for x in labels], dtype=np.int64)
    assert label_ids.min() >= 0
    assert label_ids.max() < args.nb_classes

    counts = np.bincount(label_ids, minlength=args.nb_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * args.nb_classes

    print("[INFO] counts:", counts.tolist())
    print("[INFO] weights:", weights.tolist())
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_eval_score(raw_metrics, eval_metric: str):
    metric_key_map = {
        "acc1": "Acc1",
        "weighted_f1": "WeightedF1",
        "uar": "UAR",
        "war": "WAR",
        "macro_f1": "MacroF1",
        "micro_f1": "MicroF1",
    }
    if eval_metric not in metric_key_map:
        raise ValueError(f"Unknown eval_metric: {eval_metric}")
    key = metric_key_map[eval_metric]
    if key not in raw_metrics:
        raise KeyError(f"Metric '{key}' not found in raw_metrics. Available: {list(raw_metrics.keys())}")
    return raw_metrics[key]


def merge_prediction_shards(tmp_dir: str, preds_file: str, num_tasks: int, global_rank: int):
    merged_raw_lines = 0

    with open(preds_file, "w", encoding="utf-8") as out_f:
        out_f.write("0.0, 0.0\n")
        for r in range(num_tasks):
            path_r = os.path.join(tmp_dir, f"{r}.txt")
            if not os.path.exists(path_r):
                raise FileNotFoundError(f"Missing shard file: {path_r}")
            with open(path_r, "r", encoding="utf-8") as sh_f:
                lines = sh_f.readlines()
            if len(lines) == 0:
                raise ValueError(f"Empty shard file: {path_r}")
            if len(lines) == 1:
                print(f"[WARN][rank{global_rank}] shard has header only: {path_r}", flush=True)

            payload = lines[1:]
            merged_raw_lines += len(payload)
            out_f.writelines(payload)

    with open(preds_file, "r", encoding="utf-8") as f:
        merged_lines = f.readlines()

    merged_payload_lines = max(0, len(merged_lines) - 1)
    print(
        f"[DEBUG][rank{global_rank}] merged raw prediction lines: "
        f"expected_from_shards={merged_raw_lines}, written={merged_payload_lines}",
        flush=True,
    )

    if merged_payload_lines != merged_raw_lines:
        raise ValueError(
            f"Merged line count mismatch: "
            f"expected_from_shards={merged_raw_lines}, written={merged_payload_lines}"
        )


def distributed_eval(model, data_loader, device, args, epoch, num_tasks, global_rank, mode="eval", criterion=None):
    if isinstance(epoch, int):
        epoch_str = f"{epoch:03d}"
    else:
        epoch_str = str(epoch)

    preds_file = os.path.join(args.output_dir, f"{mode}_epoch_{epoch_str}.txt")
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()

    if is_dist and args.dist_eval:
        tmp_dir = os.path.join(args.output_dir, f"_tmp_{mode}_{epoch_str}")
        os.makedirs(tmp_dir, exist_ok=True)

        shard_file = os.path.join(tmp_dir, f"{global_rank}.txt")
        _ = run_evaluation(data_loader, model, device, shard_file, criterion=criterion)

        torch.distributed.barrier()

        if utils.is_main_process():
            try:
                merge_prediction_shards(tmp_dir, preds_file, num_tasks, global_rank)
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                print(f"[rank{global_rank}] MERGE ERROR: {repr(e)}", flush=True)
                raise

        torch.distributed.barrier()

    else:
        if (not is_dist) or utils.is_main_process():
            _ = run_evaluation(data_loader, model, device, preds_file, criterion=criterion)
        if is_dist:
            torch.distributed.barrier()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return preds_file


def save_best_ckpt(
    *,
    eval_acc1,
    args,
    is_dist,
    device,
    max_accuracy,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    model_ema,
):
    if not (args.output_dir and args.save_ckpt):
        return max_accuracy, False

    if (not is_dist) or utils.is_main_process():
        is_best = (eval_acc1 is not None) and (eval_acc1 > max_accuracy)
        new_max = float(eval_acc1) if is_best else float(max_accuracy)
    else:
        is_best, new_max = False, 0.0

    if is_dist:
        t = torch.tensor([1.0 if is_best else 0.0, new_max], device=device, dtype=torch.float32)
        torch.distributed.broadcast(t, src=0)
        is_best = bool(int(t[0].item()))
        new_max = float(t[1].item())

    if is_best:
        if args.enable_deepspeed:
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch="best",
                model_ema=model_ema,
            )
        else:
            if (not is_dist) or utils.is_main_process():
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch="best",
                    model_ema=model_ema,
                )

    return new_max, is_best


def main(args, ds_init):
    utils.init_distributed_mode(args)
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_val = None
    dataset_test = None
    dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    if not args.disable_eval_during_finetuning:
        dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
        dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )

    use_dist_eval = args.dist_eval and is_dist

    if dataset_val is not None:
        if use_dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = None

    if dataset_test is not None:
        if use_dist_eval:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_test = None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    collate_func = partial(multiple_samples_collate, fold=False) if args.num_sample > 1 else None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )

    data_loader_val = None
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    data_loader_test = None
    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    if args.eval_target_in_train == "val" and data_loader_val is None:
        if utils.is_main_process():
            print("[WARN] eval_target_in_train=val but no validation set is defined. Falling back to test.")
        args.eval_target_in_train = "test"

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_st_block=args.use_st_block,
        add_intra_attention=args.add_intra_attention,
        add_fmp_attention=args.add_fmp_attention,
        fmp_num_last_layers=args.fmp_num_last_layers,
        fmp_no_use_ema=args.fmp_no_use_ema,
        fmp_use_residual=args.fmp_use_residual,
        use_fmp_flashlite=args.use_fmp_flashlite,
        stats_mode=args.stats_mode,
    )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (
        args.num_frames // 2,
        args.input_size // patch_size[0],
        args.input_size // patch_size[1],
    )
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune,
                map_location="cpu",
                check_hash=True,
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith("backbone."):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith("encoder."):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        if "pos_embed" in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model["pos_embed"]
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches

            orig_size = int(
                ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) //
                 (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5
            )
            new_size = int(
                (num_patches // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5
            )

            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(
                    -1,
                    args.num_frames // model.patch_embed.tubelet_size,
                    orig_size,
                    orig_size,
                    embedding_size,
                )
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1,
                    args.num_frames // model.patch_embed.tubelet_size,
                    new_size,
                    new_size,
                    embedding_size,
                )
                pos_tokens = pos_tokens.flatten(1, 3)
                checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = (len(data_loader_train) + args.update_freq - 1) // args.update_freq

    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequency = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        )
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list:", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model,
            args.weight_decay,
            skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
        )
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )
        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            ddp_device_ids = [args.gpu] if hasattr(args, "gpu") and args.gpu is not None else None
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=ddp_device_ids,
                find_unused_parameters=True,
            )
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args,
            model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
        loss_scaler = NativeScaler()

    print("Use step-level LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )

    if args.loss_type == "weighted_ce":
        if mixup_fn is not None:
            print("[INFO] weighted_ce selected -> disabling mixup/cutmix")
            mixup_fn = None
        args.mixup = 0.0
        args.cutmix = 0.
        args.cutmix_minmax = None
        class_weights = get_class_weights(dataset_train, args, device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss_type == "auto":
        if mixup_fn is not None:
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.0:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    eval_criterion = torch.nn.CrossEntropyLoss()

    print("criterion =", str(criterion))
    print("eval_criterion =", str(eval_criterion))
    print(f"[INFO] final mixup={args.mixup}, cutmix={args.cutmix}, cutmix_minmax={args.cutmix_minmax}, mixup_fn={'ON' if mixup_fn is not None else 'OFF'}")

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
    )

    def evaluate_and_log(preds_file: str, log_txt: str, epoch_tag):
        return compute_metrics_from_preds(
            preds_file=preds_file,
            log_txt=log_txt,
            num_classes=args.nb_classes,
            epoch=epoch_tag,
            expected_num_views=args.test_num_segment * args.test_num_crop,
            strict=args.eval_strict,
            debug=args.eval_debug,
        )

    if args.eval:
        if data_loader_test is None:
            if utils.is_main_process():
                print("[Eval-only] No test loader available")
            exit(0)

        preds_file = distributed_eval(
            model=model,
            data_loader=data_loader_test,
            device=device,
            args=args,
            epoch="eval-only",
            num_tasks=num_tasks,
            global_rank=global_rank,
            mode="eval",
            criterion=eval_criterion,
        )

        if (not is_dist) or utils.is_main_process():
            metrics = evaluate_and_log(
                preds_file=preds_file,
                log_txt=os.path.join(args.output_dir, "log_eval.txt"),
                epoch_tag="eval_only",
            )

            num_videos = metrics.get("NumVideos", "unknown")
            print(
                f"[Eval-only] Metrics on {num_videos} aggregated test videos: "
                f"Top-1: {metrics['Acc1']:.2f}%, Top-5: {metrics['Acc5']:.2f}%, "
                f"UAR={metrics['UAR']:.4f}, WAR={metrics['WAR']:.4f}, "
                f"WeightedF1={metrics['WeightedF1']:.4f}, "
                f"MicroF1={metrics['MicroF1']:.4f}, MacroF1={metrics['MacroF1']:.4f}"
            )

        if is_dist:
            torch.distributed.barrier()
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_epoch_current = -1

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if use_dist_eval:
            if data_loader_val is not None and hasattr(data_loader_val.sampler, "set_epoch"):
                data_loader_val.sampler.set_epoch(epoch)
            if data_loader_test is not None and hasattr(data_loader_test.sampler, "set_epoch"):
                data_loader_test.sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
        )

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                if args.enable_deepspeed:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        model_ema=model_ema,
                    )
                else:
                    if (not is_dist) or utils.is_main_process():
                        utils.save_model(
                            args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch=epoch,
                            model_ema=model_ema,
                        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        eval_acc1_for_ckpt = None
        eval_metrics = None

        if args.eval_target_in_train == "test":
            if data_loader_test is None:
                if utils.is_main_process():
                    print("[WARN] eval_target_in_train=test but no test loader is available")
            else:
                preds_file = distributed_eval(
                    model=model,
                    data_loader=data_loader_test,
                    device=device,
                    args=args,
                    epoch=epoch,
                    num_tasks=num_tasks,
                    global_rank=global_rank,
                    mode="train-test",
                    criterion=eval_criterion,
                )

                if (not is_dist) or utils.is_main_process():
                    raw_metrics = evaluate_and_log(
                        preds_file=preds_file,
                        log_txt=os.path.join(args.output_dir, "log.txt"),
                        epoch_tag=epoch,
                    )

                    eval_acc1_for_ckpt = get_eval_score(raw_metrics, args.eval_metric)

                    print(
                        f"[Epoch {epoch}] Test metrics: "
                        f"Acc1={raw_metrics['Acc1']:.2f}%, Acc5={raw_metrics['Acc5']:.2f}%, "
                        f"UAR={raw_metrics['UAR']:.4f}, WAR={raw_metrics['WAR']:.4f}, "
                        f"WeightedF1={raw_metrics['WeightedF1']:.4f}, "
                        f"MicroF1={raw_metrics['MicroF1']:.4f}, MacroF1={raw_metrics['MacroF1']:.4f}, "
                        f"BestMetric({args.eval_metric})={eval_acc1_for_ckpt:.4f}"
                    )

                    if log_writer is not None:
                        log_writer.update(test_acc1=raw_metrics["Acc1"], head="perf", step=epoch)
                        if raw_metrics["Acc5"] is not None:
                            log_writer.update(test_acc5=raw_metrics["Acc5"], head="perf", step=epoch)
                        log_writer.update(test_uar=raw_metrics["UAR"], head="perf", step=epoch)
                        log_writer.update(test_war=raw_metrics["WAR"], head="perf", step=epoch)

                    eval_metrics = {f"test_{k}": v for k, v in raw_metrics.items()}

        elif args.eval_target_in_train == "val" and data_loader_val is not None:
            test_stats = None

            if use_dist_eval:
                test_stats = validation_one_epoch(data_loader_val, model, device)
            else:
                if (not is_dist) or utils.is_main_process():
                    test_stats = validation_one_epoch(data_loader_val, model, device)

            if test_stats is not None:
                eval_acc1_for_ckpt = test_stats["acc1"]

                print(f"Validation Acc@1 on {len(dataset_val)} samples: {test_stats['acc1']:.1f}%")

                if log_writer is not None:
                    log_writer.update(val_acc1=test_stats["acc1"], head="perf", step=epoch)
                    log_writer.update(val_acc5=test_stats["acc5"], head="perf", step=epoch)
                    log_writer.update(val_loss=test_stats["loss"], head="perf", step=epoch)

                eval_metrics = {f"val_{k}": v for k, v in test_stats.items()}

            if is_dist:
                torch.distributed.barrier()

        if eval_acc1_for_ckpt is not None and eval_acc1_for_ckpt > max_accuracy:
            best_epoch_current = epoch

        max_accuracy, _ = save_best_ckpt(
            eval_acc1=eval_acc1_for_ckpt,
            args=args,
            is_dist=is_dist,
            device=device,
            max_accuracy=max_accuracy,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
        )

        if eval_acc1_for_ckpt is not None and ((not is_dist) or utils.is_main_process()):
            print(f"[Epoch {epoch}] Max accuracy ({args.eval_target_in_train}): {max_accuracy:.2f}%")

        if eval_metrics is not None:
            log_stats.update(eval_metrics)
        
        log_stats["best_epoch_current"] = best_epoch_current

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    if data_loader_test is not None:
        if args.enable_deepspeed:
            best_ckpt_path = os.path.join(args.output_dir, "checkpoint-best")
        else:
            best_ckpt_path = os.path.join(args.output_dir, "checkpoint-best.pth")

        if os.path.exists(best_ckpt_path):
            if (not is_dist) or utils.is_main_process():
                print(f"[Best] Loading checkpoint from: {best_ckpt_path}")

            if args.enable_deepspeed:
                model.load_checkpoint(args.output_dir, tag="checkpoint-best")
            else:
                checkpoint = torch.load(best_ckpt_path, map_location="cpu")
                ckpt_model = checkpoint.get("model", checkpoint)
                model_without_ddp.load_state_dict(ckpt_model, strict=False)

            model.to(device)
            model.eval()

            if is_dist:
                torch.distributed.barrier()

            preds_file = distributed_eval(
                model=model,
                data_loader=data_loader_test,
                device=device,
                args=args,
                epoch="best",
                num_tasks=num_tasks,
                global_rank=global_rank,
                mode="best",
                criterion=eval_criterion,
            )

            if (not is_dist) or utils.is_main_process():
                metrics = evaluate_and_log(
                    preds_file=preds_file,
                    log_txt=os.path.join(args.output_dir, "log.txt"),
                    epoch_tag="best",
                )

                num_videos = metrics.get("NumVideos", "unknown")
                print(
                    f"[Best] Metrics on {num_videos} aggregated test videos: "
                    f"Top-1: {metrics['Acc1']:.2f}%, Top-5: {metrics['Acc5']:.2f}%, "
                    f"UAR={metrics['UAR']:.4f}, WAR={metrics['WAR']:.4f}, "
                    f"WeightedF1={metrics['WeightedF1']:.4f}, "
                    f"MicroF1={metrics['MicroF1']:.4f}, MacroF1={metrics['MacroF1']:.4f}"
                )

            if is_dist:
                torch.distributed.barrier()
        else:
            if (not is_dist) or utils.is_main_process():
                print(f"[Best] Checkpoint not found: {best_ckpt_path}")
    else:
        if utils.is_main_process():
            print("[Best] No test loader available; skipping best-checkpoint evaluation.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)