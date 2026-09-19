#!/usr/bin/env python3
"""Single-video inference for a fine-tuned VideoMAE/MiRA classifier.

Place this file in the root of the VideoMAE/MiRA repository.  It intentionally
uses the repository's ``video_transforms`` and ``volume_transforms`` so that
test-time preprocessing matches the fine-tuning pipeline.

Unlike ``run_videomae_vis.py``, this script loads ``modeling_finetune`` and a
classification checkpoint.  It does not run the masked-video decoder.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from timm.models import create_model

# Importing this module registers the VideoMAE classification models in timm.
import modeling_finetune  # noqa: F401
import video_transforms
import volume_transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Single-video inference for a fine-tuned VideoMAE/MiRA classifier"
    )
    parser.add_argument("video", type=Path, help="input video")
    parser.add_argument("checkpoint", help="fine-tuned checkpoint path or URL")

    # Output
    parser.add_argument("--class-names", default=None,
                        help="JSON/TXT label file or comma-separated class names")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)

    # Runtime and multi-view inference
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps, or cpu")
    parser.add_argument("--amp-dtype", choices=("auto", "fp32", "fp16", "bf16"),
                        default="auto")
    parser.add_argument("--view-batch-size", type=int, default=4)
    parser.add_argument("--test_num_segment", "--test-num-segment",
                        dest="test_num_segment", type=int, default=None)
    parser.add_argument("--test_num_crop", "--test-num-crop",
                        dest="test_num_crop", type=int, default=None)
    parser.add_argument(
        "--aggregation",
        choices=("probabilities", "logits"),
        default="probabilities",
        help="VideoMAE evaluation averages per-view probabilities",
    )

    # Model/data parameters. None means: checkpoint args first, then defaults.
    parser.add_argument("--model", default=None)
    parser.add_argument("--nb_classes", "--nb-classes", dest="nb_classes",
                        type=int, default=None)
    parser.add_argument("--input_size", "--input-size", dest="input_size",
                        type=int, default=None)
    parser.add_argument("--short_side_size", "--short-side-size",
                        dest="short_side_size", type=int, default=None)
    parser.add_argument("--num_frames", "--num-frames", dest="num_frames",
                        type=int, default=None)
    parser.add_argument("--num_segments", "--num-segments", dest="num_segments",
                        type=int, default=None,
                        help="model-side segments; normally 1, not test-time views")
    parser.add_argument("--sampling_rate", "--sampling-rate", dest="sampling_rate",
                        type=int, default=None)
    parser.add_argument("--tubelet_size", "--tubelet-size", dest="tubelet_size",
                        type=int, default=None)
    parser.add_argument("--fc_drop_rate", "--fc-drop-rate", dest="fc_drop_rate",
                        type=float, default=None)
    parser.add_argument("--drop", type=float, default=None)
    parser.add_argument("--attn_drop_rate", "--attn-drop-rate",
                        dest="attn_drop_rate", type=float, default=None)
    parser.add_argument("--drop_path", "--drop-path", dest="drop_path",
                        type=float, default=None)
    parser.add_argument("--init_scale", "--init-scale", dest="init_scale",
                        type=float, default=None)
    parser.add_argument("--use_mean_pooling", "--use-mean-pooling",
                        dest="use_mean_pooling",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--imagenet_default_mean_and_std",
                        "--imagenet-default-mean-and-std",
                        dest="imagenet_default_mean_and_std",
                        action=argparse.BooleanOptionalAction, default=None)

    # MiRA/FMP parameters. They must match the checkpoint architecture.
    parser.add_argument("--use_st_block", "--use-st-block", dest="use_st_block",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--add_intra_attention", "--add-intra-attention",
                        dest="add_intra_attention",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--add_fmp_attention", "--add-fmp-attention",
                        dest="add_fmp_attention",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fmp_num_last_layers", "--fmp-num-last-layers",
                        dest="fmp_num_last_layers", type=int, default=None)
    parser.add_argument("--fmp_no_use_ema", "--fmp-no-use-ema",
                        dest="fmp_no_use_ema",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fmp_use_residual", "--fmp-use-residual",
                        dest="fmp_use_residual",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_fmp_flashlite", "--use-fmp-flashlite",
                        dest="use_fmp_flashlite",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stats_mode", "--stats-mode", dest="stats_mode",
                        choices=("batch_ema", "batch", "instance"), default=None)

    # Checkpoint handling
    parser.add_argument("--checkpoint-key",
                        choices=("auto", "model", "model_ema", "module", "state_dict"),
                        default="auto")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_checkpoint(source: str) -> Any:
    if source.startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(source, map_location="cpu", check_hash=False)
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        # Fine-tuning checkpoints commonly contain an argparse.Namespace.
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def checkpoint_args(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, Mapping) or "args" not in checkpoint:
        return {}
    saved = checkpoint["args"]
    if isinstance(saved, Mapping):
        return dict(saved)
    if hasattr(saved, "__dict__"):
        return dict(vars(saved))
    return {}


def is_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in value.values()
    )


def extract_state_dict(checkpoint: Any, requested_key: str) -> Mapping[str, torch.Tensor]:
    if is_state_dict(checkpoint):
        return checkpoint
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The checkpoint is neither a state_dict nor a checkpoint dictionary")

    keys = ("model", "model_ema", "module", "state_dict") if requested_key == "auto" \
        else (requested_key,)
    for key in keys:
        candidate = checkpoint.get(key)
        if is_state_dict(candidate):
            print(f"Using checkpoint key: {key}")
            return candidate
    raise KeyError(
        f"No model state_dict found. Requested={requested_key!r}; "
        f"available keys={list(checkpoint.keys())}"
    )


def normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> "OrderedDict[str, torch.Tensor]":
    """Remove wrappers handled by the user's fine-tuning loader."""
    normalized: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    removable = ("module.", "backbone.", "encoder.")
    for old_key, value in state_dict.items():
        key = old_key
        changed = True
        while changed:
            changed = False
            for prefix in removable:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        normalized[key] = value
    return normalized


def infer_num_classes(state_dict: Mapping[str, torch.Tensor]) -> int | None:
    for key in ("head.weight", "fc.weight", "classifier.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[0])
    return None


def coalesce(cli_value: Any, saved: Mapping[str, Any], name: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    value = saved.get(name, default)
    return default if value is None else value


def resolve_config(
    args: argparse.Namespace,
    saved: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> SimpleNamespace:
    defaults = {
        "model": "vit_base_patch16_224",
        "input_size": 224,
        "short_side_size": 224,
        "num_frames": 16,
        "num_segments": 1,
        "sampling_rate": 4,
        "tubelet_size": 2,
        "fc_drop_rate": 0.0,
        "drop": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path": 0.1,
        "init_scale": 0.001,
        "use_mean_pooling": True,
        "imagenet_default_mean_and_std": True,
        "use_st_block": False,
        "add_intra_attention": False,
        "add_fmp_attention": False,
        "fmp_num_last_layers": 1,
        "fmp_no_use_ema": False,
        "fmp_use_residual": False,
        "use_fmp_flashlite": False,
        "stats_mode": "batch_ema",
        "test_num_segment": 5,
        "test_num_crop": 3,
    }
    names = tuple(defaults)
    cfg = {
        name: coalesce(getattr(args, name), saved, name, defaults[name])
        for name in names
    }
    inferred_classes = infer_num_classes(state_dict)
    saved_classes = saved.get("nb_classes")
    cfg["nb_classes"] = args.nb_classes or saved_classes or inferred_classes
    if cfg["nb_classes"] is None:
        raise ValueError("Could not infer nb_classes; pass --nb-classes explicitly")
    if inferred_classes is not None and int(cfg["nb_classes"]) != inferred_classes:
        raise ValueError(
            f"nb_classes={cfg['nb_classes']} but checkpoint head has {inferred_classes} outputs"
        )

    integer_fields = (
        "input_size", "short_side_size", "num_frames", "num_segments",
        "sampling_rate", "tubelet_size", "fmp_num_last_layers",
        "test_num_segment", "test_num_crop", "nb_classes",
    )
    for name in integer_fields:
        cfg[name] = int(cfg[name])
        if cfg[name] < 1:
            raise ValueError(f"{name} must be >= 1, got {cfg[name]}")
    if cfg["short_side_size"] < cfg["input_size"]:
        raise ValueError("short_side_size must be >= input_size")
    return SimpleNamespace(**cfg)


def build_model(cfg: SimpleNamespace) -> torch.nn.Module:
    clip_frames = cfg.num_frames * cfg.num_segments
    model = create_model(
        cfg.model,
        img_size=cfg.input_size,
        pretrained=False,
        num_classes=cfg.nb_classes,
        all_frames=clip_frames,
        tubelet_size=cfg.tubelet_size,
        fc_drop_rate=cfg.fc_drop_rate,
        drop_rate=cfg.drop,
        drop_path_rate=cfg.drop_path,
        attn_drop_rate=cfg.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=False,
        use_mean_pooling=cfg.use_mean_pooling,
        init_scale=cfg.init_scale,
        use_st_block=cfg.use_st_block,
        add_intra_attention=cfg.add_intra_attention,
        add_fmp_attention=cfg.add_fmp_attention,
        fmp_num_last_layers=cfg.fmp_num_last_layers,
        fmp_no_use_ema=cfg.fmp_no_use_ema,
        fmp_use_residual=cfg.fmp_use_residual,
        use_fmp_flashlite=cfg.use_fmp_flashlite,
        stats_mode=cfg.stats_mode,
    )
    return model


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def autocast_context(device: torch.device, requested: str):
    if requested == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype_name = "fp16" if requested == "auto" else requested
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def temporal_frame_ids(
    video_length: int,
    sampling_rate: int,
    clip_frames: int,
    num_views: int,
) -> List[np.ndarray]:
    if video_length < 1:
        raise ValueError("The input video contains no frames")
    sampled = np.arange(0, video_length, sampling_rate, dtype=np.int64)
    if sampled.size == 0:
        sampled = np.array([0], dtype=np.int64)
    if sampled.size < clip_frames:
        sampled = np.pad(sampled, (0, clip_frames - sampled.size), mode="edge")

    max_start = int(sampled.size - clip_frames)
    if num_views == 1:
        starts = np.array([max_start // 2], dtype=np.int64)
    else:
        # int(chunk * step) reproduces VideoMAE's deterministic test sampling.
        step = max_start / float(num_views - 1)
        starts = np.array([int(i * step) for i in range(num_views)], dtype=np.int64)
    return [sampled[start:start + clip_frames] for start in starts]


def decode_temporal_clips(
    video_path: Path,
    sampling_rate: int,
    clip_frames: int,
    num_views: int,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    try:
        reader = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
    except Exception as exc:
        raise RuntimeError(f"Decord could not open {video_path}: {exc}") from exc

    ids_per_view = temporal_frame_ids(
        len(reader), sampling_rate, clip_frames, num_views
    )
    # Decode only the frames used by the temporal views, rather than the whole video.
    flat_ids = np.concatenate(ids_per_view)
    decoded = reader.get_batch(flat_ids).asnumpy()
    decoded = decoded.reshape(num_views, clip_frames, *decoded.shape[1:])
    clips = [decoded[i] for i in range(num_views)]
    return clips, [ids.tolist() for ids in ids_per_view]


def spatial_views(
    clip: np.ndarray,
    resize_transform: Any,
    crop_size: int,
    num_crops: int,
) -> Iterable[Tuple[np.ndarray, int]]:
    resized = resize_transform(clip)
    if isinstance(resized, list):
        resized = np.stack(resized, axis=0)
    resized = np.asarray(resized)
    _, height, width, _ = resized.shape
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Resized video is {height}x{width}, smaller than crop {crop_size}"
        )

    if height >= width:
        max_offset = height - crop_size
        offsets = [max_offset // 2] if num_crops == 1 else [
            int(i * max_offset / (num_crops - 1)) for i in range(num_crops)
        ]
        x0 = (width - crop_size) // 2
        for crop_id, y0 in enumerate(offsets):
            yield resized[:, y0:y0 + crop_size, x0:x0 + crop_size, :], crop_id
    else:
        max_offset = width - crop_size
        offsets = [max_offset // 2] if num_crops == 1 else [
            int(i * max_offset / (num_crops - 1)) for i in range(num_crops)
        ]
        y0 = (height - crop_size) // 2
        for crop_id, x0 in enumerate(offsets):
            yield resized[:, y0:y0 + crop_size, x0:x0 + crop_size, :], crop_id


def make_views(
    clips: Sequence[np.ndarray], cfg: SimpleNamespace
) -> Tuple[List[torch.Tensor], List[Dict[str, int]]]:
    mean, std = (
        (IMAGENET_MEAN, IMAGENET_STD)
        if cfg.imagenet_default_mean_and_std
        else (INCEPTION_MEAN, INCEPTION_STD)
    )
    resize = video_transforms.Resize(cfg.short_side_size, interpolation="bilinear")
    to_tensor = volume_transforms.ClipToTensor()
    normalize = video_transforms.Normalize(mean=mean, std=std)

    views: List[torch.Tensor] = []
    metadata: List[Dict[str, int]] = []
    for temporal_id, clip in enumerate(clips):
        for crop, spatial_id in spatial_views(
            clip, resize, cfg.input_size, cfg.test_num_crop
        ):
            tensor = normalize(to_tensor(crop))  # C, T, H, W
            views.append(tensor)
            metadata.append({"temporal_view": temporal_id, "spatial_crop": spatial_id})
    return views, metadata


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "output", "pred"):
            if isinstance(output.get(key), torch.Tensor):
                return output[key]
    raise TypeError(f"Could not extract classification logits from {type(output).__name__}")


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    views: Sequence[torch.Tensor],
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
    aggregation: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    per_view_logits: List[torch.Tensor] = []
    for start in range(0, len(views), batch_size):
        batch = torch.stack(views[start:start + batch_size]).to(
            device, non_blocking=(device.type == "cuda")
        )
        with autocast_context(device, amp_dtype):
            logits = extract_logits(model(batch))
        per_view_logits.append(logits.float().cpu())

    logits = torch.cat(per_view_logits, dim=0)
    if aggregation == "probabilities":
        scores = logits.softmax(dim=-1).mean(dim=0)
    else:
        scores = logits.mean(dim=0).softmax(dim=-1)
    return scores, logits


def load_class_names(spec: str | None, num_classes: int) -> List[str]:
    if not spec:
        return [str(i) for i in range(num_classes)]
    path = Path(spec).expanduser()
    if path.is_file():
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                names = [str(payload.get(str(i), payload.get(i, i))) for i in range(num_classes)]
            elif isinstance(payload, list):
                names = [str(x) for x in payload]
            else:
                raise ValueError("Class-name JSON must be a list or index-to-name object")
        else:
            names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    else:
        names = [name.strip() for name in spec.split(",") if name.strip()]
    if len(names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, found {len(names)}")
    return names


def config_for_json(cfg: SimpleNamespace) -> Dict[str, Any]:
    return {key: value for key, value in vars(cfg).items()
            if isinstance(value, (str, int, float, bool, type(None)))}


def main() -> None:
    args = get_args()
    if args.view_batch_size < 1:
        raise ValueError("view_batch_size must be >= 1")

    checkpoint = load_checkpoint(args.checkpoint)
    saved_args = checkpoint_args(checkpoint)
    state_dict = normalize_state_dict_keys(
        extract_state_dict(checkpoint, args.checkpoint_key)
    )
    cfg = resolve_config(args, saved_args, state_dict)

    model = build_model(cfg)
    incompatible = model.load_state_dict(state_dict, strict=args.strict)
    if not args.strict:
        if incompatible.missing_keys:
            print("Missing checkpoint keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("Unexpected checkpoint keys:", incompatible.unexpected_keys)

    device = resolve_device(args.device)
    model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if cfg.stats_mode == "batch" and args.view_batch_size != 1:
        print(
            "Warning: stats_mode='batch' can make predictions depend on view batching. "
            "Use --view-batch-size 1 for view-independent single-video inference."
        )

    clip_frames = cfg.num_frames * cfg.num_segments
    clips, sampled_ids = decode_temporal_clips(
        args.video,
        sampling_rate=cfg.sampling_rate,
        clip_frames=clip_frames,
        num_views=cfg.test_num_segment,
    )
    views, view_metadata = make_views(clips, cfg)
    scores, view_logits = predict(
        model,
        views,
        device,
        batch_size=args.view_batch_size,
        amp_dtype=args.amp_dtype,
        aggregation=args.aggregation,
    )

    class_names = load_class_names(args.class_names, cfg.nb_classes)
    topk = min(max(1, args.topk), cfg.nb_classes)
    values, indices = scores.topk(topk)
    predictions = [
        {
            "rank": rank,
            "class_id": int(index),
            "class_name": class_names[int(index)],
            "probability": float(value),
        }
        for rank, (value, index) in enumerate(zip(values, indices), start=1)
    ]

    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}; views: {len(views)} "
          f"({cfg.test_num_segment} temporal x {cfg.test_num_crop} spatial)")
    for item in predictions:
        print(
            f"{item['rank']:>2}. [{item['class_id']}] {item['class_name']}: "
            f"{100.0 * item['probability']:.3f}%"
        )

    if args.output_json is not None:
        payload = {
            "video": str(args.video),
            "checkpoint": args.checkpoint,
            "device": str(device),
            "aggregation": args.aggregation,
            "config": config_for_json(cfg),
            "sampled_frame_indices": sampled_ids,
            "views": view_metadata,
            "num_view_logits": int(view_logits.shape[0]),
            "predictions": predictions,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
