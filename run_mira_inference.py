#!/usr/bin/env python3
"""Single-input inference for a fine-tuned VideoMAE/MiRA classifier.

Place this file in the root of the VideoMAE/MiRA repository.  It intentionally
uses the repository's ``video_transforms`` and ``volume_transforms`` so that
test-time preprocessing matches the fine-tuning pipeline.

Unlike ``run_videomae_vis.py``, this script loads ``modeling_finetune`` and a
classification checkpoint.  It does not run the masked-video decoder.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageOps
from timm.models import create_model

# Importing this module registers the VideoMAE classification models in timm.
import modeling_finetune  # noqa: F401
import video_transforms
import volume_transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

# Class-index order defined in emotion_labels.py.
DATASET_CLASS_NAMES = {
    "DFEW": ("Disgust", "Sad", "Neutral", "Surprise", "Angry", "Happy", "Fear"),
    "DFEW_crop": ("Disgust", "Sad", "Neutral", "Surprise", "Angry", "Happy", "Fear"),
    "FERV39k": ("Disgust", "Sad", "Neutral", "Surprise", "Angry", "Happy", "Fear"),
    "MAFW": (
        "Contempt", "Anxiety", "Neutral", "Sadness", "Anger", "Disgust",
        "Fear", "Surprise", "Happiness", "Helplessness", "Disappointment",
    ),
    "MAFW_crop": (
        "Contempt", "Anxiety", "Neutral", "Sadness", "Anger", "Disgust",
        "Fear", "Surprise", "Happiness", "Helplessness", "Disappointment",
    ),
    "AVCAFFE_V": ("Unpleasant", "Unsatisfied", "Neutral", "Pleased", "Pleasant"),
    "AVCAFFE_A": ("Excited", "Neutral", "Dull", "Wide-awake", "Calm"),
}

RAW_FRAME_DATASETS = {"DFEW_crop", "MAFW_crop", "FERV39k", "AVCAFFE_V", "AVCAFFE_A"}

# Architecture can be recovered safely from the classifier input dimension.
MODEL_SPECS = {
    768: ("vit_base_patch16_224", 12),
    1024: ("vit_large_patch16_224", 24),
    1280: ("vit_huge_patch16_224", 32),
}


def add_optional_bool(
    parser: argparse.ArgumentParser,
    *flags: str,
    dest: str,
    default: Optional[bool] = None,
) -> None:
    """Python 3.8-compatible positive/negative boolean override."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(*flags, dest=dest, action="store_true")
    negative_flags = tuple("--no-" + flag[2:] for flag in flags if flag.startswith("--"))
    group.add_argument(*negative_flags, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Single-input inference for a fine-tuned VideoMAE/MiRA classifier"
    )
    parser.add_argument("input", type=Path, help="input video file or raw-frame directory")
    parser.add_argument(
        "checkpoint",
        help="fine-tuned .pth/.pt file, DeepSpeed checkpoint directory, or URL",
    )

    # Output
    parser.add_argument("--class-names", default=None,
                        help="optional JSON/TXT label file or comma-separated override")
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
    parser.add_argument("--mode", choices=("vanilla", "exact", "flashlite"), default=None,
                        help="required when the checkpoint does not store training args")
    parser.add_argument("--data_set", "--data-set", dest="data_set", default=None,
                        choices=tuple(DATASET_CLASS_NAMES))
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
    add_optional_bool(
        parser, "--use_mean_pooling", "--use-mean-pooling",
        dest="use_mean_pooling",
    )
    add_optional_bool(
        parser, "--imagenet_default_mean_and_std",
        "--imagenet-default-mean-and-std",
        dest="imagenet_default_mean_and_std",
    )

    # MiRA/FMP parameters. They must match the checkpoint architecture.
    add_optional_bool(
        parser, "--use_st_block", "--use-st-block", dest="use_st_block"
    )
    add_optional_bool(
        parser, "--add_intra_attention", "--add-intra-attention",
        dest="add_intra_attention",
    )
    add_optional_bool(
        parser, "--add_fmp_attention", "--add-fmp-attention",
        dest="add_fmp_attention",
    )
    parser.add_argument("--fmp_num_last_layers", "--fmp-num-last-layers",
                        dest="fmp_num_last_layers", type=int, default=None)
    add_optional_bool(
        parser, "--fmp_no_use_ema", "--fmp-no-use-ema", dest="fmp_no_use_ema"
    )
    add_optional_bool(
        parser, "--fmp_use_residual", "--fmp-use-residual",
        dest="fmp_use_residual",
    )
    add_optional_bool(
        parser, "--use_fmp_flashlite", "--use-fmp-flashlite",
        dest="use_fmp_flashlite",
    )
    parser.add_argument("--stats_mode", "--stats-mode", dest="stats_mode",
                        choices=("batch_ema", "batch", "instance"), default=None)

    # Checkpoint handling
    parser.add_argument("--checkpoint-key",
                        choices=("auto", "model", "model_ema", "module", "state_dict"),
                        default="auto")
    add_optional_bool(parser, "--strict", dest="strict", default=True)
    return parser.parse_args()


def load_checkpoint(source: str) -> Any:
    if source.startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(source, map_location="cpu", check_hash=False)
    path = Path(source).expanduser()
    if path.is_dir():
        preferred = path / "mp_rank_00_model_states.pt"
        if preferred.is_file():
            path = preferred
        else:
            candidates = sorted(path.glob("*model_states.pt"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Could not uniquely resolve a DeepSpeed model checkpoint in {path}"
                )
            path = candidates[0]
        print(f"Resolved DeepSpeed checkpoint: {path}")
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


def infer_num_classes(state_dict: Mapping[str, torch.Tensor]) -> Optional[int]:
    for key in ("head.weight", "fc.weight", "classifier.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[0])
    return None


def infer_model_spec(
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Optional[str], Optional[int]]:
    for key in ("head.weight", "fc.weight", "classifier.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            spec = MODEL_SPECS.get(int(weight.shape[1]))
            if spec is not None:
                return spec
    return None, None


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
    inferred_model, inferred_depth = infer_model_spec(state_dict)
    model_name = args.model or saved.get("model") or inferred_model
    if model_name is None:
        raise ValueError("Could not infer the model architecture; pass --model explicitly")

    data_set = args.data_set or saved.get("data_set")
    if data_set is None:
        raise ValueError(
            "The checkpoint does not store data_set; pass --data-set explicitly"
        )

    if args.mode is not None:
        mode = args.mode
    elif "add_fmp_attention" in saved:
        if not bool(saved.get("add_fmp_attention")):
            mode = "vanilla"
        elif bool(saved.get("use_fmp_flashlite")):
            mode = "flashlite"
        else:
            mode = "exact"
    else:
        raise ValueError(
            "The checkpoint does not store MiRA settings; pass --mode "
            "{vanilla,exact,flashlite} explicitly"
        )

    default_sampling_rate = 4
    if data_set == "FERV39k" and model_name != "vit_huge_patch16_224":
        default_sampling_rate = 1

    mira_enabled = mode != "vanilla"
    defaults = {
        "model": model_name,
        "data_set": data_set,
        "input_size": 224,
        "short_side_size": 224,
        "num_frames": 16,
        "num_segments": 1,
        "sampling_rate": default_sampling_rate,
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
        "add_fmp_attention": mira_enabled,
        "fmp_num_last_layers": inferred_depth or 1,
        "fmp_no_use_ema": False,
        "fmp_use_residual": mira_enabled,
        "use_fmp_flashlite": mode == "flashlite",
        "stats_mode": "instance" if mira_enabled else "batch_ema",
        "test_num_segment": 5,
        "test_num_crop": 3,
    }
    names = tuple(defaults)
    cfg = {
        name: coalesce(getattr(args, name), saved, name, defaults[name])
        for name in names
    }

    # An explicit mode is a complete convenience override for parameter-free
    # MiRA settings absent from released DeepSpeed checkpoints.
    if args.mode is not None:
        if args.add_fmp_attention is None:
            cfg["add_fmp_attention"] = mira_enabled
        if args.use_fmp_flashlite is None:
            cfg["use_fmp_flashlite"] = mode == "flashlite"
        if args.fmp_use_residual is None:
            cfg["fmp_use_residual"] = mira_enabled
        if args.fmp_num_last_layers is None:
            cfg["fmp_num_last_layers"] = inferred_depth or cfg["fmp_num_last_layers"]
        if args.stats_mode is None:
            cfg["stats_mode"] = "instance" if mira_enabled else "batch_ema"
    cfg["mode"] = mode

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
    if data_set in RAW_FRAME_DATASETS and cfg["short_side_size"] != cfg["input_size"]:
        raise ValueError(
            "Raw-frame inference requires short_side_size == input_size, "
            "matching RawFrameClsDataset"
        )
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


def raw_frame_ids(
    total_frames: int,
    sampling_rate: int,
    clip_frames: int,
    num_views: int,
) -> List[np.ndarray]:
    """Match RawFrameClsDataset._choose_indices(..., mode='test')."""
    if total_frames < 1:
        raise ValueError("The input contains no frames")

    need = clip_frames * sampling_rate
    if total_frames < need:
        if total_frames >= clip_frames:
            ids = np.round(
                np.linspace(0, total_frames - 1, num=clip_frames)
            ).astype(np.int64)
        else:
            ids = np.array(
                list(range(total_frames))
                + [total_frames - 1] * (clip_frames - total_frames),
                dtype=np.int64,
            )
        return [ids.copy() for _ in range(num_views)]

    views: List[np.ndarray] = []
    for view_id in range(num_views):
        if num_views <= 1:
            center = total_frames // 2
            start = center - need // 2
        else:
            segment_center = (view_id + 0.5) * total_frames / float(num_views)
            start = int(round(segment_center - need / 2))
        start = max(0, min(total_frames - need, start))
        ids = np.arange(start, start + need, sampling_rate, dtype=np.int64)
        if ids.size < clip_frames:
            ids = np.pad(ids, (0, clip_frames - ids.size), mode="edge")
        views.append(np.clip(ids[:clip_frames], 0, total_frames - 1))
    return views


def natural_frame_key(path: Path) -> Tuple[float, str]:
    base = path.stem
    match = re.search(r"(\d+)$", base)
    return (int(match.group(1)) if match else float("inf"), base)


def list_frame_files(frame_dir: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png"}
    files = [
        path for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions and path.stat().st_size > 0
    ]
    return sorted(files, key=natural_frame_key)


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


def decode_raw_frame_clips(
    input_path: Path,
    sampling_rate: int,
    clip_frames: int,
    num_views: int,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    if input_path.is_dir():
        files = list_frame_files(input_path)
        if not files:
            raise FileNotFoundError(f"No JPG/PNG frames found in {input_path}")
        ids_per_view = raw_frame_ids(
            len(files), sampling_rate, clip_frames, num_views
        )
        clips: List[np.ndarray] = []
        for ids in ids_per_view:
            frames = []
            for index in ids:
                with Image.open(files[int(index)]) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    frames.append(np.asarray(image).copy())
            clips.append(np.stack(frames, axis=0))
        return clips, [ids.tolist() for ids in ids_per_view]

    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    try:
        reader = VideoReader(str(input_path), num_threads=1, ctx=cpu(0))
    except Exception as exc:
        raise RuntimeError(f"Decord could not open {input_path}: {exc}") from exc
    ids_per_view = raw_frame_ids(
        len(reader), sampling_rate, clip_frames, num_views
    )
    flat_ids = np.concatenate(ids_per_view)
    decoded = reader.get_batch(flat_ids).asnumpy()
    decoded = decoded.reshape(num_views, clip_frames, *decoded.shape[1:])
    return [decoded[i] for i in range(num_views)], [ids.tolist() for ids in ids_per_view]


def decode_input_clips(
    input_path: Path,
    cfg: SimpleNamespace,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    clip_frames = cfg.num_frames * cfg.num_segments
    if cfg.data_set in RAW_FRAME_DATASETS:
        return decode_raw_frame_clips(
            input_path,
            sampling_rate=cfg.sampling_rate,
            clip_frames=clip_frames,
            num_views=cfg.test_num_segment,
        )
    return decode_temporal_clips(
        input_path,
        sampling_rate=cfg.sampling_rate,
        clip_frames=clip_frames,
        num_views=cfg.test_num_segment,
    )


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


def square_resize_clip(clip: np.ndarray, size: int) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image)
    frames = []
    for frame in clip:
        image = Image.fromarray(frame).convert("RGB")
        image = image.resize((size, size), resampling.BILINEAR)
        frames.append(np.asarray(image))
    return np.stack(frames, axis=0)


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
        if cfg.data_set in RAW_FRAME_DATASETS:
            # RawFrameClsDataset square-resizes each temporal clip. Its spatial
            # crop loop repeats the same clip, so preserve that evaluation behavior.
            resized = square_resize_clip(clip, cfg.short_side_size)
            tensor = normalize(to_tensor(resized))
            for spatial_id in range(cfg.test_num_crop):
                views.append(tensor)
                metadata.append(
                    {"temporal_view": temporal_id, "spatial_crop": spatial_id}
                )
            continue

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


def load_class_names(
    spec: Optional[str],
    data_set: Optional[str],
    num_classes: int,
) -> List[str]:
    if not spec:
        dataset_names = DATASET_CLASS_NAMES.get(data_set)
        if dataset_names is not None:
            if len(dataset_names) != num_classes:
                raise ValueError(
                    f"Dataset {data_set} defines {len(dataset_names)} classes, "
                    f"but the checkpoint head has {num_classes} outputs"
                )
            return list(dataset_names)
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

    clips, sampled_ids = decode_input_clips(args.input, cfg)
    views, view_metadata = make_views(clips, cfg)
    scores, view_logits = predict(
        model,
        views,
        device,
        batch_size=args.view_batch_size,
        amp_dtype=args.amp_dtype,
        aggregation=args.aggregation,
    )

    class_names = load_class_names(args.class_names, cfg.data_set, cfg.nb_classes)
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

    print(f"Input: {args.input}")
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
            "input": str(args.input),
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
