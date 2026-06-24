import os
import glob
import re
import warnings
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageOps, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

import video_transforms as video_transforms
import volume_transforms as volume_transforms
from emotion_labels import class_label_map
from loader import get_image_smart_loader
from random_erasing import RandomErasing

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _make_video_id(sample: str) -> str:
    return os.path.normpath(sample)


class VideoClsDataset(Dataset):
    """Load video classification data from video files."""

    def __init__(
        self,
        dataset_name,
        anno_path,
        data_path,
        mode="train",
        class_label_map_fn=False,
        clip_len=8,
        frame_sample_rate=2,
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        num_segment=1,
        num_crop=1,
        test_num_segment=10,
        test_num_crop=3,
        args=None,
    ):
        self.dataset_name = dataset_name
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.class_label_map_fn = class_label_map_fn
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False

        if self.mode == "train":
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        if VideoReader is None:
            raise ImportError("Unable to import decord, which is required to read videos.")

        data_mode = "test" if self.mode == "validation" else self.mode

        cleaned = pd.read_csv(self.anno_path)
        cleaned = cleaned[(cleaned["data_split"] == data_mode) | (cleaned["data_split"] == str(data_mode))]
        cleaned = cleaned[(cleaned["bool_file"] == True) | (cleaned["bool_file"] == "True")]
        cleaned = cleaned[cleaned["annotation"] != False]
        cleaned = cleaned[cleaned["annotation"] != "False"]
        cleaned = cleaned[cleaned["annotation"] != None]
        cleaned = cleaned[cleaned["annotation"] != -1]
        cleaned = cleaned[cleaned["annotation"] != str(-1)]

        self.dataset_samples = list(cleaned["filename"].values)
        self.label_array = list(cleaned["annotation"].values)

        if self.mode == "validation":
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        elif self.mode == "test":
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=short_side_size, interpolation="bilinear")
            ])

            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []

            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        self.test_label_array.append(self.label_array[idx])
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == "train":
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(f"video {sample} not correctly loaded during training")
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list, label_list, index_list = [], [], []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}

            buffer = self._aug_frame(buffer, args)

            numeric_label = self.class_label_map_fn(self.label_array[index])
            if numeric_label == -1:
                raise AttributeError(
                    "'{}' input has no numeric label, output: '{}'".format(sample, self.label_array[index])
                )
            return buffer, numeric_label, index, {}

        elif self.mode == "validation":
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                raise RuntimeError(f"Video not correctly loaded during validation: {sample}")

            buffer = self.data_transform(buffer)

            numeric_label = self.class_label_map_fn(self.label_array[index])
            if numeric_label == -1:
                raise AttributeError(
                    "'{}' input has no numeric label, output: '{}'".format(sample, self.label_array[index])
                )

            video_id = _make_video_id(sample)
            return buffer, numeric_label, video_id

        elif self.mode == "test":
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            if len(buffer) == 0:
                raise RuntimeError(
                    f"Video not found during testing: sample={sample}, chunk={chunk_nb}, split={split_nb}"
                )

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = (
                1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size)
                / (self.test_num_crop - 1)
            )
            temporal_step = max(
                1.0 * (buffer.shape[0] - self.clip_len) / (self.test_num_segment - 1), 0
            )
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)

            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[
                    temporal_start:temporal_start + self.clip_len,
                    spatial_start:spatial_start + self.short_side_size,
                    :,
                    :,
                ]
            else:
                buffer = buffer[
                    temporal_start:temporal_start + self.clip_len,
                    :,
                    spatial_start:spatial_start + self.short_side_size,
                    :,
                ]

            buffer = self.data_transform(buffer)

            numeric_label = self.class_label_map_fn(self.test_label_array[index])
            if numeric_label == -1:
                raise AttributeError(
                    "'{}' input has no numeric label, output: '{}'".format(sample, self.test_label_array[index])
                )

            video_id = _make_video_id(sample)
            return buffer, numeric_label, video_id, chunk_nb, split_nb

        else:
            raise NameError(f"mode {self.mode} unknown")

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)
        buffer = buffer.permute(0, 2, 3, 1)

        buffer = tensor_normalize(
            buffer,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        buffer = buffer.permute(3, 0, 1, 2)

        scl, asp = ([0.08, 1.0], [0.75, 1.3333])

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == "SSV2" else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        fname = sample

        if not os.path.exists(fname):
            return []

        if os.path.getsize(fname) < 1 * 1024:
            print("SKIP:", fname, "-", os.path.getsize(fname))
            return []

        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(
                    fname,
                    width=self.new_width,
                    height=self.new_height,
                    num_threads=1,
                    ctx=cpu(0),
                )
        except Exception:
            print("video cannot be loaded by decord:", fname)
            return []

        if self.mode == "test":
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            vr.seek(0)
            return vr.get_batch(all_index).asnumpy()

        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate(
                    (
                        index,
                        np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len,
                    )
                )
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)

            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[:: int(sample_rate_scale)]
        vr.seek(0)
        return vr.get_batch(all_index).asnumpy()

    def __len__(self):
        return len(self.dataset_samples) if self.mode != "test" else len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    assert spatial_idx in [-1, 0, 1, 2]

    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )

        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)

    else:
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(frames, min_scale, max_scale)
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)

    return frames


def tensor_normalize(tensor, mean, std):
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0

    if isinstance(mean, list):
        mean = torch.tensor(mean)
    if isinstance(std, list):
        std = torch.tensor(std)

    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _letterbox_pil(img: Image.Image, out_size: int, pad_color=(0, 0, 0)) -> Image.Image:
    assert isinstance(img, Image.Image)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if w == out_size and h == out_size:
        return img

    scale = min(out_size / max(1, w), out_size / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (out_size, out_size), pad_color)
    left, top = (out_size - nw) // 2, (out_size - nh) // 2
    canvas.paste(img, (left, top))
    return canvas


class RawFrameClsDataset(Dataset):
    """Load classification data from raw frame folders."""

    def __init__(
        self,
        dataset_name,
        anno_path,
        data_path,
        mode="train",
        class_label_map_fn=False,
        clip_len=8,
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        frame_sample_rate=None,
        exact_clip_dataset=False,
        num_segment=1,
        num_crop=1,
        test_num_segment=10,
        test_num_crop=3,
        filename_tmpl="{:01}.jpg",
        start_idx=1,
        args=None,
    ):
        self.dataset_name = dataset_name
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.class_label_map_fn = class_label_map_fn
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.filename_tmpl = filename_tmpl
        self.start_idx = start_idx
        self.args = args
        self.aug = False
        self.rand_erase = False

        self.frame_sample_rate = 1 if frame_sample_rate in (None, 0) else int(frame_sample_rate)
        self.exact_clip_dataset = exact_clip_dataset

        if self.mode == "train":
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self.image_loader = get_image_smart_loader()

        data_mode = "test" if self.mode == "validation" else self.mode

        cleaned = pd.read_csv(self.anno_path)
        cleaned = cleaned[(cleaned["data_split"] == data_mode) | (cleaned["data_split"] == str(data_mode))]
        cleaned = cleaned[(cleaned["bool_file"] == True) | (cleaned["bool_file"] == "True")]
        cleaned = cleaned[cleaned["annotation"] != False]
        cleaned = cleaned[cleaned["annotation"] != "False"]
        cleaned = cleaned[cleaned["annotation"] != None]
        cleaned = cleaned[cleaned["annotation"] != -1]
        cleaned = cleaned[cleaned["annotation"] != str(-1)]

        self.dataset_samples = list(cleaned["filename"].values)
        self.label_array = list(cleaned["annotation"].values)
        self.total_frames = []

        if dataset_name == "DFEW_crop":
            for i in range(len(self.dataset_samples)):
                sample = self.dataset_samples[i]
                temp = sample.split("/")
                sample_name = temp[-1].split(".")[0]
                video_frame_folder = "%0*d" % (5, int(sample_name))

                temp[temp.index("original")] = self.data_path
                sample_dir = "/".join(temp[:-2])
                sample_dir = os.path.join(sample_dir, video_frame_folder)

                sample_frames = glob.glob(sample_dir + "/*.jpg")
                if len(sample_frames) > 0:
                    self.dataset_samples[i] = sample_dir
                    self.total_frames.append(len(sample_frames))

        if dataset_name == "MAFW_crop":
            for i in range(len(self.dataset_samples)):
                sample = self.dataset_samples[i]
                temp = sample.split("/")
                sample_name = temp[-1].split(".")[0]
                video_frame_folder = "%0*d" % (5, int(sample_name))

                temp[temp.index("clips")] = "frames"
                sample_dir = "/".join(temp[:-1])
                sample_dir = os.path.join(sample_dir, video_frame_folder)

                sample_frames = glob.glob(sample_dir + "/*.png")
                if len(sample_frames) > 0:
                    self.dataset_samples[i] = sample_dir
                    self.total_frames.append(len(sample_frames))

        if dataset_name == "FERV39k":
            for i in range(len(self.dataset_samples)):
                sample_dir = self.dataset_samples[i]
                sample_frames = sorted(glob.glob(sample_dir + "/*.jpg"))
                if len(sample_frames) > 0:
                    self.dataset_samples[i] = sample_dir
                    self.total_frames.append(len(sample_frames))

        if dataset_name == "AVCAFFE_V" or dataset_name == "AVCAFFE_A":
            _, self.emotion_mode = dataset_name.split("_")
            new_samples, new_labels = [], []
            add_path = ["data", "face_crops"]

            for i in range(len(self.dataset_samples)):
                sample = self.dataset_samples[i]
                temp = sample.split("/")
                temp_data_root = temp[:3] + add_path
                temp_person_id = temp[-2]
                temp_task_id = temp[-1].split(".")[0]
                temp_folder = str(temp_person_id + "_" + temp_task_id)
                video_frame_folder = ["./"] + temp_data_root + [temp_folder]
                video_frame_folder = Path(os.path.join(*video_frame_folder))

                label_match = re.search(rf"{self.emotion_mode}:([^/]+)", self.label_array[i])
                new_label = label_match.group(1)
                clip_folders = [f.name for f in video_frame_folder.iterdir() if f.is_dir()]

                for j in range(len(clip_folders)):
                    sample_dir = os.path.join(
                        video_frame_folder,
                        clip_folders[j],
                        os.path.join(*["shorter_segments_face", temp_person_id, temp_task_id]),
                        clip_folders[j],
                    )
                    sample_frames = glob.glob(sample_dir + "/*.jpg")
                    if len(sample_frames) > 0:
                        new_samples.append(sample_dir)
                        new_labels.append(new_label)
                        self.total_frames.append(len(sample_frames))
                    else:
                        print("[DROP-AVCAFFE]", sample_dir)

            self.dataset_samples = new_samples
            self.label_array = new_labels

        if self.mode == "validation":
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation="bilinear"),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        elif self.mode == "test":
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=short_side_size, interpolation="bilinear")
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self.test_seg = []
            self.test_dataset = []
            self.test_total_frames = []
            self.test_label_array = []

            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        self.test_seg.append((ck, cp))
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_total_frames.append(self.total_frames[idx])
                        self.test_label_array.append(self.label_array[idx])

    def __getitem__(self, index):
        if self.mode == "train":
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_frame(sample, total_frame, sample_rate_scale=scale_t)

            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(f"video {sample} not correctly loaded during training")
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    total_frame = self.total_frames[index]
                    buffer = self.load_frame(sample, total_frame, sample_rate_scale=scale_t)

            if isinstance(buffer, list):
                side = getattr(self, "short_side_size", self.crop_size)
                buffer = [
                    np.array(Image.fromarray(fr).resize((side, side), Image.BILINEAR))
                    for fr in buffer
                ]

            if args.num_sample > 1:
                frame_list, label_list, index_list = [], [], []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(self.class_label_map_fn(label))
                    index_list.append(index)
                return frame_list, label_list, index_list, {}

            buffer = self._aug_frame(buffer, args)

            numeric_label = self.class_label_map_fn(self.label_array[index])
            if numeric_label == -1:
                raise AttributeError(
                    "'{}' input has no numeric label, output: '{}'".format(sample, self.label_array[index])
                )
            return buffer, numeric_label, index, {}

        elif self.mode == "validation":
            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]

            buffer = self.load_frame(sample, total_frame)
            if len(buffer) == 0:
                raise RuntimeError(f"Video not correctly loaded during validation: {sample}")

            if isinstance(buffer, list):
                side = self.short_side_size
                buffer = [
                    np.array(Image.fromarray(fr).resize((side, side), Image.BILINEAR))
                    for fr in buffer
                ]
                buffer = np.stack(buffer, 0)

            buffer = self.data_transform(buffer)

            numeric_label = self.class_label_map_fn(self.label_array[index])
            if numeric_label == -1:
                raise AttributeError(
                    "'{}' input has no numeric label, output: '{}'".format(sample, self.label_array[index])
                )

            video_id = _make_video_id(sample)
            return buffer, numeric_label, video_id

        elif self.mode == "test":
            sample = self.test_dataset[index]
            total_frame = self.test_total_frames[index]
            chunk_nb, split_nb = self.test_seg[index]

            self._cur_test_seg_idx = int(chunk_nb)
            self._cur_test_seg_count = int(self.test_num_segment)

            buffer = self.load_frame(sample, total_frame)
            if len(buffer) == 0:
                raise RuntimeError(
                    f"Video not found during testing: sample={sample}, chunk={chunk_nb}, split={split_nb}"
                )

            S = self.short_side_size
            resize = transforms.Resize((S, S), interpolation=Image.BILINEAR)

            if isinstance(buffer, list):
                frames = []
                for im in buffer:
                    if not isinstance(im, Image.Image):
                        im = Image.fromarray(im)
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im = resize(im)
                    frames.append(np.asarray(im))
                buffer = np.stack(frames, axis=0)
            else:
                frames = []
                for fr in list(buffer):
                    im = Image.fromarray(fr).convert("RGB")
                    im = resize(im)
                    frames.append(np.asarray(im))
                buffer = np.stack(frames, axis=0)

            buffer = self.data_transform(buffer)

            numeric_label = self.class_label_map_fn(self.test_label_array[index])
            if numeric_label == -1:
                raise AttributeError(
                    "'{}' input has no numeric label, output: '{}'".format(sample, self.test_label_array[index])
                )

            video_id = _make_video_id(sample)
            return buffer, numeric_label, video_id, chunk_nb, split_nb

        else:
            raise NameError(f"mode {self.mode} unknown")

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)
        buffer = buffer.permute(0, 2, 3, 1)

        buffer = tensor_normalize(
            buffer,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        buffer = buffer.permute(3, 0, 1, 2)

        scl, asp = ([0.08, 1.0], [0.75, 1.3333])

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == "SSV2" else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def _list_frame_files(self, frame_dir):
        pats = []
        for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
            pats += glob.glob(os.path.join(frame_dir, f"*.{ext}"))

        def _nat_key(p):
            base = os.path.splitext(os.path.basename(p))[0]
            m = re.search(r"(\d+)$", base)
            return (int(m.group(1)) if m else float("inf"), base)

        files = sorted(pats, key=_nat_key)
        files = [p for p in files if os.path.isfile(p) and os.path.getsize(p) > 0]
        return files

    def _load_by_indices(self, files, indices):
        imgs = []
        total = len(files)

        for k in indices:
            k = int(max(0, min(int(k), total - 1)))
            path = files[k]
            img = self.image_loader(path)
            if img is None:
                continue
            imgs.append(img)

        return imgs

    def _choose_indices(self, total_frames, mode, seg_idx=None, seg_count=None, fsr=None):
        T = int(self.clip_len)
        fsr = int(fsr or getattr(self, "frame_sample_rate", 1))
        need = T * fsr

        if total_frames <= 0:
            return np.array([], dtype=np.int64)

        if getattr(self, "exact_clip_dataset", False):
            if total_frames >= T:
                return np.arange(0, T, dtype=np.int64)
            base = list(range(total_frames))
            pad = [total_frames - 1] * (T - total_frames)
            return np.array(base + pad, dtype=np.int64)

        if total_frames < need:
            if total_frames >= T:
                idx = np.linspace(0, total_frames - 1, num=T)
                idx = np.round(idx).astype(np.int64)
            else:
                base = list(range(total_frames))
                pad = [total_frames - 1] * (T - total_frames)
                idx = np.array(base + pad, dtype=np.int64)
            return np.clip(idx, 0, total_frames - 1)

        if mode == "train":
            start = np.random.randint(0, total_frames - need + 1)
        elif mode in ["validation", "val"]:
            center = total_frames // 2
            half = need // 2
            start = max(0, min(total_frames - need, center - half))
        else:
            if seg_idx is None or not seg_count or seg_count <= 1:
                center = total_frames // 2
                half = need // 2
                start = max(0, min(total_frames - need, center - half))
            else:
                seg_len = total_frames / float(seg_count)
                seg_cen = (seg_idx + 0.5) * seg_len
                start = int(round(seg_cen - need / 2))
                start = max(0, min(total_frames - need, start))

        raw = np.arange(start, start + need, fsr, dtype=np.int64)
        if len(raw) >= T:
            idx = raw[:T]
        else:
            pad = np.array([raw[-1]] * (T - len(raw)), dtype=np.int64)
            idx = np.concatenate([raw, pad])

        return np.clip(idx, 0, total_frames - 1)

    def load_frame(self, sample, num_frames, sample_rate_scale=1):
        files = self._list_frame_files(sample)
        total = len(files)
        if total == 0:
            return []

        seg_idx = getattr(self, "_cur_test_seg_idx", None) if self.mode == "test" else None
        seg_count = getattr(self, "_cur_test_seg_count", None) if self.mode == "test" else None

        base_fsr = int(getattr(self, "frame_sample_rate", 1))
        sc = sample_rate_scale if sample_rate_scale else 1
        eff_fsr = max(1, int(round(base_fsr * sc)))

        indices = self._choose_indices(
            total_frames=total,
            mode=self.mode,
            seg_idx=seg_idx,
            seg_count=seg_count,
            fsr=eff_fsr,
        )

        if len(indices) == 0:
            return []

        return self._load_by_indices(files, indices)

    def __len__(self):
        return len(self.dataset_samples) if self.mode != "test" else len(self.test_dataset)


def _natkey(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]


def _letterbox(img: Image.Image, out_size: int, pad_color=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    if w == out_size and h == out_size:
        return img

    scale = min(out_size / max(1, w), out_size / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BICUBIC)

    canvas = Image.new("RGB", (out_size, out_size), pad_color)
    left = (out_size - nw) // 2
    top = (out_size - nh) // 2
    canvas.paste(img, (left, top))
    return canvas


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    assert spatial_idx in [-1, 0, 1, 2]

    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )

        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)

    else:
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(frames, min_scale, max_scale)
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)

    return frames


def tensor_normalize(tensor, mean, std):
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0

    if isinstance(mean, list):
        mean = torch.tensor(mean)
    if isinstance(std, list):
        std = torch.tensor(std)

    tensor = tensor - mean
    tensor = tensor / std
    return tensor


class VideoMAE(torch.utils.data.Dataset):
    """Load pretraining data for VideoMAE."""

    def __init__(
        self,
        root,
        setting,
        train=True,
        test_mode=False,
        name_pattern="img_%05d.jpg",
        video_ext="mp4",
        is_color=True,
        modality="rgb",
        num_segments=1,
        num_crop=1,
        new_length=1,
        new_step=1,
        transform=None,
        temporal_jitter=False,
        video_loader=False,
        use_decord=False,
        lazy_init=False,
        aug_paired_input=True,
        debug_mode=False,
    ):
        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init
        self.debug_mode = debug_mode

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting, debug_mode=self.debug_mode)
            if len(self.clips) == 0:
                raise RuntimeError(
                    "Found 0 video clips in subfolders of: " + str(root) + "\n"
                    "Check your data directory (opt.data-dir)."
                )

    def __getitem__(self, index):
        directory, target = self.clips[index]

        if self.video_loader:
            if "." in directory.split("/")[-1]:
                video_name = directory
            else:
                video_name = f"{directory}.{self.video_ext}"

            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)

        segment_indices, skip_offsets = self._sample_train_indices(duration)
        images = self._video_TSN_decord_batch_loader(
            directory,
            decord_vr,
            duration,
            segment_indices,
            skip_offsets,
        )

        process_data, mask = self.transform((images, None))
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0, 1)

        return process_data, mask

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting, debug_mode=False):
        print("Dataset is re-arranging for VideoMAE-Dataset Class ...")
        if not os.path.exists(setting):
            raise RuntimeError(f"Setting file {setting} doesn't exist.")

        df = pd.read_csv(setting)
        num_files_origin = len(df)
        df = df[df["bool_file"] == True]

        if self.train and not self.test_mode:
            df = df[(df["data_split"] == "train") | (df["data_split"] == False) | (df["data_split"] == "False")]

        clips = []
        for _, df_row in df.iterrows():
            clip_path = df_row["filename"]
            target = int(class_label_map(df_row["annotation"]))
            clips.append((clip_path, target))

        num_files_extracted = len(clips)
        print("The data rearrangement process is complete.")
        print(
            "The number of original video files: {} --> The number of extracted video files: {}".format(
                num_files_origin,
                num_files_extracted,
            )
        )
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(average_duration, size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(num_frames - self.skip_length + 1, size=self.num_segments)
            )
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.skip_length // self.new_step, dtype=int)

        return offsets + 1, skip_offsets

    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []

        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step

        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [
                Image.fromarray(video_data[vid, :, :, :]).convert("RGB")
                for vid, _ in enumerate(frame_id_list)
            ]
        except Exception:
            raise RuntimeError(
                "Error occured in reading frames {} from video {} of duration {}.".format(
                    frame_id_list,
                    directory,
                    duration,
                )
            )

        return sampled_list