import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
# from kinetics import VideoClsDataset, VideoMAE
# from ssv2 import SSVideoClsDataset
from facialvideo_datasets import VideoClsDataset, VideoMAE, RawFrameClsDataset
from emotion_labels import DFEW_class_label_map, MAFW_class_label_map, AVCAFFE_V_class_label_map, AVCAFFE_A_class_label_map, FERV39k_class_label_map


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        # GroupMultiScaleCrop: BASE [1, .875, .75, .66], LARGE/HUGE [1, .875, .75, .66, .60, .60, .55, .55]
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66, .60, .60, .55, .55])  
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        debug_mode=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    anno_path = args.data_path
    mode = None
    if is_train is True:
        mode = 'train'
    elif test_mode is True:
        mode = 'test'
    else:  
        mode = 'validation'

    if args.data_set == 'DFEW':
        dataset = VideoClsDataset(
            dataset_name = 'DFEW',
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            class_label_map_fn=DFEW_class_label_map,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 7
    
    elif args.data_set == 'MAFW':
        dataset = VideoClsDataset(
            dataset_name = 'MAFW',
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            class_label_map_fn=MAFW_class_label_map,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 11
    
    elif args.data_set == 'DFEW_crop':
        dataset = RawFrameClsDataset(
            dataset_name = 'DFEW_crop',
            anno_path=anno_path,
            data_path='clip_224x224_16f',
            mode=mode,
            class_label_map_fn=DFEW_class_label_map,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            # file_ext='jpg',
            args=args)
        nb_classes = 7
        print ("**CROP DATASET")
    
    elif args.data_set == 'MAFW_crop':
        dataset = RawFrameClsDataset(
            dataset_name = 'MAFW_crop',
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            class_label_map_fn=MAFW_class_label_map,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            # file_ext='png',
            args=args)
        nb_classes = 11
        print ("**CROP DATASET")
    
    elif args.data_set == 'AVCAFFE_V':
        dataset = RawFrameClsDataset(
            dataset_name = 'AVCAFFE_V',
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            class_label_map_fn=AVCAFFE_V_class_label_map,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            # file_ext='png',
            args=args)
        nb_classes = 5
        print ("**CROP DATASET")

    elif args.data_set == 'AVCAFFE_A':
        dataset = RawFrameClsDataset(
            dataset_name = 'AVCAFFE_A',
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            class_label_map_fn=AVCAFFE_A_class_label_map,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            # file_ext='jpg',
            args=args)
        nb_classes = 5
        print ("**CROP DATASET")
    
    elif args.data_set == 'FERV39k':
        dataset = RawFrameClsDataset(
            dataset_name = 'FERV39k',
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            class_label_map_fn=FERV39k_class_label_map,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            # file_ext='jpg',
            args=args)
        nb_classes = 7
        print ("**CROP DATASET")

    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
