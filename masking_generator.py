import numpy as np

# class TubeMaskingGenerator:
#     def __init__(self, input_size, mask_ratio):
#         self.frames, self.height, self.width = input_size
#         self.num_patches_per_frame =  self.height * self.width
#         self.total_patches = self.frames * self.num_patches_per_frame 
#         self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
#         self.total_masks = self.frames * self.num_masks_per_frame

#         self.block_size = 5
#         self.num_patch_per_block = 5 # 3, 5, 7, ... 
#         self.repeat_masking = 2

#     def __repr__(self):
#         repr_str = "Maks: total patches {}, mask patches {}".format(
#             self.total_patches, self.total_masks
#         )
#         return repr_str

#     def fair_blockwise_masking(self, prev_mask):
#         h = self.height
#         w = self.width
#         bs = self.block_size
#         pad_h = (bs - h % bs) % bs
#         pad_w = (bs - w % bs) % bs
#         ref_mask = np.pad(prev_mask, ((0, pad_h), (0, pad_w)), constant_values=1)
#         mask = np.ones(ref_mask.shape, dtype=int)
#         # select visible patch locations
#         for i in range(0, mask.shape[0], bs):
#             for j in range(0, mask.shape[1], bs):
#                 # 50% probability of masking in a block
#                 if np.random.choice([0, 1], 1): 
#                     ref_block = ref_mask[i:i+bs, j:j+bs]
#                     block = mask[i:i+bs, j:j+bs]
#                     ys, xs = np.where(ref_block == 1) 
#                     idx = np.random.choice(len(ys), self.num_patch_per_block)
#                     y, x = ys[idx], xs[idx]
#                     block[y, x] = 0
#                     mask[i:i+bs, j:j+bs] = block
#         mask = mask[:h, :w]
#         return mask

#     def __call__(self, aug_random_mask=False):
#         # mask = 1 / visble = 0
#         if aug_random_mask: 
#             masks = []
#             accum_prev_mask = np.ones((self.height, self.width), dtype=int)
#             for i in range(self.repeat_masking):
#                 mask = self.fair_blockwise_masking(prev_mask=accum_prev_mask)
#                 masks.append(np.tile(mask.flatten(), (self.frames,1)).flatten().astype(np.float64))
#                 accum_prev_mask = accum_prev_mask*mask
#             return np.vstack(masks)
        
#         masks = []
#         for i in range(self.repeat_masking):
#             mask_per_frame = np.hstack([
#                 np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
#                 np.ones(self.num_masks_per_frame),
#                 ])
#             np.random.shuffle(mask_per_frame)
#             mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
#             masks.append(mask)
#         return np.vstack(masks)

#         ## original single mask
#         # mask_per_frame = np.hstack([
#         #     np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
#         #     np.ones(self.num_masks_per_frame),
#         #     ])
#         # np.random.shuffle(mask_per_frame)
#         # mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
#          return mask 


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 


