import glob
import os
import random
from copy import deepcopy
import time
import subprocess
import gc
from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torchvision.transforms import Normalize, Compose
from tqdm import tqdm


class PatchDataset(data.Dataset):

    def __init__(
        self,
        # Directory containing the datasets
        data_dir: str,
        # Expected slices per fragment
        slice_depth: int = 4,
        # Size of an individual patch
        patch_size_x: int = 1028,
        patch_size_y: int = 256,
        # Image resize ratio
        resize_ratio: float = 1.0,
        # Training vs Testing mode
        train: bool = True,
        # Filenames of the images we'll use
        image_mask_filename='mask.png',
        image_labels_filename='inklabels.png',
        slices_dir_filename='surface_volume',
    ):
        print(f"Initializing CurriculumDataset")
        # Train mode also loads the labels
        self.train = train
        # Resize ratio reduces the size of the image
        self.resize_ratio = resize_ratio
        # Data will be B x slice_depth x patch_size_x x patch_size_y
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.patch_size_x_half = int(patch_size_x / 2)
        self.patch_size_y_half = int(patch_size_y / 2)
        self.slice_depth = slice_depth
        assert os.path.exists(
            data_dir), f"Data directory {data_dir} does not exist"
        # Open Mask image
        _image_mask_filepath = os.path.join(data_dir, image_mask_filename)
        _mask_img = Image.open(_image_mask_filepath).convert("1")
        # Get original size and resized size
        self.original_size = _mask_img.size
        self.resized_size = (
            int(self.original_size[0] * self.resize_ratio),
            int(self.original_size[1] * self.resize_ratio),
        )
        # Resize the mask
        # print(f"Mask original size: {original_size}")
        _mask_img = _mask_img.resize(self.resized_size, resample=Image.BILINEAR)
        # print(f"Mask resized size: {_mask_img.size}")
        _mask = torch.from_numpy(np.array(_mask_img)).to(torch.bool)
        # print(f"Mask tensor shape: {_mask.shape}")
        # print(f"Mask tensor dtype: {_mask.dtype}")
        if train:
            _image_labels_filepath = os.path.join(
                data_dir, image_labels_filename)
            _labels_img = Image.open(_image_labels_filepath).convert("1")
            # print(f"Labels original size: {original_size}")
            _labels_img = _labels_img.resize(
                self.resized_size, resample=Image.BILINEAR)
            # print(f"Labels resized size: {_labels_img.size}")
            self.labels = torch.from_numpy(
                np.array(_labels_img)).to(torch.bool)
            # print(f"Labels tensor shape: {self.labels.shape}")
            # print(f"Labels tensor dtype: {self.labels.dtype}")
        # Pre-allocate the entire fragment
        self.fragment = torch.zeros((
            self.slice_depth,
            self.resized_size[1],
            self.resized_size[0],
        ), dtype=torch.float32
        )
        # print(f"Fragment tensor shape: {self.fragment.shape}")
        # print(f"Fragment tensor dtype: {self.fragment.dtype}")
        # Open up slices
        _slice_dir = os.path.join(data_dir, slices_dir_filename)
        for i in tqdm(range(self.slice_depth)):
            _slice_filepath = os.path.join(_slice_dir, f"{i:02d}.tif")
            _slice_img = Image.open(_slice_filepath).convert('F')
            # print(f"Slice original size: {original_size}")
            _slice_img = _slice_img.resize(
                self.resized_size, resample=Image.BILINEAR)
            # print(f"Slice resized size: {_slice_img.size}")
            _slice = torch.from_numpy(np.array(_slice_img)/65535.0)
            # print(f"Slice tensor shape: {_slice.shape}")
            # print(f"Slice tensor dtype: {_slice.dtype}")
            self.fragment[i, :, :] = _slice

        print(f"Fragment tensor shape: {self.fragment.shape}")
        print(f"Fragment tensor dtype: {self.fragment.dtype}")
        print(f"Fragment tensor min: {self.fragment.min()}")
        print(f"Fragment tensor max: {self.fragment.max()}")
        # print(f"Fragment tensor mean: {self.fragment.mean()}")
        # print(f"Fragment tensor std: {self.fragment.std()}")

        # Get mean/std for fragment only on mask indices
        _fragment_mask = _mask.unsqueeze(0).expand(self.slice_depth, -1, -1)
        self.mean = self.fragment[_fragment_mask].mean()
        self.std = self.fragment[_fragment_mask].std()
        # print(f"Fragment tensor mean (no mask): {self.mean}")
        # print(f"Fragment tensor std (no mask): {self.std}")

        # Get indices where mask is 1
        self.mask_indices = torch.nonzero(_mask).to(torch.int32)
        # print(f"Mask indices shape: {self.mask_indices.shape}")
        # print(f"Mask indices dtype: {self.mask_indices.dtype}")

        # Pad the fragment with zeros based on patch size
        self.fragment = F.pad(
            self.fragment,
            (
                # Padding in Y
                self.patch_size_y_half, self.patch_size_y_half,
                # Padding in X
                self.patch_size_x_half, self.patch_size_x_half,
                # No padding on z
                0, 0,
            ),
            mode='constant',
            value=0,
        )

    def __len__(self):
        return self.mask_indices.shape[0]

    def __getitem__(self, index):

        # Get the x, y from the mask indices
        x, y = self.mask_indices[index]
        # print(f"Index: {index}, x: {x}, y: {y}")

        # Pre-allocate the patch
        patch = self.fragment[
                :,
                x: x + self.patch_size_x,
                y: y + self.patch_size_y,
        ]
        # print(f"Patch tensor shape: {patch.shape}")
        # print(f"Patch tensor dtype: {patch.dtype}")
        # print(f"Patch tensor min: {patch.min()}")
        # print(f"Patch tensor max: {patch.max()}")

        # Label is going to be the label of the center voxel
        if self.train:
            label = self.labels[
                x,
                y,
            ]
            return patch, label.unsqueeze(0).to(torch.float32)
        else:
            # If we're not training, we don't have labels
            return patch


# data_dir = '/home/tren/dev/ashenvenus/data/train/1'
# dataset = PatchDataset(
#     # Directory containing the dataset
#     data_dir=data_dir,
#     # Expected slices per fragment
#     slice_depth=4,
#     # Size of an individual patch
#     patch_size_x=256,
#     patch_size_y=64,
#     # Image resize ratio
#     resize_ratio=0.1,
#     # Training vs Testing mode
#     train=True,
# )
# img_transform = Compose([
#     Normalize(dataset.mean, dataset.std)
# ])
# indices_to_try = [
#     len(dataset) - 1,
#     0,
#     len(dataset) // 2,
#     random.randint(0, len(dataset)),
#     random.randint(0, len(dataset)),
#     random.randint(0, len(dataset)),
#     random.randint(0, len(dataset)),
# ]
# for i in indices_to_try:
#     print(f"Trying index {i} out of {len(dataset)}")
#     print(f"Mock Batch for {data_dir}")
#     patch, label = dataset[i]
#     print(f"Pre-Transform")
#     print(f"\t\t patch.shape = {patch.shape}")
#     print(f"\t\t patch.dtype = {patch.dtype}")
#     print(f"\t\t patch.min() = {patch.min()}")
#     print(f"\t\t patch.max() = {patch.max()}")
#     print(f"\t\t label value = {label.numpy()}")
#     print(f"\t\t label.shape = {label.shape}")
#     print(f"\t\t label.dtype = {label.dtype}")
    
#     patch = img_transform(patch)
#     print(f"Post-Transform")
#     print(f"\t\t patch.shape = {patch.shape}")
#     print(f"\t\t patch.dtype = {patch.dtype}")
#     print(f"\t\t patch.min() = {patch.min()}")
#     print(f"\t\t patch.max() = {patch.max()}")
#     print(f"\t\t label value = {label.numpy()}")
#     print(f"\t\t label.shape = {label.shape}")
#     print(f"\t\t label.dtype = {label.dtype}")
    
# del dataset

