import glob
import logging
import os

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image as Image
import torch.utils.data as data
from torchvision.transforms import ToTensor
from tqdm import tqdm
import random
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_image(image_filepath: str = 'image.png', viz: bool = False) -> torch.Tensor:
    _image = Image.open(image_filepath).convert('L')
    _pt = ToTensor()(_image)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"Loading {image_filepath}")
        log.debug(f"\tImage shape: {_pt.shape}")
        log.debug(f"\tImage type: {_pt.dtype}")
        log.debug(f"\tImage min: {_pt.min()}")
        log.debug(f"\tImage max: {_pt.max()}")
        log.debug(f"\tImage mean: {_pt.mean()}")
        log.debug(f"\tImage std: {_pt.std()}")
    if viz:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.title(image_filepath)
        # Move the channel axis to the last position
        _pt_np = _pt.numpy()
        # Remove the channel dimension for grayscale
        _pt_np = np.squeeze(_pt_np)
        plt.imshow(_pt_np, cmap='gray')
        plt.subplot(122)
        plt.title('Histogram of Greyscale Pixel Values')
        plt.hist(_pt_np.flatten(), bins=256, range=(0, 1))
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
    return _pt


class FragmentPatchesDataset(data.Dataset):

    def __init__(
        self,
        # Directory containing the dataset
        data_dir: str,
        # Filenames of the images we'll use
        image_mask_filename='mask.png',
        image_labels_filename='inklabels.png',
        image_ir_filename='ir.png',
        slices_dir_filename='surface_volume',
        # Expected slices per fragment
        expected_slice_count: int = 65,
        # Size of an individual patch
        patch_size_x: int = 256, #1028,
        patch_size_y: int = 256, #1028,
        patch_size_z: int = 32, #32,
        # Visualization toggle for debugging
        viz: bool = False,
        # Training vs Testing mode
        train: bool = True,
    ):
        self.train = train
        self.viz = viz
        log.info(f"Initializing FragmentDataset with data_dir={data_dir}")
        # Verify paths and directories for images and metadata
        self.data_dir = data_dir
        assert os.path.exists(
            data_dir), f"Data directory {data_dir} does not exist"
        self.image_mask_filename = image_mask_filename
        self.image_mask_filepath = os.path.join(
            data_dir, self.image_mask_filename)
        assert os.path.exists(
            self.image_mask_filepath), f"Mask file {self.image_mask_filepath} does not exist"
        self.image_labels_filename = image_labels_filename
        self.image_labels_filepath = os.path.join(
            data_dir, self.image_labels_filename)
        self.slices_dir = os.path.join(data_dir, slices_dir_filename)
        assert os.path.exists(
            self.slices_dir), f"Slices directory {self.slices_dir} does not exist"
        if self.train:
            assert os.path.exists(
                self.image_labels_filepath), f"Labels file {self.image_labels_filepath} does not exist"
            self.image_ir_filename = image_ir_filename
            self.image_ir_filepath = os.path.join(
                data_dir, self.image_ir_filename)
            assert os.path.exists(
                self.image_ir_filepath), f"IR file {self.image_ir_filepath} does not exist"

        # Load the meta data (mask, labels, and IR images)
        self.mask = load_image(self.image_mask_filepath, viz=self.viz)
        if self.train:
            self.labels = load_image(self.image_labels_filepath, viz=self.viz)
            self.ir = load_image(self.image_ir_filepath, viz=self.viz)

        # Assert that there are the correct amount of slices
        self.expected_slice_count = expected_slice_count
        if log.isEnabledFor(logging.DEBUG):
            self.actual_slice_count = len(
                glob.glob(os.path.join(self.slices_dir, '*.tif')))
            assert self.actual_slice_count == self.expected_slice_count, f"Expected {self.expected_slice_count} slices, but found {self.actual_slice_count}"

        # Load a single slice to get the width and height
        _slice = load_image(os.path.join(self.slices_dir, '00.tif'))
        self.fragment_size_x = _slice.shape[1]
        self.fragment_size_y = _slice.shape[2]
        self.fragment_size_z = self.expected_slice_count

        # Load the slices (tif files) into one tensor, pre-allocate
        self.fragment = torch.zeros(
            self.expected_slice_count, self.fragment_size_x, self.fragment_size_y)
        for i in tqdm(range(self.expected_slice_count)):
            slice_path = os.path.join(self.slices_dir, f"{i:02d}.tif")
            _slice = load_image(slice_path)
            self.fragment[i, :, :] = _slice

        # Print some information about our slices
        log.info(f"Loaded slices into tensor: {self.fragment.shape}")
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"\tSlices type: {self.fragment.dtype}")
            log.debug(f"\tSlices min: {self.fragment.min()}")
            log.debug(f"\tSlices max: {self.fragment.max()}")
            log.debug(f"\tSlices mean: {self.fragment.mean()}")

        # Make sure the patch sizes are valid
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.patch_size_z = patch_size_z
        if log.isEnabledFor(logging.DEBUG):
            assert self.patch_size_x <= self.fragment.shape[
                1], f"Patch size x ({self.patch_size_x}) is larger than the fragment width ({self.fragment.shape[1]})"
            assert self.patch_size_y <= self.fragment.shape[
                2], f"Patch size y ({self.patch_size_y}) is larger than the fragment height ({self.fragment.shape[2]})"
            assert self.patch_size_z <= self.fragment.shape[
                0], f"Patch size z ({self.patch_size_z}) is larger than the fragment depth ({self.fragment.shape[0]})"
            log.debug(
                f"Patch size: ({self.patch_size_x}, {self.patch_size_y}, {self.patch_size_z})")

        # Store the half-sizes of the patches
        self.patch_half_size_x = self.patch_size_x // 2
        self.patch_half_size_y = self.patch_size_y // 2
        self.patch_half_size_z = self.patch_size_z // 2

        # Pad the fragment to make sure we can get patches from the edges
        pad_sizes = (self.patch_half_size_x, self.patch_half_size_x, self.patch_half_size_y,
                     self.patch_half_size_y, self.patch_half_size_z, self.patch_half_size_z)
        self.fragment = torch.nn.functional.pad(
            self.fragment, pad_sizes, value=0.0)

        # Pad the mask, labels, and IR images
        pad_sizes = (self.patch_half_size_x, self.patch_half_size_x, self.patch_half_size_y,
                     self.patch_half_size_y, 0, 0)  # No padding on z
        self.mask_padded = torch.nn.functional.pad(
            self.mask, pad_sizes, value=0.0)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Mask padded: {self.mask_padded.shape}")
            assert self.mask_padded.shape[1:] == self.fragment.shape[1:
                                                                 ], f"Mask and fragment are not the same size: {self.mask_padded.shape} vs {self.fragment.shape}"
        if self.train:
            self.labels_padded = torch.nn.functional.pad(
                self.labels, pad_sizes, value=0.0)
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Labels padded: {self.labels_padded.shape}")
                assert self.labels_padded.shape[1:] == self.fragment.shape[1:
                                                                       ], f"Labels and fragment are not the same size: {self.labels_padded.shape} vs {self.fragment.shape}"

    def __len__(self):
        return self.fragment_size_x * self.fragment_size_y * self.fragment_size_z

    def __getitem__(self, index):
        # Get the x, y, and z indices
        z = index // (self.fragment_size_x * self.fragment_size_y)
        y = (index - z * self.fragment_size_x *
             self.fragment_size_y) // self.fragment_size_x
        x = index - z * self.fragment_size_x * \
            self.fragment_size_y - y * self.fragment_size_x
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Dataset index is {index} out of {self.__len__()}")
            log.debug(f"\t\t Z index is {z} out of {self.fragment_size_z}")
            log.debug(f"\t\t X index is {x} out of {self.fragment_size_x}")
            log.debug(f"\t\t Y index is {y} out of {self.fragment_size_y}")

        # Get the patch
        patch = self.fragment[
            z: z + self.patch_size_z,
            x: x + self.patch_size_x,
            y: y + self.patch_size_y,
        ]
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"Patch shape is {patch.shape}")
            log.debug(f"\t\t Patch type is {patch.dtype}")
            log.debug(f"\t\t Patch min is {patch.min()}")
            log.debug(f"\t\t Patch max is {patch.max()}")
            log.debug(f"\t\t Patch mean is {patch.mean()}")
            assert patch.shape == (self.patch_size_z, self.patch_size_x,
                                self.patch_size_y), f"Patch shape is {patch.shape} but should be {self.patch_size_z, self.patch_size_x, self.patch_size_y}"

        # Get the mask
        mask_patch = self.mask_padded[
            0,
            x: x + self.patch_size_x,
            y: y + self.patch_size_y,
        ]
        log.debug(f"Mask patch shape is {mask_patch.shape}")
        assert mask_patch.shape == (self.patch_size_x,
                                    self.patch_size_y), f"Mask patch shape is {mask_patch.shape} but should be {self.patch_size_x, self.patch_size_y}"

        # Get the label
        if self.train:
            label_patch = self.labels_padded[
                0,
                x: x + self.patch_size_x,
                y: y + self.patch_size_y,
            ]
            log.debug(f"Label patch shape is {label_patch.shape}")
            assert label_patch.shape == (self.patch_size_x,
                                         self.patch_size_y), f"Label patch shape is {label_patch.shape} but should be {self.patch_size_x, self.patch_size_y}"

        if self.viz:
            # Visualize the patch bbox, mask patch, label patch, and patch itself
            fig, ax = plt.subplots(1, 3, figsize=(20, 20))

            # Visualize the patch in the infrared image
            ax[0].set_title("Patch in IR image")
            if self.train:
                _pt_np = self.ir.numpy()
            else:
                _pt_np = self.mask.numpy()
            _pt_np = np.transpose(_pt_np, (1, 2, 0))
            ax[0].imshow(_pt_np, cmap='gray')
            # This box is in the original image coordinates (No padding)
            bbox_anchor = (y - self.patch_half_size_y,
                           x - self.patch_half_size_x)
            bbox = patches.Rectangle(
                (bbox_anchor), self.patch_size_y, self.patch_size_x, linewidth=2, edgecolor='r', facecolor='none')
            ax[0].add_patch(bbox)

            # Visualize the mask of the patch
            ax[1].set_title("Mask patch")
            # These are in padded image coordinates
            _pt_np = self.mask_padded[
                0,
                x: x + self.patch_size_x,
                y: y + self.patch_size_y,
            ]
            # _pt_np = np.transpose(_pt_np, (1, 2, 0))
            ax[1].imshow(_pt_np, cmap='gray')

            # Visualize the label of the patch
            if self.train:
                ax[2].set_title(f"Label patch")
                _pt_np = self.labels_padded[
                    0,
                    x: x + self.patch_size_x,
                    y: y + self.patch_size_y,
                ]
                # _pt_np = np.transpose(_pt_np, (1, 2, 0))
                ax[2].imshow(_pt_np, cmap='gray')
                plt.show()

            # Calculate the number of rows and columns
            num_cols = int(np.ceil(np.sqrt(self.patch_size_z)))
            num_rows = int(np.ceil(self.patch_size_z / num_cols))

            # Visualize each depth of the patch
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))
            fig.suptitle(
                f"Raw input to NN (shown sliced, original size is {self.patch_size_z}x{self.patch_size_x}x{self.patch_size_y})")

            # Flatten the axs array in case it's 2D
            axs_flat = axs.flatten()

            for i in range(self.patch_size_z):
                axs_flat[i].imshow(patch[i, :, :].numpy(), cmap='gray')

            # Hide the remaining unused subplots if any
            for i in range(self.patch_size_z, num_rows * num_cols):
                axs_flat[i].axis('off')

            plt.show()

        if self.train:
            return patch, mask_patch, label_patch
        # If we're not training, we don't have labels
        return patch, mask_patch


if __name__ == '__main__':
    log.setLevel(logging.DEBUG)

    # Test the dataset class
    # data_dir = 'data/train/1/'
    # data_dir = 'data/train/2/'
    data_dir = 'data/train/3/'
    # data_dir = 'data/test/a/'
    # data_dir = 'data/test/b/'

    # Train mode w/ viz
    dataset = FragmentPatchesDataset(data_dir, viz=True, train=True)
    indices_to_try = [
        0,
        len(dataset) // 2,
        len(dataset) - 1,
        random.randint(0, len(dataset)),
        random.randint(0, len(dataset)),
    ]
    for i in indices_to_try:
        log.debug(f"Trying index {i} out of {len(dataset)}")
        patch, mask, label = dataset[i]
    del dataset
    
    # Test mode w/ viz
    dataset = FragmentPatchesDataset(data_dir, viz=True, train=False)
    for i in indices_to_try:
        log.debug(f"Trying index {i} out of {len(dataset)}")
        patch, mask = dataset[i]
    del dataset

    # # Batched train mode w/o viz
    # dataset = FragmentPatchesDataset(data_dir, train=True)
    # dataset_loader = data.DataLoader(dataset, batch_size=2, shuffle=True)
    # for i, (patch, mask, label) in enumerate(dataset_loader):
    #     log.debug(f"Batch {i} has shape {patch.shape}")
    #     if i == 2:
    #         break
    # del dataset