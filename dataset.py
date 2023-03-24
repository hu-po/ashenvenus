import glob
import logging
import os

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image as Image
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import tqdm
import random
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_image(image_filepath: str = 'image.png', viz: bool = False) -> torch.Tensor:
    assert os.path.exists(
        image_filepath), f"Image file {image_filepath} does not exist"
    log.debug(f"Loading {image_filepath}")
    _image = Image.open(image_filepath)
    _pt = ToTensor()(_image)
    log.debug(f"\tImage shape: {_pt.shape}")
    log.debug(f"\tImage type: {_pt.dtype}")
    log.debug(f"\tImage min: {_pt.min()}")
    log.debug(f"\tImage max: {_pt.max()}")
    log.debug(f"\tImage mean: {_pt.mean()}")
    log.debug(f"\tImage std: {_pt.std()}")
    if viz:
        plt.title(image_filepath)
        _pt_np = _pt.numpy()
        if _pt_np.shape[0] in [1, 3, 4]:  # If there are 1, 3, or 4 channels
            # Move the channel axis to the last position
            _pt_np = np.transpose(_pt_np, (1, 2, 0))
        plt.imshow(_pt_np, cmap='gray')
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
        patch_size_x: int = 64,
        patch_size_y: int = 64,
        patch_size_z: int = 4,
    ):
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
        assert os.path.exists(
            self.image_labels_filepath), f"Labels file {self.image_labels_filepath} does not exist"
        self.image_ir_filename = image_ir_filename
        self.image_ir_filepath = os.path.join(data_dir, self.image_ir_filename)
        assert os.path.exists(
            self.image_ir_filepath), f"IR file {self.image_ir_filepath} does not exist"
        self.slices_dir = os.path.join(data_dir, slices_dir_filename)
        assert os.path.exists(
            self.slices_dir), f"Slices directory {self.slices_dir} does not exist"

        # Load the meta data (mask, labels, and IR images)
        self.mask = load_image(self.image_mask_filepath)
        self.labels = load_image(self.image_labels_filepath)
        self.ir = load_image(self.image_ir_filepath)

        # Assert that there are the correct amount of slices
        self.expected_slice_count = expected_slice_count
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
        for i, slice_path in enumerate(tqdm(glob.glob(os.path.join(self.slices_dir, '*.tif')))):
            _slice = load_image(slice_path)
            self.fragment[i, :, :] = _slice

        # Print some information about our slices
        log.info(f"Loaded slices into tensor: {self.fragment.shape}")
        log.debug(f"\tSlices type: {self.fragment.dtype}")
        log.debug(f"\tSlices min: {self.fragment.min()}")
        log.debug(f"\tSlices max: {self.fragment.max()}")
        log.debug(f"\tSlices mean: {self.fragment.mean()}")

        # Make sure the patch sizes are valid
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.patch_size_z = patch_size_z
        assert self.patch_size_x <= self.fragment.shape[
            1], f"Patch size x ({self.patch_size_x}) is larger than the fragment width ({self.fragment.shape[1]})"
        assert self.patch_size_y <= self.fragment.shape[
            2], f"Patch size y ({self.patch_size_y}) is larger than the fragment height ({self.fragment.shape[2]})"
        assert self.patch_size_z <= self.fragment.shape[
            0], f"Patch size z ({self.patch_size_z}) is larger than the fragment depth ({self.fragment.shape[0]})"
        
        # Store the half-sizes of the patches
        self.patch_half_size_x = self.patch_size_x // 2
        self.patch_half_size_y = self.patch_size_y // 2
        self.patch_half_size_z = self.patch_size_z // 2

        # Pad the fragment to make sure we can get patches from the edges
        self.fragment = torch.nn.functional.pad(self.fragment, (self.patch_half_size_x, self.patch_half_size_x, self.patch_half_size_y, self.patch_half_size_y, self.patch_half_size_z, self.patch_half_size_z))

        # TODO: Make sure only indices that are within the mask are valid

    def __len__(self):
        return self.fragment_size_x * self.fragment_size_y * self.fragment_size_z

    def __getitem__(self, index, viz=False):
        # Get the x, y, and z indices
        z = index // (self.fragment_size_x * self.fragment_size_y)
        y = (index - z * self.fragment_size_x *
             self.fragment_size_y) // self.fragment_size_x
        x = index - z * self.fragment_size_x * \
            self.fragment_size_y - y * self.fragment_size_x
        log.debug(f"Dataset index is {index} out of {self.__len__()}")
        log.debug(f"\t\t Z index is {z} out of {self.fragment_size_z}")
        log.debug(f"\t\t X index is {x} out of {self.fragment_size_x}")
        log.debug(f"\t\t Y index is {y} out of {self.fragment_size_y}")

        # Get the patch and label
        patch = self.fragment[
            z - self.patch_half_size_x : z + self.patch_half_size_x,
            x - self.patch_half_size_y : x + self.patch_half_size_y,
            y - self.patch_half_size_z : y + self.patch_half_size_z,
        ]
        # TODO: Should label be average of patch? center of patch?
        label = self.labels[0, x, y]

        if viz:
            # Visualize the patch
            fig, ax = plt.subplots(1, self.patch_size_z + 1)
            # Visualize the patch in the infrared image
            _pt_np = self.ir.numpy()
            if _pt_np.shape[0] in [1, 3, 4]:  # If there are 1, 3, or 4 channels
                # Move the channel axis to the last position
                _pt_np = np.transpose(_pt_np, (1, 2, 0))
            ax[0].imshow(_pt_np, cmap='gray')
            bbox_anchor = (y - self.patch_half_size_y, x - self.patch_half_size_x)
            bbox = patches.Rectangle(
                bbox_anchor, self.patch_size_y, self.patch_size_x, linewidth=10, edgecolor='r', facecolor='none')
            ax[0].add_patch(bbox)
            # Visualize each depth of the patch
            for i in range(self.patch_size_z):
                ax[i + 1].imshow(patch[:, :, i].numpy(), cmap='gray')
            plt.show()

        return patch, label


if __name__ == '__main__':
    log.setLevel(logging.DEBUG)

    # # Test the image loading function
    # data_dir = 'data/train/1/'
    # mask = load_image(os.path.join(data_dir, 'mask.png'), viz=True)
    # labels = load_image(os.path.join(data_dir, 'inklabels.png'), viz=True)
    # ir = load_image(os.path.join(data_dir, 'ir.png'), viz=True)
    # tif = load_image(os.path.join(data_dir, 'surface_volume/00.tif'), viz=True)

    # Test the dataset class
    data_dir = 'data/train/1/'
    dataset = FragmentPatchesDataset(data_dir)
    for _ in range(3):
        i = random.randint(0, len(dataset))
        patch, label = dataset.__getitem__(i, viz=True)
