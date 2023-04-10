"""
    Get the pixel statistics of the data.
"""

import os

import numpy as np
import yaml
import pprint
from PIL import Image
from tqdm import tqdm

image_mask_filename='mask.png'
image_labels_filename='inklabels.png'
image_ir_filename = 'ir.png'
slices_dir_filename='surface_volume'
num_slices = 65

data_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data"

# Find all nested sub directories that contain an image mask image
dataset_dirs = []
for root, dirs, files in os.walk(data_dir):
    if image_labels_filename in files and image_mask_filename in files:
        print(f"Found dataset: {root}")
        dataset_dirs.append(root)

for dataset_dir in dataset_dirs:
    print(f"Processing dataset: {dataset_dir}")

    # Output will be a YAML file in the dataset directory
    pixel_stats = {}

    # Open IR image
    _image_ir_filepath = os.path.join(dataset_dir, image_ir_filename)
    ir_img = Image.open(_image_ir_filepath).convert("L")

    # Open Mask Image
    _image_mask_filepath = os.path.join(dataset_dir, image_mask_filename)
    mask_img = Image.open(_image_mask_filepath).convert("L")
    mask = np.array(mask_img, dtype=np.bool8)

    # Open Labels image
    _image_labels_filepath = os.path.join(dataset_dir, image_labels_filename)
    labels_img = Image.open(_image_labels_filepath).convert("L")
    labels = np.array(labels_img, dtype=np.bool8)

    # Open Slices into numpy array
    fragment = np.zeros((num_slices, labels_img.height, labels_img.width), dtype=np.float32)
    _slice_dir = os.path.join(dataset_dir, slices_dir_filename)
    for i in tqdm(range(num_slices), postfix='converting slices'):
        _slice_filepath = os.path.join(_slice_dir, f"{i:02d}.tif")
        slice_img = Image.open(_slice_filepath).convert("F")
        fragment[i, :, :] = np.array(slice_img) / 65535.0

  # Raw - Every Pixel in Fragment
    pixel_stats['raw'] = {
        'min': float(fragment.min()),
        'max': float(fragment.max()),
        'mean': float(fragment.mean()),
        'std': float(fragment.std()),
        'count': int(fragment.size),
    }

    # Mask - Only Pixels in Mask
    _fragment = fragment.copy()
    _fragment[:,:,:] = mask[np.newaxis,:,:] == True
    pixel_stats['mask'] = {
        'min': float(_fragment.min()),
        'max': float(_fragment.max()),
        'mean': float(_fragment.mean()),
        'std': float(_fragment.std()),
        'count': int(_fragment.size),
    }

    # Ink - Only Pixels labeled as ink
    _fragment = fragment.copy()
    _fragment[:,:,:] = labels[np.newaxis,:,:] == True
    pixel_stats['ink'] = {
        'min': float(_fragment.min()),
        'max': float(_fragment.max()),
        'mean': float(_fragment.mean()),
        'std': float(_fragment.std()),
        'count': int(_fragment.size),
    }

    # Background - Pixels in Mask but that aren't Ink
    _fragment = fragment.copy()
    _fragment[:,:,:] = (mask[np.newaxis,:,:] == True) & (labels[np.newaxis,:,:] == False)
    pixel_stats['bg'] = {
        'min': float(_fragment.min()),
        'max': float(_fragment.max()),
        'mean': float(_fragment.mean()),
        'std': float(_fragment.std()),
        'count': int(_fragment.size),
    }

    print(f"Pixel Stats: {pprint.pformat(pixel_stats)}")

    # Save stats to yaml file
    _pixel_stats_yaml_filepath = os.path.join(dataset_dir, 'pixel_stats.yaml')
    with open(_pixel_stats_yaml_filepath, 'w') as f:
        yaml.dump(pixel_stats, f)