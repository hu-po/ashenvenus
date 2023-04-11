"""
    Get the pixel statistics of the data.
"""

import os

import numpy as np
import yaml
import pprint
from PIL import Image
from tqdm import tqdm

image_mask_filename = 'mask.png'
image_labels_filename = 'inklabels.png'
image_ir_filename = 'ir.png'
slices_dir_filename = 'surface_volume'
num_slices = 50

# data_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\split*"
data_dir = '/home/tren/dev/ashenvenus/data/split/'

# Find all nested sub directories that contain an image mask image
dataset_dirs = []
for root, dirs, files in os.walk(data_dir):
    if image_labels_filename in files and image_mask_filename in files:
        print(f"Found dataset: {root}")
        dataset_dirs.append(root)

for dataset_dir in dataset_dirs:
    print(f"Processing dataset: {dataset_dir}")

    # Output will be a YAML file in the dataset directory
    pixel_stats = {
        'raw': {'min': [], 'max': [], 'mean': [], 'std': [], 'count': 0},
        'mask': {'min': [], 'max': [], 'mean': [], 'std': [], 'count': 0},
        'ink': {'min': [], 'max': [], 'mean': [], 'std': [], 'count': 0},
        'bg': {'min': [], 'max': [], 'mean': [], 'std': [], 'count': 0},
    }

    # Open Mask Image
    _image_mask_filepath = os.path.join(dataset_dir, image_mask_filename)
    mask_img = Image.open(_image_mask_filepath).convert("L")
    mask = np.array(mask_img, dtype=np.uint8)

    # Open Labels image
    _image_labels_filepath = os.path.join(dataset_dir, image_labels_filename)
    labels_img = Image.open(_image_labels_filepath).convert("L")
    labels = np.array(labels_img, dtype=np.uint8)

    # Open Slices into numpy array
    fragment = np.zeros((num_slices, labels_img.height,
                         labels_img.width), dtype=np.float32)
    _slice_dir = os.path.join(dataset_dir, slices_dir_filename)
    for i in tqdm(range(num_slices), postfix='converting slices'):
        _slice_filepath = os.path.join(_slice_dir, f"{i:02d}.tif")
        slice_img = Image.open(_slice_filepath).convert("F")
        fragment[i, :, :] = np.array(slice_img) / 65535.0

        # Raw - Every Pixel in Fragment
        pixel_stats['raw']['min'].append(float(fragment.min()))
        pixel_stats['raw']['max'].append(float(fragment.max()))
        pixel_stats['raw']['mean'].append(float(fragment.mean()))
        pixel_stats['raw']['std'].append(float(fragment.std()))
        pixel_stats['raw']['count'] += int(fragment.size)

        # Mask - Only Pixels in Mask
        _fragment = fragment.copy()
        _fragment = mask > 0
        pixel_stats['mask']['min'].append(float(_fragment.min()))
        pixel_stats['mask']['max'].append(float(_fragment.max()))
        pixel_stats['mask']['mean'].append(float(_fragment.mean()))
        pixel_stats['mask']['std'].append(float(_fragment.std()))
        pixel_stats['mask']['count'] += int(_fragment.size)

        # Ink - Only Pixels labeled as ink
        _fragment = fragment.copy()
        _fragment = labels > 0
        pixel_stats['ink']['min'].append(float(_fragment.min()))
        pixel_stats['ink']['max'].append(float(_fragment.max()))
        pixel_stats['ink']['mean'].append(float(_fragment.mean()))
        pixel_stats['ink']['std'].append(float(_fragment.std()))
        pixel_stats['ink']['count'] += int(_fragment.size)

        # Background - Pixels in Mask but that aren't Ink
        _fragment = fragment.copy()
        _fragment = (mask > 0) & (labels > 0)
        pixel_stats['bg']['min'].append(float(_fragment.min()))
        pixel_stats['bg']['max'].append(float(_fragment.max()))
        pixel_stats['bg']['mean'].append(float(_fragment.mean()))
        pixel_stats['bg']['std'].append(float(_fragment.std()))
        pixel_stats['bg']['count'] += int(_fragment.size)

    # Calculate average statistics
    for key in pixel_stats.keys():
        pixel_stats[key]['min'] = float(np.mean(pixel_stats[key]['min']))
        pixel_stats[key]['max'] = float(np.mean(pixel_stats[key]['max']))
        pixel_stats[key]['mean'] = float(np.mean(pixel_stats[key]['mean']))
        pixel_stats[key]['std'] = float(np.mean(pixel_stats[key]['std']))

    print(f"Pixel Stats: {pprint.pformat(pixel_stats)}")

    # Save stats to yaml file
    _pixel_stats_yaml_filepath = os.path.join(dataset_dir, 'pixel_stats.yaml')
    with open(_pixel_stats_yaml_filepath, 'w') as f:
        yaml.dump(pixel_stats, f)
