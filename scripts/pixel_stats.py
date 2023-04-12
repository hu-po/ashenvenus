"""
    Get the pixel statistics of the data.
"""

import os

import numpy as np
import yaml
import pprint
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

image_mask_filename = 'mask.png'
image_labels_filename = 'inklabels.png'
image_ir_filename = 'ir.png'
slices_dir_filename = 'surface_volume'
num_slices = 50

data_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\"
# data_dir = '/home/tren/dev/ashenvenus/data/split/'

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

        # # Raw - Every Pixel in Fragment
        # pixel_stats['raw']['min'].append(float(fragment.min()))
        # pixel_stats['raw']['max'].append(float(fragment.max()))
        # pixel_stats['raw']['mean'].append(float(fragment.mean()))
        # pixel_stats['raw']['std'].append(float(fragment.std()))
        # pixel_stats['raw']['count'] += int(fragment.size)

        # Mask - Only Pixels in Mask
        fragment_mask = fragment[:, mask > 0]
        pixel_stats['mask']['min'].append(float(fragment_mask.min()))
        pixel_stats['mask']['max'].append(float(fragment_mask.max()))
        pixel_stats['mask']['mean'].append(float(fragment_mask.mean()))
        pixel_stats['mask']['std'].append(float(fragment_mask.std()))
        pixel_stats['mask']['count'] += int(fragment_mask.size)

        # Ink - Only Pixels labeled as ink
        fragment_ink = fragment[:, labels > 0]
        pixel_stats['ink']['min'].append(float(fragment_ink.min()))
        pixel_stats['ink']['max'].append(float(fragment_ink.max()))
        pixel_stats['ink']['mean'].append(float(fragment_ink.mean()))
        pixel_stats['ink']['std'].append(float(fragment_ink.std()))
        pixel_stats['ink']['count'] += int(fragment_ink.size)

        # Background - Pixels in Mask but that aren't Ink
        fragment_bg = fragment[:, (mask > 0) & (labels < 1)]
        pixel_stats['bg']['min'].append(float(fragment_bg.min()))
        pixel_stats['bg']['max'].append(float(fragment_bg.max()))
        pixel_stats['bg']['mean'].append(float(fragment_bg.mean()))
        pixel_stats['bg']['std'].append(float(fragment_bg.std()))
        pixel_stats['bg']['count'] += int(fragment_bg.size)

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

    # Matplotlib Histogram of all pixels with lines for each type of fragment
    # plt.hist(fragment.flatten(), bins=100, alpha=0.5, label='raw')
    plt.hist(fragment_mask.flatten(), bins=100, alpha=0.5, label='mask')
    plt.hist(fragment_bg.flatten(), bins=100, alpha=0.5, label='bg')
    plt.hist(fragment_ink.flatten(), bins=100, alpha=0.5, label='ink')
    plt.legend(loc='upper right')
    plt.title('Pixel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')

    # Save to image
    _pixel_stats_png_filepath = os.path.join(dataset_dir, 'pixel_stats.png')
    plt.savefig(_pixel_stats_png_filepath)
