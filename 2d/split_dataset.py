"""
    Split dataset into Train and Validation, by cutting 
    the image into a top half and bottom half.
"""

import os
from PIL import Image
from tqdm import tqdm

image_mask_filename='mask.png'
image_labels_filename='inklabels.png'
image_ir_filename = 'ir.png'
slices_dir_filename='surface_volume'


num_slices = 65
split = 0.85

target_dir = '/home/tren/dev/ashenvenus/data/train'
output_dir_train = '/home/tren/dev/ashenvenus/data/split_train'
output_dir_valid = '/home/tren/dev/ashenvenus/data/split_valid'

# target_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\train"
# output_dir_train = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\split_train"
# output_dir_valid = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\split_valid"

# Baseline is to use image mask to create guess submission
for dataset in os.listdir(target_dir):
    dataset_filepath = os.path.join(target_dir, dataset)
    dataset_train_filepath = os.path.join(output_dir_train, dataset)
    os.makedirs(dataset_train_filepath, exist_ok=True)
    dataset_valid_filepath = os.path.join(output_dir_valid, dataset)
    os.makedirs(dataset_valid_filepath, exist_ok=True)

    # Open IR image
    _image_ir_filepath = os.path.join(dataset_filepath, image_ir_filename)
    _image_ir_train_filepath = os.path.join(dataset_train_filepath, image_ir_filename)
    _image_ir_valid_filepath = os.path.join(dataset_valid_filepath, image_ir_filename)
    _ir_img = Image.open(_image_ir_filepath).convert("L")

    # Split into train and val
    width, height = _ir_img.size
    _ir_img_train = _ir_img.crop((0, 0, width, int(height * split)))
    _ir_img_train.save(_image_ir_train_filepath)
    _ir_img_val = _ir_img.crop((0, int(height * split), width, height))
    _ir_img_val.save(_image_ir_valid_filepath)

    continue

    # Open Mask image
    _image_mask_filepath = os.path.join(dataset_filepath, image_mask_filename)
    _image_mask_train_filepath = os.path.join(dataset_train_filepath, image_mask_filename)
    _image_mask_valid_filepath = os.path.join(dataset_valid_filepath, image_mask_filename)
    _mask_img = Image.open(_image_mask_filepath).convert("1")

    # Split into train and val
    width, height = _mask_img.size
    _mask_img_train = _mask_img.crop((0, 0, width, int(height * split)))
    _mask_img_train.save(_image_mask_train_filepath)
    _mask_img_val = _mask_img.crop((0, int(height * split), width, height))
    _mask_img_val.save(_image_mask_valid_filepath)
    
    # Open Labels image
    _image_labels_filepath = os.path.join(dataset_filepath, image_labels_filename)
    _image_labels_train_filepath = os.path.join(dataset_train_filepath, image_labels_filename)
    _image_labels_valid_filepath = os.path.join(dataset_valid_filepath, image_labels_filename)
    _labels_img = Image.open(_image_labels_filepath).convert("1")

    # Split into train and val
    width, height = _labels_img.size
    _labels_img_train = _labels_img.crop((0, 0, width, int(height * split)))
    _labels_img_train.save(_image_labels_train_filepath)
    _labels_img_val = _labels_img.crop((0, int(height * split), width, height))
    _labels_img_val.save(_image_labels_valid_filepath)

    # Open Slices
    _slice_dir = os.path.join(dataset_filepath, slices_dir_filename)
    _slice_train_dir = os.path.join(dataset_train_filepath, slices_dir_filename)
    os.makedirs(_slice_train_dir, exist_ok=True)
    _slice_valid_dir = os.path.join(dataset_valid_filepath, slices_dir_filename)
    os.makedirs(_slice_valid_dir, exist_ok=True)

    for i in tqdm(range(num_slices), postfix='converting slices'):
        _slice_filepath = os.path.join(_slice_dir, f"{i:02d}.tif")
        _slice_img = Image.open(_slice_filepath)

        # Split into train and val
        width, height = _slice_img.size
        _slice_img_train = _slice_img.crop((0, 0, width, int(height * split)))
        _slice_img_train.save(os.path.join(_slice_train_dir, f"{i:02d}.tif"))
        _slice_img_val = _slice_img.crop((0, int(height * split), width, height))
        _slice_img_val.save(os.path.join(_slice_valid_dir, f"{i:02d}.tif"))