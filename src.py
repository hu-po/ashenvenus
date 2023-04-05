import csv
import gc
import os
import subprocess
import time
from typing import Union
import shutil

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import yaml
from PIL import Image, ImageFilter
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms, models
import torch.quantization as quantization
from tqdm import tqdm
import os
import pprint
import uuid

import numpy as np
import yaml


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
        interpolation: str = 'bilinear',
        # Training vs Testing mode
        train: bool = True,
        # Type of interpolation to use when resizing
        # Filenames of the images we'll use
        image_mask_filename='mask.png',
        image_labels_filename='inklabels.png',
        slices_dir_filename='surface_volume',
    ):
        print(f"Creating CurriculumDataset for {data_dir}")
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
        _mask_img = _mask_img.resize(
            self.resized_size, resample=INTERPOLATION_MODES[interpolation])
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
                self.resized_size, resample=INTERPOLATION_MODES[interpolation])
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
        for i in tqdm(range(self.slice_depth), postfix='loading dataset'):
            _slice_filepath = os.path.join(_slice_dir, f"{i:02d}.tif")
            _slice_img = Image.open(_slice_filepath).convert('F')
            # print(f"Slice original size: {original_size}")
            _slice_img = _slice_img.resize(
                self.resized_size, resample=INTERPOLATION_MODES[interpolation])
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

        # TODO: Use Predictions to additionally balance the dataset
        # if self.train:
        #     # Get indices where labels are 1
        #     self.labels_indices = torch.nonzero(self.labels).to(torch.int32)
        #     # print(f"Labels indices shape: {self.labels_indices.shape}")
        #     # print(f"Labels indices dtype: {self.labels_indices.dtype}")

        #     # Indices where mask is 0 and labels is 1
        #     self.mask_0_labels_1_indices = torch.nonzero(
        #         (~_mask) & self.labels
        #     ).to(torch.int32)

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


def get_device(device: str = None):
    if device == None or device == "gpu":
        if torch.cuda.is_available():
            print("Using GPU")
            return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def image_to_rle(img, threshold=0.5):
    # TODO: Histogram of image to see where threshold should be
    flat_img = img.flatten()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return starts_ix, lengths


def save_rle_as_image(rle_csv_path, output_dir, subtest_name, image_shape):
    with open(rle_csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            _subtest_name, rle_data = row
            if _subtest_name != subtest_name:
                continue
            rle_pairs = list(map(int, rle_data.split()))

            # Decode RLE data
            img = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
            for i in range(0, len(rle_pairs), 2):
                start = rle_pairs[i] - 1
                end = start + rle_pairs[i + 1]
                img[start:end] = 1

            # Reshape decoded image data to original shape
            img = img.reshape(image_shape)
            img = Image.fromarray(img * 255).convert('1')
            _image_filepath = os.path.join(
                output_dir, f"pred_{subtest_name}_rle.png")
            img.save(_image_filepath)


def dice_score(preds, label, beta=0.5, epsilon=1e-6):
    # Implementation of DICE coefficient
    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    preds = torch.sigmoid(preds)
    preds = preds.flatten()
    # print(f"Predictions tensor shape: {preds.shape}")
    # print(f"Predictions tensor dtype: {preds.dtype}")
    # print(f"Predictions tensor min: {preds.min()}")
    # print(f"Predictions tensor max: {preds.max()}")
    label = label.flatten()
    # print(f"Label tensor shape: {label.shape}")
    # print(f"Label tensor dtype: {label.dtype}")
    # print(f"Label tensor min: {label.min()}")
    # print(f"Label tensor max: {label.max()}")
    tp = preds[label == 1].sum()
    fp = preds[label == 0].sum()
    fn = label.sum() - tp
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    _score = (1 + beta * beta) * (p * r) / (beta * beta * p + r + epsilon)
    # print(f"DICE score: {_score}")
    return _score


def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free',
                             '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, text=True)
    gpu_memory = [tuple(map(int, line.split(',')))
                  for line in result.stdout.strip().split('\n')]
    for i, (used, free) in enumerate(gpu_memory):
        print(f"GPU {i}: Memory Used: {used} MiB | Memory Available: {free} MiB")


def clear_gpu_memory():
    if torch.cuda.is_available():
        print('Clearing GPU memory')
        torch.cuda.empty_cache()
        gc.collect()


def print_size_of_model(model: nn.Module):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Model Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


# Types of interpolation used in resizing
INTERPOLATION_MODES = {
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'nearest': Image.NEAREST,
}


class ImageModel(nn.Module):

    models = {
        'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
        'convnext_small': (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT),
        'convnext_base': (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT),
        'convnext_large': (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT),
        'resnext50_32x4d': (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.DEFAULT),
        'resnext101_32x8d': (models.resnext101_32x8d, models.ResNeXt101_32X8D_Weights.DEFAULT),
        'resnext101_64x4d': (models.resnext101_64x4d, models.ResNeXt101_64X4D_Weights.DEFAULT),
    }

    def __init__(
        self,
        slice_depth: int = 65,
        model: str = 'convnext_tiny',
        load_fresh: bool = False,
        freeze: bool = False,
    ):
        super().__init__()
        print(f"Initializing new model: {model}")
        # Channel reduction
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(slice_depth, 3, 1),
            nn.Tanh(),
        )
        assert model in self.models, f"Model {model} not supported"
        _f, weights = self.models[model]
        # Optionally load fresh pre-trained model from scratch
        self.model = _f(weights=weights if load_fresh else None)
        # Put model in training mode
        if freeze:
            self.model.eval()
        else:
            self.model.train()
        # Binary classification head on top
        self.head = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 1),
        )

    def forward(self, x):
        x = self.channel_reduce(x)
        x = self.model(x)
        x = self.head(x)
        return x


class VideoModel(nn.Module):

    models = {
        'r2plus1d_18': (models.video.r2plus1d_18, models.video.R2Plus1D_18_Weights.DEFAULT),
        # 'swin3d_b': (models.video.swin3d_b, models.video.Swin3D_B_Weights.DEFAULT),
        # 'mvit_v2_s': (models.video.mvit_v2_s, models.video.MViT_V2_S_Weights.DEFAULT),
    }

    def __init__(
        self,
        slice_depth: int = 65,
        model: str = 'r2plus1d_18',
        load_fresh: bool = False,
        freeze: bool = False,
    ):
        super().__init__()
        print(f"Initializing new model: {model}")
        assert model in self.models, f"Model {model} not supported"
        _f, weights = self.models[model]
        # Optionally load fresh pre-trained model from scratch
        self.model = _f(weights=weights if load_fresh else None)
        # Put model in training mode
        if freeze:
            self.model.eval()
        else:
            self.model.train()
        # Binary classification head on top
        self.head = nn.Sequential(
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 1),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1, 3, x.shape[-2], x.shape[-1])
        x = self.model(x)
        x = self.head(x)
        return x


def train(
    train_dir: str = "data/train/",
    model: str = "simplenet",
    freeze: bool = False,
    weights_filepath: str = None,
    optimizer: str = "adam",
    weight_decay: float = 0.,
    curriculum: str = "1",
    num_samples: int = 100,
    num_workers: int = 1,
    output_dir: str = "output/train",
    image_augs: bool = False,
    slice_depth: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    interpolation: str = "bilinear",
    batch_size: int = 16,
    lr: float = 0.001,
    lr_gamma: float = None,
    num_epochs: int = 2,
    max_time_hours: float = 8,
    writer: SummaryWriter = None,
    save_model: bool = True,
    device: str = "gpu",
    **kwargs,
):
    # Notebook will only run for this amount of time
    print(f"Training will run for {max_time_hours} hours")
    time_train_max_seconds = max_time_hours * 60 * 60
    time_start = time.time()
    time_elapsed = 0

    # Get GPU
    device = get_device(device)
    clear_gpu_memory()

    # Load the model, try to fit on GPU
    if isinstance(model, str):
        if model in ["simplenet", "convnext_tiny", "convnext_small", "convnext_medium", "convnext_large", "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d"]:
            model = ImageModel(
                model=model,
                slice_depth=slice_depth,
                freeze=freeze,
            )
        elif model in ["r2plus1d_18"]:
            model = VideoModel(
                model=model,
                slice_depth=slice_depth,
                freeze=freeze,
            )
        else:
            raise ValueError(f"Model {model} not supported")
        if weights_filepath is not None:
            print(f"Loading weights from {weights_filepath}")
            model.load_state_dict(torch.load(
                weights_filepath,
                map_location=device,
            ))
    model = model.to(device)
    model.train()

    # Create optimizers
    if optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # Scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    if lr_gamma is not None:
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    # Writer for Tensorboard
    if writer:
        writer = SummaryWriter(output_dir)

    # Train the model
    best_score = 0
    step = 0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        # Curriculum defines the order of the training
        for current_dataset_id in curriculum:

            _train_dir = os.path.join(train_dir, current_dataset_id)
            print(f"Training on dataset: {_train_dir}")

            # Training dataset
            train_dataset = PatchDataset(
                # Directory containing the dataset
                _train_dir,
                # Expected slices per fragment
                slice_depth=slice_depth,
                # Size of an individual patch
                patch_size_x=patch_size_x,
                patch_size_y=patch_size_y,
                # Image resize ratio
                resize_ratio=resize_ratio,
                interpolation=interpolation,
                # Training vs Testing mode
                train=True,
            )
            total_dataset_size = len(train_dataset)
            print(f"Raw train dataset size: {total_dataset_size}")

            # Image augmentations
            img_transform_list = [
                transforms.Normalize(train_dataset.mean, train_dataset.std)
            ]
            if image_augs:
                img_transform_list += [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(),
                ]
            img_transform = transforms.Compose(img_transform_list)

            # DataLoaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=RandomSampler(
                    train_dataset, num_samples=num_samples),
                num_workers=num_workers,
                # This will make it go faster if it is loaded into a GPU
                pin_memory=True,
            )

            print(f"Training...")
            train_loss = 0
            score = 0
            _loader = tqdm(train_dataloader)
            for patch, label in _loader:
                # writer.add_histogram('patch_input', patch, step)
                # writer.add_histogram('label_input', label, step)
                patch = patch.to(device)
                label = label.to(device)
                with torch.cuda.amp.autocast():
                    pred = model(img_transform(patch))
                    loss = loss_fn(pred, label)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                _loader.set_postfix_str(
                    f"Train.{current_dataset_id}.{loss_fn.__class__.__name__}: {loss.item():.4f}")

                step += 1
                with torch.no_grad():
                    train_loss += loss.item()
                    score += dice_score(pred, label).item()

                # Check if we have exceeded the time limit
                time_elapsed = time.time() - time_start
                if time_elapsed > time_train_max_seconds:
                    print("Time limit exceeded, stopping batches")
                    break

            if writer:
                train_loss /= num_samples
                writer.add_scalar(
                    f'{loss_fn.__class__.__name__}/{current_dataset_id}/train', train_loss, step)

            # Score is average dice score for all batches
            score /= len(train_dataloader)
            if score > best_score:
                print("New best score: %.4f" % score)
                best_score = score
                if save_model:
                    print("Saving model...")
                    torch.save(model.state_dict(), f"{output_dir}/model.pth")
            if writer:
                writer.add_scalar(
                    f'Dice/{current_dataset_id}/train', score, step)

            # Check if we have exceeded the time limit
            time_elapsed = time.time() - time_start
            if time_elapsed > time_train_max_seconds:
                print("Time limit exceeded, stopping curriculum")
                break
        print(f"Time elapsed: {time_elapsed} seconds")

        if lr_gamma is not None:
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print("Epoch %d: SGD lr %.4f -> %.4f" %
                  (epoch, before_lr, after_lr))

        # Check if we have exceeded the time limit
        time_elapsed = time.time() - time_start
        if time_elapsed > time_train_max_seconds:
            print("Time limit exceeded, stopping training")
            break

    return best_score, model


def eval(
    eval_dir: str = "data/test/",
    weights_filepath: str = None,
    model: Union[str, nn.Module] = "convnext_tiny",
    output_dir: str = "output",
    slice_depth: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    interpolation: str = "bilinear",
    freeze: bool = False,
    batch_size: int = 16,
    num_workers: int = 1,
    save_pred_img: bool = True,
    save_submit_csv: bool = False,
    threshold: float = 0.5,
    postproc_kernel: int = 3,
    writer: SummaryWriter = None,
    quantize: bool = False,
    device: str = "gpu",
    **kwargs,
):
    # Get GPU
    device = get_device(device)
    clear_gpu_memory()
    # device = torch.device('cpu')

    # Load the model, try to fit on GPU
    if isinstance(model, str):
        if model in ["simplenet", "convnext_tiny", "convnext_small", "convnext_medium", "convnext_large", "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d"]:
            model = ImageModel(
                model=model,
                slice_depth=slice_depth,
                freeze=freeze,
            )
        elif model in ["r2plus1d_18"]:
            model = VideoModel(
                model=model,
                slice_depth=slice_depth,
                freeze=freeze,
            )
        else:
            raise ValueError(f"Model {model} not supported")
        if weights_filepath is not None:
            print(f"Loading weights from {weights_filepath}")
            model.load_state_dict(torch.load(
                weights_filepath,
                map_location=device,
            ))
    model = model.to(device)
    model = model.eval()
    print_size_of_model(model)

    if quantize:
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(model, inplace=True)
        quantization.convert(model, inplace=True)
        print_size_of_model(model)

    if save_submit_csv:
        submission_filepath = os.path.join(output_dir, 'submission.csv')
        with open(submission_filepath, 'w') as f:
            # Write header
            f.write("Id,Predicted\n")

    # Baseline is to use image mask to create guess submission
    for subtest_name in os.listdir(eval_dir):

        # Name of sub-directory inside test dir
        subtest_filepath = os.path.join(eval_dir, subtest_name)

        # Evaluation dataset
        eval_dataset = PatchDataset(
            # Directory containing the dataset
            subtest_filepath,
            # Expected slices per fragment
            slice_depth=slice_depth,
            # Size of an individual patch
            patch_size_x=patch_size_x,
            patch_size_y=patch_size_y,
            # Image resize ratio
            resize_ratio=resize_ratio,
            interpolation=interpolation,
            # Training vs Testing mode
            train=False,
        )
        img_transforms = transforms.Compose([
            transforms.Normalize(eval_dataset.mean, eval_dataset.std),
        ])

        # DataLoaders
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(eval_dataset),
            num_workers=num_workers,
            # This will make it go faster if it is loaded into a GPU
            pin_memory=True,
        )

        # Make a blank prediction image
        pred_image = np.zeros(eval_dataset.resized_size, dtype=np.float32).T
        print(f"Pred image {subtest_name} shape: {pred_image.shape}")
        print(f"Pred image min: {pred_image.min()}, max: {pred_image.max()}")

        # score = 0
        for i, patch in enumerate(tqdm(eval_dataloader, postfix=f"Eval {subtest_name}")):
            patch = patch.to(device)
            patch = img_transforms(patch)
            with torch.no_grad():
                preds = model(patch)
                preds = torch.sigmoid(preds)
                # score += dice_score(pred, label)

            # Iterate through each image and prediction in the batch
            for j, pred in enumerate(preds):
                pixel_index = eval_dataset.mask_indices[i * batch_size + j]
                pred_image[pixel_index[0], pixel_index[1]] = pred

        # # Score is average dice score for all batches
        # score /= len(eval_dataloader)
        # if writer:
        #     writer.add_scalar(f'Dice/{subtest_name}/eval', score)

        if writer is not None:
            print("Writing prediction image to TensorBoard...")
            # Add batch dimmension to pred_image for Tensorboard
            writer.add_image(f'pred_{subtest_name}',
                             np.expand_dims(pred_image, axis=0))

        # Resize pred_image to original size
        img = Image.fromarray(pred_image * 255).convert('1')
        img = img.resize((
            eval_dataset.original_size[0],
            eval_dataset.original_size[1],
        ), resample=INTERPOLATION_MODES[interpolation])

        if save_pred_img:
            print("Saving prediction image...")
            _image_filepath = os.path.join(
                output_dir, f"pred_{subtest_name}.png")
            img.save(_image_filepath)

        if postproc_kernel is not None:
            print("Postprocessing...")
            # Erosion then Dilation
            img = img.filter(ImageFilter.MinFilter(postproc_kernel))
            img = img.filter(ImageFilter.MaxFilter(postproc_kernel))

        if save_pred_img:
            print("Saving prediction image...")
            _image_filepath = os.path.join(
                output_dir, f"pred_{subtest_name}_post.png")
            img.save(_image_filepath)

        if save_submit_csv:
            print("Saving submission csv...")
            starts_ix, lengths = image_to_rle(
                np.array(img), threshold=threshold)
            inklabels_rle = " ".join(
                map(str, sum(zip(starts_ix, lengths), ())))
            with open(submission_filepath, 'a') as f:
                f.write(f"{subtest_name},{inklabels_rle}\n")

    if save_pred_img and save_submit_csv:
        save_rle_as_image(submission_filepath, output_dir,
                          subtest_name, pred_image.shape)


def eval_from_episode_dir(
    episode_dir: str = None,
    output_dir: str = None,
    hparams_filename: str = 'hparams.yaml',
    weights_filename: str = 'model.pth',
    **kwargs,
):
    # Get hyperparams from text file
    _hparams_filepath = os.path.join(episode_dir, hparams_filename)
    with open(_hparams_filepath, 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    hparams['output_dir'] = output_dir
    _weights_filepath = os.path.join(episode_dir, weights_filename)
    # Merge kwargs with hparams, kwargs takes precedence
    hparams = {**hparams, **kwargs}
    print(f"Hyperparams:\n{pprint.pformat(hparams)}\n")
    eval(
        weights_filepath=_weights_filepath,
        **hparams,
    )


def sweep_episode(hparams) -> float:

    # Print hyperparam dict with logging
    print(f"\n\nHyperparams:\n\n{pprint.pformat(hparams)}\n\n")

    # Create directory based on run_name
    run_name: str = str(uuid.uuid4())[:8]
    hparams['output_dir'] = os.path.join(hparams['output_dir'], run_name)
    os.makedirs(hparams['output_dir'], exist_ok=True)

    # HACK: Simpler parameters based
    # split input size string into 3 integers
    _input_size = [int(x) for x in hparams['input_size'].split('.')]
    hparams['patch_size_x'] = _input_size[0]
    hparams['patch_size_y'] = _input_size[1]
    hparams['slice_depth'] = _input_size[2]

    # Save hyperparams to file with YAML
    with open(os.path.join(hparams['output_dir'], 'hparams.yaml'), 'w') as f:
        yaml.dump(hparams, f)

    try:
        writer = SummaryWriter(log_dir=hparams['output_dir'])
        # Train and evaluate a TFLite model
        score, model = train(
            **hparams,
            writer=writer,
        )
        writer.add_hparams(hparams, {'dice_score': score})
        del hparams['model']
        # Eval takes a while, make sure you actually want to do this.
        eval(
            **hparams,
            model=model,
            writer=writer,
        )
        writer.close()
    except Exception as e:
        print(f"\n\n Model Training FAILED with \n{e}\n\n")
        score = 0
    # Maximize score is minimize negative score
    return -score
