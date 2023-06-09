import csv
import gc
import math
import os
import sys
import pprint
import time
from io import StringIO
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFilter
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.modeling import (ImageEncoderViT, MaskDecoder,
                                       PromptEncoder, Sam)
from segment_anything.utils.amg import (MaskData, batched_mask_to_box,
                                        build_point_grid)
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torchvision import transforms
from tqdm import tqdm


def get_device(device: str = None):
    if device == None or device == "gpu":
        if torch.cuda.is_available():
            print("Using GPU")
            print("Clearing GPU memory")
            torch.cuda.empty_cache()
            gc.collect()
            return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def image_to_rle_fast(img):
    img[0] = 0
    img[-1] = 0
    runs = np.where(img[1:] != img[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    f = StringIO()
    np.savetxt(f, runs.reshape(1, -1), delimiter=" ", fmt="%d")
    return f.getvalue().strip()


def save_rle_as_image(rle_csv_path, output_dir, subtest_name, image_shape):
    with open(rle_csv_path, "r") as csvfile:
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
            img = Image.fromarray(img * 255).convert("1")
            _image_filepath = os.path.join(output_dir,
                                           f"pred_{subtest_name}_rle.png")
            img.save(_image_filepath)


def dice_score(preds, label, beta=0.5, epsilon=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.flatten()
    label = label.flatten()
    tp = preds[label == 1].sum()
    fp = preds[label == 0].sum()
    fn = label.sum() - tp
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    _score = (1 + beta * beta) * (p * r) / (beta * beta * p + r + epsilon)
    return _score


# Types of interpolation used in resizing
INTERPOLATION_MODES = {
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'nearest': Image.NEAREST,
}

class TiledDataset(Dataset):
    def __init__(
        self,
        # Directory containing the dataset
        data_dir: str,
        # Filenames of the images we'll use
        image_mask_filename="mask.png",
        image_labels_filename="inklabels.png",
        slices_dir_filename="surface_volume",
        ir_image_filename="ir.png",
        pixel_stat_filename="pixel_stats.yaml",
        resize: float = 1.0,
        interp: str = "bilinear",
        # Pixel normalization to use
        pixel_norm: str = "mask",
        # Expected slices per fragment
        crop_size: Tuple[int] = (256, 256),
        encoder_size: Tuple[int] = (1024, 1024),
        label_size: Tuple[int] = (8, 8),
        # Depth into scan to take label from
        min_depth: int = 0,
        max_depth: int = 42,
        # Training vs Testing mode
        train: bool = True,
        # Device to use
        device: str = "cuda",
    ):
        print(f"Making TiledDataset Dataset from {data_dir}")
        self.train = train
        self.device = device
        self.crop_size = crop_size
        self.crop_half_height = self.crop_size[0] // 2
        self.crop_half_width = self.crop_size[1] // 2
        self.encoder_size = encoder_size
        self.label_size = label_size
        self.label_half_height = self.label_size[0] // 2
        self.label_half_width = self.label_size[1] // 2
        self.interp = interp

        # Pixel stats for ir image, only for values inside mask
        _pixel_stats_filepath = os.path.join(data_dir, pixel_stat_filename)
        with open(_pixel_stats_filepath, "r") as f:
            pixel_stats = yaml.safe_load(f)
        self.pixel_mean = pixel_stats[pixel_norm]["mean"]
        self.pixel_std = pixel_stats[pixel_norm]["std"]

        # Open Mask image
        _image_mask_filepath = os.path.join(data_dir, image_mask_filename)
        mask_image = Image.open(_image_mask_filepath).convert("L")
        self.original_size = mask_image.height, mask_image.width
        self.resize = resize
        self.resized_size = [
            int(self.original_size[0] * resize), 
            int(self.original_size[1] * resize),
        ]
        if self.resize != 1.0:
            mask_image = mask_image.resize(self.resized_size[::-1], resample=INTERPOLATION_MODES[interp])
        mask = np.array(mask_image, dtype=np.uint8)
        # Open Label image
        if self.train:
            _image_labels_filepath = os.path.join(data_dir,image_labels_filename)
            labels_image = Image.open(_image_labels_filepath).convert("L")
            if self.resize != 1.0:
                labels_image = labels_image.resize(self.resized_size[::-1], resample=INTERPOLATION_MODES[interp])
            labels = np.array(labels_image, dtype=np.uint8) / 255.0

        # Open Slices into numpy array
        self.crop_depth = max_depth - min_depth
        fragment = np.zeros((
            self.crop_depth,
            self.original_size[0],
            self.original_size[1],
        ),
                                 dtype=np.float32)
        _slice_dir = os.path.join(data_dir, slices_dir_filename)
        _loader = tqdm(
            range(min_depth, max_depth),
            postfix=f"Opening Slices",
            position=0, leave=True)
        for i in _loader:
            _slice_filepath = os.path.join(_slice_dir, f"{i:02d}.tif")
            slice_img = Image.open(_slice_filepath).convert("F")
            if self.resize != 1.0:
                slice_img = slice_img.resize(self.resized_size[::-1], resample=INTERPOLATION_MODES[interp])
            fragment[i, :, :] = np.array(slice_img) / 65535.0

        if train:
            # Sample evenly from label and background
            _label_points = np.where(labels > 0)
            _bg_points =  np.where((labels == 0) & (mask == 255))
            _num_points = min(len(_label_points[0]), len(_bg_points[0]))
            _label_points = (
                _label_points[0][:_num_points],
                _label_points[1][:_num_points],
            )
            _bg_points = (
                _bg_points[0][:_num_points],
                _bg_points[1][:_num_points],
            )
            self.sample_points = (
                np.concatenate((_label_points[0], _bg_points[0])),
                np.concatenate((_label_points[1], _bg_points[1])),
            )
        else:
            self.sample_points = np.where(mask == 255)

        # Pad the fragment using crop size
        self.fragment = np.pad(
            fragment,
            ((0, 0), (self.crop_half_height, self.crop_half_height),
                (self.crop_half_width, self.crop_half_width)),
            mode='constant', constant_values=0.,
        )

        if train:
            # Pad the labels using label size
            self.labels = np.pad(
                labels,
                ((self.label_half_height, self.label_half_height),
                    (self.label_half_width, self.label_half_width)),
                mode="constant", constant_values=0.,
            )

    def __len__(self):
        return len(self.sample_points[0])

    def __getitem__(self, idx):
        crop_center_height = self.sample_points[0][idx]
        crop_center_width = self.sample_points[1][idx]

        # Account for padding
        crop_center_height_pad = self.crop_half_height + crop_center_height
        crop_center_width_pad = self.crop_half_width + crop_center_width
        # Crop the fragment
        crop = self.fragment[:, 
            crop_center_height_pad - self.crop_half_height:crop_center_height_pad + self.crop_half_height,
            crop_center_width_pad - self.crop_half_width:crop_center_width_pad + self.crop_half_width,
        ]

        # Calculate tileing dimmensions (square image)
        n_tiles = int(np.ceil(self.crop_depth / 3.0))
        n_rows = int(np.ceil(np.sqrt(n_tiles)))
        n_cols = int(np.ceil(n_tiles / n_rows))
        
        # Initialize a larger array of 3xNxN
        tiled_image = np.zeros((3, n_rows * self.crop_size[0], n_cols * self.crop_size[1]), dtype=np.float32)
        for idx in range(n_tiles):
            row = idx // n_cols
            col = idx % n_cols
            # Fill in the 3xNxN array with the cropped slices
            tiled_image[:, row * self.crop_size[0]:(row + 1) * self.crop_size[0], col * self.crop_size[1]:(col + 1) * self.crop_size[1]] = crop[idx * 3:(idx + 1) * 3, :, :]

        # # Resize to encoder size
        if self.resize != 1.0 or \
            tiled_image.shape[1] != self.encoder_size[0] \
                or tiled_image.shape[2] != self.encoder_size[1]:
            # TODO: This just seems to fuck it up, loosing information
            tiled_image = tiled_image.transpose((1, 2, 0)) * 255.0
            tiled_image = Image.fromarray(tiled_image, 'RGB')
            tiled_image = tiled_image.resize(self.encoder_size[::-1], resample=INTERPOLATION_MODES[self.interp])
            tiled_image = np.array(tiled_image, dtype=np.float32) / 255.0
            tiled_image = np.transpose(tiled_image, (2, 0, 1))

        # Normalize image
        tiled_image = (tiled_image - self.pixel_mean) / self.pixel_std

        # Centerpoint of crop in pixel space
        centerpoint = (crop_center_height, crop_center_width)

        if self.train:
            # Account for padding
            crop_center_height_pad = crop_center_height + self.label_half_height
            crop_center_width_pad = crop_center_width + self.label_half_width
            return tiled_image, centerpoint, self.labels[
                crop_center_height_pad - self.label_half_height : crop_center_height_pad + self.label_half_height,
                crop_center_width_pad - self.label_half_width : crop_center_width_pad + self.label_half_width,
            ]
        else:
            return tiled_image, centerpoint


class ClassyModel(nn.Module):
    def __init__(self,
                 image_encoder: nn.Module = None,
                 label_size: Tuple[int] = (8, 8),
                 num_channels=256,
                 hidden_dim1=128,
                 hidden_dim2=64,
                 dropout_prob=0.2):
        super(ClassyModel, self).__init__()
        # Outputs a (batch_size, 256, 64, 64)
        self.image_encoder = image_encoder
        self.label_size = label_size

        # Add the classifier head
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            # Go from flat 1d hidden_dim (B, hidden_dim2) to label size (B, 8, 8)
            nn.Linear(hidden_dim2, label_size[0] * label_size[1]),
        )

    def forward(self, x):
        x = self.image_encoder(x)
        x = self.classifier_head(x)
        # reshape output to labels size
        x = x.view(-1, self.label_size[0], self.label_size[1])
        return x


def warmup_cosine_annealing(epoch, total_epochs, warmup_epochs, eta_min=0, eta_max=1):
    if epoch < warmup_epochs:
        return (eta_max - eta_min) * (epoch / warmup_epochs) + eta_min
    else:
        T_cur = epoch - warmup_epochs
        T_max = total_epochs - warmup_epochs
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + torch.cos(torch.tensor(T_cur / T_max * math.pi)))


def warmup_gamma(epoch, total_epochs, warmup_epochs, gamma=0.1, eta_min=0, eta_max=1):
    if epoch < warmup_epochs:
        return (eta_max - eta_min) * (epoch / warmup_epochs) + eta_min
    else:
        T_cur = epoch - warmup_epochs
        T_max = total_epochs - warmup_epochs
        return eta_min + (eta_max - eta_min) * (gamma ** (T_cur / T_max))


def train_valid(
    run_name: str = "testytest",
    output_dir: str = None,
    train_dir: str = None,
    valid_dir: str = None,
    # Model
    model: str = "vit_b",
    weights_filepath: str = "path/to/model.pth",
    freeze: bool = True,
    save_model: bool = True,
    num_channels: int = 256,
    hidden_dim1: int = 128,
    hidden_dim2: int = 64,
    dropout_prob: int = 0.2,
    # Training
    device: str = None,
    num_samples_train: int = 2,
    num_samples_valid: int = 2,
    num_epochs: int = 2,
    warmup_epochs: int = 0,
    batch_size: int = 1,
    threshold: float = 0.3,
    optimizer: str = "adam",
    lr: float = 1e-5,
    lr_sched = "cosine",
    wd: float = 1e-4,
    writer=None,
    log_images: bool = False,
    # Dataset
    curriculum: str = "1",
    resize: float = 1.0,
    interp: str = "bilinear",
    pixel_norm: bool = "mask",
    crop_size: Tuple[int] = (3, 256, 256),
    label_size: Tuple[int] = (8, 8),
    min_depth: int = 0,
    max_depth: int = 60,
    **kwargs,
):
    device = get_device(device)
    # device = "cpu"
    sam_model = sam_model_registry[model](checkpoint=weights_filepath)
    model = ClassyModel(
        image_encoder=sam_model.image_encoder,
        label_size = label_size,
        num_channels = num_channels,
        hidden_dim1 = hidden_dim1,
        hidden_dim2 = hidden_dim2,
        dropout_prob = dropout_prob,
    )
    model.train()
    if freeze:
        print("Freezing image encoder")
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    if lr_sched == "cosine":
        _f = lambda x: warmup_cosine_annealing(x, num_epochs, warmup_epochs, eta_min=0.1 * lr, eta_max=lr)
    elif lr_sched == "gamma":
        _f = lambda x: warmup_gamma(x, num_epochs, warmup_epochs, gamma=0.9, eta_min=0.1 * lr, eta_max=lr)
    else:
        _f = lambda x: lr
    lr_sched = LambdaLR(optimizer, _f)

    step = 0
    best_score_dict: Dict[str, float] = {}
    for epoch in range(num_epochs):
        print(f"\n\n --- Epoch {epoch+1} of {num_epochs} --- \n\n")
        for phase, data_dir, num_samples in [
            ("Train", train_dir, num_samples_train),
            ("Valid", valid_dir, num_samples_valid),
        ]:
            for _dataset_id in curriculum:
                _dataset_filepath = os.path.join(data_dir, _dataset_id)
                _score_name = f"Dice/{phase}/{_dataset_id}"
                if _score_name not in best_score_dict:
                    best_score_dict[_score_name] = 0
                _dataset = TiledDataset(
                    data_dir=_dataset_filepath,
                    crop_size=crop_size,
                    label_size=label_size,
                    resize=resize,
                    interp=interp,
                    pixel_norm=pixel_norm,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    train=True,
                    device=device,
                )
                _dataloader = DataLoader(
                    dataset=_dataset,
                    batch_size=batch_size,
                    sampler = RandomSampler(
                        _dataset,
                        num_samples=num_samples,
                        # Generator with constant seed for reproducibility during validation
                        generator=torch.Generator().manual_seed(42) if phase == "Valid" else None,
                    ),
                    pin_memory=True,
                )
                # TODO: prevent tqdm from printing on every iteration
                _loader = tqdm(_dataloader, postfix=f"{phase}/{_dataset_id}/", position=0, leave=True)
                score = 0
                print(f"{phase} on {_dataset_filepath} ...")
                for images, centerpoint, labels in _loader:
                    step += 1
                    if writer and log_images:
                        writer.add_images(f"input-images/{phase}/{_dataset_id}", images, step)
                        writer.add_images(f"input-labels/{phase}/{_dataset_id}", labels.to(dtype=torch.uint8).unsqueeze(1) * 255, step)
                    images = images.to(dtype=torch.float32, device=device)
                    labels = labels.to(dtype=torch.float32, device=device)
                    preds = model(images)
                    loss = loss_fn(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    _loss_name = f"{loss_fn.__class__.__name__}/{phase}/{_dataset_id}"
                    _loader.set_postfix_str(f"{_loss_name}: {loss.item():.4f}")
                    if writer:
                        writer.add_scalar(_loss_name, loss.item(), step)
                    with torch.no_grad():
                        preds = (torch.sigmoid(preds) > threshold).to(dtype=torch.uint8)
                        score += dice_score(preds, labels).item()
                    if writer and log_images:
                        writer.add_images(f"output-preds/{phase}/{_dataset_id}", preds.unsqueeze(1) * 255, step)
                score /= len(_dataloader)
                if writer:
                    writer.add_scalar(f"Dice/{phase}/{_dataset_id}", score, step)
                # Overwrite best score if it is better
                if score > best_score_dict[_score_name]:
                    print(f"New best score! {score:.4f} ")
                    print(f"(was {best_score_dict[_score_name]:.4f})")
                    best_score_dict[_score_name] = score
                    if save_model:
                        _model_filepath = os.path.join(
                            output_dir,
                            f"model.pth")
                            #  f"model_{run_name}_best_{_dataset_id}.pth")
                        print(f"Saving model to {_model_filepath}")
                        torch.save(model.state_dict(), _model_filepath)
                # Flush ever batch
                writer.flush()
    return best_score_dict


def eval(
    output_dir: str = None,
    eval_dir: str = None,
    # Model
    model: str = "vit_b",
    weights_filepath: str = "path/to/model.pth",
    num_channels: int = 256,
    hidden_dim1: int = 128,
    hidden_dim2: int = 64,
    dropout_prob: int = 0.2,
    # Evaluation
    device: str = None,
    batch_size: int = 2,
    threshold: float = 0.5,
    postproc_kernel: int = 3,
    log_images: bool = False,
    save_pred_img: bool = True,
    save_submit_csv: bool = False,
    save_histograms: bool = False,
    writer=None,
    max_time_hours: float = 0.5,
    # Dataset
    eval_on: str = '123',
    resize: float = 1.0,
    interp: str = "bilinear",
    pixel_norm: str = "mask",
    max_num_samples_eval: int = 1000,
    crop_size: Tuple[int] = (256, 256),
    label_size: Tuple[int] = (8, 8),
    min_depth: int = 0,
    max_depth: int = 60,
    **kwargs,
):
    print(f"Eval will run for maximum of {max_time_hours} hours (PER DATASET)")
    max_time_seconds = max_time_hours * 60 * 60

    device = get_device(device)
    sam_model = sam_model_registry[model](checkpoint=None)
    model = ClassyModel(
        image_encoder=sam_model.image_encoder,
        label_size = label_size,
        num_channels = num_channels,
        hidden_dim1 = hidden_dim1,
        hidden_dim2 = hidden_dim2,
        dropout_prob = dropout_prob,
    )
    model = model.eval()
    with open(weights_filepath, "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    model.to(device=device)

    if save_submit_csv:
        submission_filepath = os.path.join(output_dir, 'submission.csv')
        with open(submission_filepath, 'w') as f:
            # Write header
            f.write("Id,Predicted\n")

    best_score_dict: Dict[str, float] = {}
    step = 0
    for _dataset_id in eval_on:
        _dataset_filepath = os.path.join(eval_dir, _dataset_id)
        print(f"Evaluate on {_dataset_filepath} ...")
        _score_name = f"Dice/Eval/{_dataset_id}"
        if _score_name not in best_score_dict:
            best_score_dict[_score_name] = 0
        _dataset = TiledDataset(
            data_dir=_dataset_filepath,
            crop_size=crop_size,
            label_size=label_size,
            resize=resize,
            interp=interp,
            pixel_norm=pixel_norm,
            min_depth=min_depth,
            max_depth=max_depth,
            train=False,
            device=device,
        )
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=batch_size,
            # sampler = SequentialSampler(_dataset),
            sampler = RandomSampler(
                _dataset,
                # TODO: Num samples just based on time, and average the predictions
                # thus making it iteratively better over time. You can maybe
                # condition the image on the previous prediction (aka the label is one of the tiles)
                # which you can emulate with the labels during training.
                # effectively doing a kind of pseudo-labeling, which is similar to the
                # original SAM approach.
                num_samples=max_num_samples_eval,
                # Generator with constant seed for reproducibility during eval
                generator=torch.Generator().manual_seed(42),
            ),
            pin_memory=True,
        )
        
        # Make a blank prediction image
        pred_image = np.zeros(_dataset.resized_size, dtype=np.float32)
        # Pad prediction with label size
        label_size_half = (label_size[0] // 2, label_size[1] // 2)
        pred_image = np.pad(pred_image,
            ((label_size_half[0], label_size_half[0]), (label_size_half[1], label_size_half[1])),
            mode='constant', constant_values=0,
        )

        time_start = time.time()
        time_elapsed = 0
        phase = "Eval"
        _loader = tqdm(_dataloader, postfix=f"{phase}/{_dataset_id}/", position=0, leave=True)
        for i, (images, centerpoint) in enumerate(_loader):
            step += 1
            if writer and log_images:
                writer.add_images(f"input-images/{phase}/{_dataset_id}", images, step)
            with torch.no_grad():
                images = images.to(dtype=torch.float32, device=device)
                preds = model(images)
                preds = torch.sigmoid(preds)
                if writer and log_images:
                    writer.add_images(f"output-preds/{phase}/{_dataset_id}", preds.unsqueeze(1) * 255, step)
            for i, pred in enumerate(preds):
                # centerpoint to padded pixel coords, and then size of label
                h_s = int(centerpoint[0][i].cpu() + label_size[0] // 2 - label_size[0] // 2)
                h_e = int(centerpoint[0][i].cpu() + label_size[0] // 2 + label_size[0] // 2)
                w_s = int(centerpoint[1][i].cpu() + label_size[1] // 2 - label_size[1] // 2)
                w_e = int(centerpoint[1][i].cpu() + label_size[1] // 2 + label_size[1] // 2)
                _existing_pred = pred_image[h_s:h_e, w_s:w_e]
                # Combine existing prediction with new prediction
                pred = pred.cpu().numpy()
                pred_image[h_s:h_e, w_s:w_e] = np.mean(np.stack([_existing_pred, pred]), axis=0)
            
            # Check if we have exceeded the time limit
            time_elapsed = time.time() - time_start
            if time_elapsed > max_time_seconds:
                print(f"Time limit exceeded for dataset {_dataset_id}")
                break

        # Remove padding from prediction
        pred_image = pred_image[label_size_half[0]:-label_size_half[0], label_size_half[1]:-label_size_half[1]]

        if writer is not None:
            print("Writing prediction image to TensorBoard...")
            writer.add_image(f'pred_{_dataset_id}', np.expand_dims(pred_image, axis=0) * 255, step)

        if save_histograms:
            # Save histogram of predictions as image
            print("Saving prediction histogram...")
            _num_bins = 100
            np_hist, _ = np.histogram(pred_image.flatten(),
                                      bins=_num_bins,
                                      range=(0, 1),
                                      density=True)
            np_hist = np_hist / np_hist.sum()
            hist = np.zeros((100, _num_bins), dtype=np.uint8)
            for bin in range(_num_bins):
                _height = int(np_hist[bin] * 100)
                hist[0:_height, bin] = 255
            hist_img = Image.fromarray(hist)
            _histogram_filepath = os.path.join(output_dir, f"pred_{_dataset_id}_hist.png")
            hist_img.save(_histogram_filepath)

            if writer is not None:
                print("Writing prediction histogram to TensorBoard...")
                writer.add_histogram(f'pred_{_dataset_id}', hist, step)

        # Resize pred_image to original size
        if resize != 1.0:
            img = Image.fromarray(pred_image * 255).convert('1')
            img = img.resize((
                _dataset_id.original_size[0],
                _dataset_id.original_size[1],
            ), resample=INTERPOLATION_MODES[interp])

        if save_pred_img:
            print("Saving prediction image...")
            img = Image.fromarray(pred_image * 255).convert('1')
            _image_filepath = os.path.join(output_dir, f"pred_{_dataset_id}.png")
            img.save(_image_filepath)

        # if postproc_kernel is not None:
        #     print("Postprocessing...")
        #     # Erosion then Dilation
        #     img = img.filter(ImageFilter.MinFilter(postproc_kernel))
        #     img = img.filter(ImageFilter.MaxFilter(postproc_kernel))

        # if save_pred_img:
        #     print("Saving prediction image...")
        #     _image_filepath = os.path.join(output_dir, f"pred_{_dataset_id}_post.png")
        #     img.save(_image_filepath)

        if save_submit_csv:
            print("Saving submission csv...")
            # Convert image to binary using threshold
            # img_thresh = np.where(pred_image > threshold, 1, 0).astype(np.uint8)
            pred_image[pred_image > threshold] = 1
            pred_image[pred_image <= threshold] = 0
            inklabels_rle_fast = image_to_rle_fast(pred_image)
            with open(submission_filepath, 'a') as f:
                f.write(f"{_dataset_id},{inklabels_rle_fast}\n")

    if save_pred_img and save_submit_csv:
        save_rle_as_image(submission_filepath, output_dir, _dataset_id,
                          pred_image.shape)


def eval_from_episode_dir(
    episode_dir: str = None,
    eval_dir: str = None,
    output_dir: str = None,
    weights_filename: str = "model.pth",
    hparams_filename: str = "hparams.yaml",
    **kwargs,
):
    # Get hyperparams from text file
    _hparams_filepath = os.path.join(episode_dir, hparams_filename)
    with open(_hparams_filepath, "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    _weights_filepath = os.path.join(episode_dir, weights_filename)
    # Merge kwargs with hparams, kwargs takes precedence
    hparams = {**hparams, **kwargs}
    print(f"Hyperparams:\n{pprint.pformat(hparams)}\n")
    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    eval(
        eval_dir=eval_dir,
        output_dir=output_dir,
        weights_filepath=_weights_filepath,
        **hparams,
    )
