import csv
import gc
import os
import pprint
from io import StringIO
from typing import Dict, Tuple, Union
import math

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFilter
from torch.optim.lr_scheduler import LambdaLR
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.modeling import (ImageEncoderViT, MaskDecoder,
                                       PromptEncoder, Sam)
from segment_anything.utils.amg import (MaskData, batched_mask_to_box,
                                        build_point_grid)
from tensorboardX import SummaryWriter
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
        # Expected slices per fragment
        crop_size: Tuple[int] = (256, 256),
        encoder_size: Tuple[int] = (1024, 1024),
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
        self.encoder_size = encoder_size
        self.interp = interp

        # Pixel stats for ir image, only for values inside mask
        _pixel_stats_filepath = os.path.join(data_dir, pixel_stat_filename)
        with open(_pixel_stats_filepath, "r") as f:
            pixel_stats = yaml.safe_load(f)
        self.pixel_mean = pixel_stats["raw"]["mean"]
        self.pixel_std = pixel_stats["raw"]["std"]
        # TODO: Need to re-run pixel stat script
        # self.pixel_mean = pixel_stats["mask"]["mean"]
        # self.pixel_std = pixel_stats["mask"]["std"]

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
            self.labels = np.array(labels_image, dtype=np.uint8) / 255.0

        # Open Slices into numpy array
        self.crop_depth = max_depth - min_depth
        self.fragment = np.zeros((
            self.crop_depth,
            self.original_size[0],
            self.original_size[1],
        ),
                                 dtype=np.float32)
        _slice_dir = os.path.join(data_dir, slices_dir_filename)
        for i in tqdm(range(min_depth, max_depth),
                      postfix='converting slices'):
            _slice_filepath = os.path.join(_slice_dir, f"{i:02d}.tif")
            slice_img = Image.open(_slice_filepath).convert("F")
            if self.resize != 1.0:
                slice_img = slice_img.resize(self.resized_size[::-1], resample=INTERPOLATION_MODES[interp])
            self.fragment[i, :, :] = np.array(slice_img) / 65535.0
        # Pad the fragment using crop size
        self.fragment = np.pad(self.fragment, ((0, 0), (crop_size[0], crop_size[0]), (crop_size[1], crop_size[1])), mode='constant', constant_values=0.0)

        # First index where mask is 1
        self.sample_points = np.where(mask == 255)

    def __len__(self):
        return len(self.sample_points[0])

    def __getitem__(self, idx):
        crop_center_height = self.sample_points[0][idx]
        crop_center_width = self.sample_points[1][idx]
        crop_half_height = self.crop_size[0] // 2
        crop_half_width = self.crop_size[1] // 2
        # Account for padding
        crop_center_height_pad = self.crop_size[0] + crop_center_height
        crop_center_width_pad = self.crop_size[1] + crop_center_width
        # Crop the fragment
        crop = self.fragment[:, 
            crop_center_height_pad - crop_half_height:crop_center_height_pad + crop_half_height,
            crop_center_width_pad - crop_half_width:crop_center_width_pad + crop_half_width,
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
        # tiled_image = (tiled_image - self.pixel_mean) / self.pixel_std

        if self.train:
            return tiled_image, self.labels[crop_center_height, crop_center_width]
        else:
            return tiled_image


class ClassyModel(nn.Module):
    def __init__(self,
                 image_encoder,
                 num_channels=256,
                 hidden_dim1=128,
                 hidden_dim2=64,
                 dropout_prob=0.2):
        super(ClassyModel, self).__init__()
        # Outputs a (batch_size, 256, 64, 64)
        self.image_encoder = image_encoder

        # Add the classifier head
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(num_channels, hidden_dim1), nn.LayerNorm(hidden_dim1),
            nn.GELU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim1, hidden_dim2), nn.LayerNorm(hidden_dim2),
            nn.GELU(), nn.Dropout(dropout_prob), nn.Linear(hidden_dim2, 1))

    def forward(self, x):
        x = self.image_encoder(x)  # Get features from the pre-trained model
        x = self.classifier_head(
            x)  # Pass the features through the classifier head
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
    save_model: bool = True,
    # Training
    device: str = None,
    num_samples_train: int = 2,
    num_samples_valid: int = 2,
    num_epochs: int = 2,
    warmup_epochs: int = 0,
    batch_size: int = 1,
    threshold: float = 0.4,
    optimizer: str = "adam",
    lr: float = 1e-5,
    lr_sched = "cosine",
    wd: float = 1e-4,
    writer=None,
    log_images: bool = False,
    # Dataset
    curriculum: str = "1",
    resize: float = 1.0,
    crop_size: Tuple[int] = (3, 256, 256),
    min_depth: int = 0,
    max_depth: int = 60,
    **kwargs,
):
    device = get_device(device)
    # device = "cpu"
    sam_model = sam_model_registry[model](checkpoint=weights_filepath)
    model = ClassyModel(sam_model.image_encoder)
    model.train()
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    if lr_sched == "cosine":
        _f = lambda x: warmup_cosine_annealing(x, num_epochs, warmup_epochs, eta_min=0.1 * lr, eta_max=lr)
    elif lr_sched == "gamma":
        _f = lambda x: warmup_gamma(x, num_epochs, warmup_epochs, gamma=0.1, eta_min=0.1 * lr, eta_max=lr)
    else:
        _f = lambda x: lr
    lr_sched = LambdaLR(optimizer, _f)

    train_step = 0
    best_score_dict: Dict[str, float] = {}
    for epoch in range(num_epochs):
        print(f"\n\n --- Epoch {epoch+1} of {num_epochs} --- \n\n")
        for phase, data_dir, num_samples in [
            ("Train", train_dir, num_samples_train),
            ("Valid", valid_dir, num_samples_valid),
        ]:
            for _dataset_id in curriculum:
                _dataset_filepath = os.path.join(data_dir, _dataset_id)
                print(f"{phase} on {_dataset_filepath} ...")
                _score_name = f"Dice/{phase}/{_dataset_id}"
                if _score_name not in best_score_dict:
                    best_score_dict[_score_name] = 0
                _dataset = TiledDataset(
                    data_dir=_dataset_filepath,
                    crop_size=crop_size,
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
                _loader = tqdm(_dataloader)
                score = 0
                for images, labels in _loader:
                    train_step += 1
                    if writer and log_images:
                        writer.add_images(f"input-images/{phase}/{_dataset_id}",
                                          images, train_step)
                        writer.flush()
                        # writer.add_images(f"input-labels/{phase}/{_dataset_id}",
                        #                   labels * 255, train_step)
                    images = images.to(dtype=torch.float32, device=device)
                    labels = labels.to(dtype=torch.float32, device=device)
                    preds = model(images)
                    loss = loss_fn(preds, labels.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    _loss_name = f"{loss_fn.__class__.__name__}/{phase}/{_dataset_id}"
                    _loader.set_postfix_str(f"{_loss_name}: {loss.item():.4f}")
                    if writer:
                        writer.add_scalar(_loss_name, loss.item(), train_step)
                    with torch.no_grad():
                        score += dice_score(preds, torch.sigmoid(labels) > threshold)
                score /= len(_dataloader)
                if writer:
                    writer.add_scalar(f"Dice/{phase}/{_dataset_id}", score, train_step)
                # Overwrite best score if it is better
                if score > best_score_dict[_score_name]:
                    print(f"New best score! {score:.4f} ")
                    print(f"(was {best_score_dict[_score_name]:.4f})")
                    best_score_dict[_score_name] = score
                    if save_model:
                        _model_filepath = os.path.join(
                            output_dir,
                            f"model_{run_name}_best_{_dataset_id}.pth")
                        print(f"Saving model to {_model_filepath}")
                        torch.save(model.state_dict(), _model_filepath)
        # Flush writer every epoch
        writer.flush()
    writer.close()
    return best_score_dict


def eval(
    output_dir: str = None,
    eval_dir: str = None,
    # Model
    model: str = "vit_b",
    weights_filepath: str = "path/to/model.pth",
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
    # Dataset
    eval_on: str = '123',
    resize: float = 1.0,
    crop_size: Tuple[int] = (3, 256, 256),
    min_depth: int = 0,
    max_depth: int = 60,
    **kwargs,
):
    device = get_device(device)

    # device = "cpu"
    sam_model = sam_model_registry[model](checkpoint=weights_filepath)
    model = ClassyModel(sam_model.image_encoder)
    model.train()
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    model.to(device=device)
    model = model.eval()

    if save_submit_csv:
        submission_filepath = os.path.join(output_dir, 'submission.csv')
        with open(submission_filepath, 'w') as f:
            # Write header
            f.write("Id,Predicted\n")

    best_score_dict: Dict[str, float] = {}
    for _dataset_id in eval_on:
        _dataset_filepath = os.path.join(eval_dir, _dataset_id)
        print(f"Evaluate on {_dataset_filepath} ...")
        _score_name = f"Dice/Eval/{_dataset_id}"
        if _score_name not in best_score_dict:
            best_score_dict[_score_name] = 0
        _dataset = TiledDataset(
            data_dir=_dataset_filepath,
            resize=resize,
            crop_size=crop_size,
            min_depth=min_depth,
            max_depth=max_depth,
            train=True,
            device=device,
        )
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=batch_size,
            sampler = SequentialSampler(_dataset),
            pin_memory=True,
        )
        
        # Make a blank prediction image
        pred_image = np.zeros(_dataset.resized_size, dtype=np.float32).T
        print(f"Pred image {_dataset_id} shape: {pred_image.shape}")
        print(f"Pred image min: {pred_image.min()}, max: {pred_image.max()}")

        _loader = tqdm(_dataloader, postfix=f"Eval {_dataset_id}")
        for i, (images, labels) in enumerate(_loader):
            train_step += 1
            if writer and log_images:
                writer.add_images(f"input-image/Eval/{_dataset_id}",
                                    images * 255, train_step)
            with torch.no_grad():
                images = images.to(dtype=torch.float32, device=device)
                labels = labels.to(dtype=torch.float32, device=device)
                preds = model(images)
                preds = torch.sigmoid(preds)

            # Iterate through each image and prediction in the batch
            for j, pred in enumerate(preds):
                pixel_index = _dataset.mask_indices[i * batch_size + j]
                pred_image[pixel_index[0], pixel_index[1]] = pred

        if writer is not None:
            print("Writing prediction image to TensorBoard...")
            writer.add_image(f'pred_{_dataset_id}',
                             np.expand_dims(pred_image, axis=0))

        if save_histograms:
            # Save histogram of predictions as image
            print("Saving prediction histogram...")
            _num_bins = 100
            np_hist, _ = np.histogram(pred_image.flatten(),
                                      bins=_num_bins,
                                      range=(0, 1),
                                      density=True)
            np_hist = np_hist / np_hist.sum()
            hist = np.zeros((_num_bins, 100), dtype=np.uint8)
            for bin in range(_num_bins):
                _height = int(np_hist[bin] * 100)
                hist[bin, 0:_height] = 255
            hist_img = Image.fromarray(hist)
            _histogram_filepath = os.path.join(
                output_dir, f"pred_{_dataset_id}_histogram.png")
            hist_img.save(_histogram_filepath)

            if writer is not None:
                print("Writing prediction histogram to TensorBoard...")
                writer.add_histogram(f'pred_{_dataset_id}', pred_image)

        # Resize pred_image to original size
        img = Image.fromarray(pred_image * 255).convert('1')
        img = img.resize((
            _dataset_id.original_size[0],
            _dataset_id.original_size[1],
        ),
                         resample=Image.BICUBIC)

        if save_pred_img:
            print("Saving prediction image...")
            _image_filepath = os.path.join(output_dir,
                                           f"pred_{_dataset_id}.png")
            img.save(_image_filepath)

        if postproc_kernel is not None:
            print("Postprocessing...")
            # Erosion then Dilation
            img = img.filter(ImageFilter.MinFilter(postproc_kernel))
            img = img.filter(ImageFilter.MaxFilter(postproc_kernel))

        if save_pred_img:
            print("Saving prediction image...")
            _image_filepath = os.path.join(output_dir,
                                           f"pred_{_dataset_id}_post.png")
            img.save(_image_filepath)

        if save_submit_csv:
            print("Saving submission csv...")
            img = np.array(img).flatten()
            # Convert image to binary using threshold
            img = np.where(img > threshold, 1, 0).astype(np.uint8)
            inklabels_rle_fast = image_to_rle_fast(img)
            with open(submission_filepath, 'a') as f:
                f.write(f"{_dataset_id},{inklabels_rle_fast}\n")

    if save_pred_img and save_submit_csv:
        save_rle_as_image(submission_filepath, output_dir, _dataset_id,
                          pred_image.shape)


def eval_from_episode_dir(
    episode_dir: str = None,
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
    eval(
        output_dir=output_dir,
        weights_filepath=_weights_filepath,
        **hparams,
    )
