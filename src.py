import csv
import gc
import os
import pprint
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

class TiledDataset(Dataset):
    def __init__(
        self,
        # Directory containing the dataset
        data_dir: str,
        # Number of random crops to take from fragment volume
        dataset_size: int = 2,
        # Filenames of the images we'll use
        image_mask_filename="mask.png",
        image_labels_filename="inklabels.png",
        slices_dir_filename="surface_volume",
        ir_image_filename="ir.png",
        pixel_stat_filename="pixel_stats.yaml",
        # Expected slices per fragment
        crop_size: Tuple[int] = (3, 256, 256),
        encoder_size: Tuple[int] = (3, 1024, 1024),
        # Depth into scan to take label from
        min_depth: int = 0,
        max_depth: int = 42,
        # Training vs Testing mode
        train: bool = True,
        # Device to use
        device: str = "cuda",
    ):
        print(f"Making TiledDataset Dataset from {data_dir}")
        self.dataset_size = dataset_size
        self.train = train
        self.device = device

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
        self.mask = np.array(mask_image, dtype=np.uint8)
        self.original_size = self.mask.shape
        self.crop_size = crop_size
        self.encoder_size = encoder_size
        # Open Label image
        if self.train:
            _image_labels_filepath = os.path.join(data_dir, image_labels_filename)
            labels_image = Image.open(_image_labels_filepath).convert("L")
            self.labels = np.array(labels_image, dtype=np.uint8)

        # Slices
        self.num_slices = max_depth - min_depth
        # Open Slices into numpy array
        self.fragment = np.zeros((
            self.num_slices,
            self.original_size[0],
            self.original_size[1],
        ), dtype=np.float32)
        _slice_dir = os.path.join(data_dir, slices_dir_filename)
        for i in tqdm(range(min_depth, max_depth), postfix='converting slices'):
            _slice_filepath = os.path.join(_slice_dir, f"{i:02d}.tif")
            slice_img = Image.open(_slice_filepath).convert("F")
            self.fragment[i, :, :] = np.array(slice_img) / 65535.0
        # Pin the fragment to the device
        # self.fragment = torch.from_numpy(self.fragment).to(device=self.device)

        # Sample random crops within the image
        self.indices = np.zeros((dataset_size, 2, 2), dtype=np.int64)
        for i in range(dataset_size):
            # Select a random starting point
            start_height = np.random.randint(
                0, self.original_size[0] - self.crop_size[1])
            start_width = np.random.randint(
                0, self.original_size[1] - self.crop_size[2])
            self.indices[i, 0, :] = [start_height, start_width]
            # End point is start point + crop size
            self.indices[i, 1, :] = [
                start_height + self.crop_size[1],
                start_width + self.crop_size[2],
            ]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Start and End points for the crop in pixel space
        start = self.indices[idx, 0, :]
        end = self.indices[idx, 1, :]

        # Sub-volume
        tensor = self.fragment[:, start[0]:end[0], start[1]:end[1]]

        # Slice the tensor along the first dimension
        tiles = np.split(tensor, len(tensor) // 3, axis=0)

        # Calculate tileing dimmensions (square image)
        n_rows = int(np.ceil(np.sqrt(len(tiles))))
        n_cols = int(np.ceil(len(tiles) / n_rows))

        # Initialize a larger array of 3xNxN
        tiled_image = np.zeros((3, n_rows * self.crop_size[1], n_cols * self.crop_size[2]))

        # Lay out the tiles left to right, top to bottom into the larger array
        for idx, tile in enumerate(tiles):
            row = idx // n_cols
            col = idx % n_cols
            tiled_image[:,
                row * self.crop_size[1]:(row + 1) * self.crop_size[1],
                col * self.crop_size[2]:(col + 1) * self.crop_size[2],
            ] = tile

        # Normalize image
        tiled_image = (tiled_image - self.pixel_mean) / self.pixel_std

        if self.train:
            label = self.labels[
                start[0] + self.crop_size[1] // 2,
                start[1] + self.crop_size[2] // 2,
            ]
            return tiled_image, label
        else:
            return tiled_image


class ClassyModel(nn.Module):
    def __init__(self, image_encoder, num_channels=256, hidden_dim1=128, hidden_dim2=64, dropout_prob=0.2):
        super(ClassyModel, self).__init__()
        # Outputs a (batch_size, 256, 64, 64)
        self.image_encoder = image_encoder

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
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        x = self.image_encoder(x)  # Get features from the pre-trained model
        x = self.classifier_head(x)  # Pass the features through the classifier head
        return x


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
    batch_size: int = 1,
    optimizer: str = "adam",
    lr: float = 1e-5,
    wd: float = 1e-4,
    writer=None,
    log_images: bool = False,
    # Dataset
    curriculum: str = "1",
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
    # TODO: Learning rate scheduler
    # TODO: Learning rate warmup

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
                    dataset_size=num_samples,
                    crop_size=crop_size,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    train=True,
                    device=device,
                )
                _dataloader = DataLoader(
                    dataset=_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                )
                _loader = tqdm(_dataloader)
                score = 0
                for images, labels in _loader:
                    train_step += 1
                    if writer and log_images:
                        writer.add_images(f"input-image/{phase}/{_dataset_id}",
                                          images * 255, train_step)
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
                        writer.add_scalar(f"{_loss_name}", loss.item(),
                                          train_step)

                    score += dice_score(preds, labels)
                score /= len(_dataloader)
                if writer:
                    writer.add_scalar(f"Dice/{phase}", score, train_step)
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
    threshold: float = 0.5,
    postproc_kernel: int = 3,
    log_images: bool = False,
    save_pred_img: bool = True,
    save_submit_csv: bool = False,
    save_histograms: bool = False,
    writer=None,
    # Dataset
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

        for i, patch in enumerate(tqdm(eval_dataloader, postfix=f"Eval {subtest_name}")):
            patch = patch.to(device)
            patch = img_transforms(patch)
            with torch.no_grad():
                preds = model(patch)
                preds = torch.sigmoid(preds)

            # Iterate through each image and prediction in the batch
            for j, pred in enumerate(preds):
                pixel_index = eval_dataset.mask_indices[i * batch_size + j]
                pred_image[pixel_index[0], pixel_index[1]] = pred

        if writer is not None:
            print("Writing prediction image to TensorBoard...")
            writer.add_image(f'pred_{subtest_name}', np.expand_dims(pred_image, axis=0))

        if save_histograms:
            # Save histogram of predictions as image
            print("Saving prediction histogram...")
            _num_bins = 100
            np_hist, _ = np.histogram(pred_image.flatten(), bins=_num_bins, range=(0, 1), density=True)
            np_hist = np_hist / np_hist.sum()
            hist = np.zeros((_num_bins, 100), dtype=np.uint8)
            for bin in range(_num_bins):
                _height = int(np_hist[bin] * 100)
                hist[bin, 0:_height] = 255
            hist_img = Image.fromarray(hist)
            _histogram_filepath = os.path.join(
                output_dir, f"pred_{subtest_name}_histogram.png")
            hist_img.save(_histogram_filepath)

            if writer is not None:
                print("Writing prediction histogram to TensorBoard...")
                writer.add_histogram(f'pred_{subtest_name}', pred_image)

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
            img = np.array(img).flatten()
            # Convert image to binary using threshold
            img = np.where(img > threshold, 1, 0).astype(np.uint8)
            # Convert image to RLE
            # start_time = time.time()
            # inklabels_rle_original = image_to_rle(img)
            # print(f"RLE conversion (ORIGINAL) took {time.time() - start_time:.2f} seconds")
            start_time = time.time()
            inklabels_rle_fast = image_to_rle_fast(img)
            print(f"RLE conversion (FAST) took {time.time() - start_time:.2f} seconds")
            # assert inklabels_rle_original == inklabels_rle_fast, "RLE conversion is not the same!"
            with open(submission_filepath, 'a') as f:
                f.write(f"{subtest_name},{inklabels_rle_fast}\n")

    if save_pred_img and save_submit_csv:
        save_rle_as_image(submission_filepath, output_dir,
                          subtest_name, pred_image.shape)


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
