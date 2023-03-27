import torch
import os

from model import BinaryCNNClassifier, SimpleNet
from utils import get_device
from dataset import ClassificationDataset
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import transforms
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train_valid_loop(
    train_dir: str = "data/train/1",
    output_dir: str = "output/train",
    run_name: str = "debug",
    slice_depth: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    train_dataset_size: int = None,
    valid_dataset_size: int = None,
    batch_size: int = 16,
    lr: float = 0.001,
    epochs: int = 2,
    num_workers: int = 16,
) -> nn.Module:
    device = get_device()

    # Load the model, try to fit on GPU
    model = SimpleNet(
        slice_depth=slice_depth,
    )
    model = model.to(device)

    # Create directory based on run_name with uuid for uniqueness
    run_name += f"_{np.random.randint(1000):03d}"
    output_dir = os.path.join(output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Writer for Tensorboard
    writer = SummaryWriter(output_dir)

    # Training dataset
    train_dataset = ClassificationDataset(
        # Directory containing the dataset
        train_dir,
        # Expected slices per fragment
        slice_depth=slice_depth,
        # Size of an individual patch
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
        # Image resize ratio
        resize_ratio=resize_ratio,
        # Training vs Testing mode
        train=True,
    )
    total_dataset_size = len(train_dataset)

    # Split indices into train and validation
    start_idx_train = 0
    end_idx_train = int(0.7 * total_dataset_size)
    start_idx_valid = int(0.8 * total_dataset_size)
    end_idx_valid = total_dataset_size
    train_idx = [i for i in range(start_idx_train, end_idx_train)]
    valid_idx = [i for i in range(start_idx_valid, end_idx_valid)]
    log.debug(f"Raw train dataset size: {len(train_idx)}")
    log.debug(f"Raw eval dataset size: {len(valid_idx)}")

    # Reduce dataset size based on max values
    if train_dataset_size is not None:
        train_idx = train_idx[:train_dataset_size]
        log.debug(f"Reduced train dataset size: {len(train_idx)}")
    if valid_dataset_size is not None:
        valid_idx = valid_idx[:valid_dataset_size]
        log.debug(f"Reduced eval dataset size: {len(valid_idx)}")

    # Sampler for Train and Validation
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # Shuffle does NOT work
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        # This will make it go faster if it is loaded into a GPU
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # Shuffle does NOT work
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_workers,
        # This will make it go faster if it is loaded into a GPU
        pin_memory=True,
    )

    # Create optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Simple optimizer and loss
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # loss_fn = nn.CrossEntropyLoss()

    # Train the model
    best_valid_loss = 0
    for epoch in range(epochs):
        log.info(f"Epoch {epoch + 1} of {epochs}")

        log.info(f"Training...")
        train_loss = 0
        for patch, label in tqdm(train_dataloader):
            optimizer.zero_grad()
            patch = patch.to(device)
            label = label.to(device).unsqueeze(1).to(torch.float32)
            pred = model(patch)
            # log.debug(f"Labels {label}")
            # log.debug(f"Predic {pred}")
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()  # Accumulate the training loss

        # Calculate the average training loss
        train_loss /= len(train_dataloader)
        # Log the average training loss
        writer.add_scalar(f'{loss_fn.__class__.__name__}/train', train_loss, epoch)

        
        log.info(f"Validation...")
        valid_loss = 0
        for patch, label in tqdm(valid_dataloader):
            patch = patch.to(device)
            label = label.to(device).unsqueeze(1).to(torch.float32)
            with torch.no_grad():
                pred = model(patch)
                loss = loss_fn(pred, label)
                valid_loss += loss.item()

        # Calculate the average validation loss
        valid_loss /= len(valid_dataloader)
        # Log the average validation loss
        writer.add_scalar('Loss/valid', valid_loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), f"{output_dir}/model.pth")

    writer.close()  # Close the SummaryWriter
    return model


def evaluate(
    model: nn.Module,
    data_dir="data/test/a",
    output_dir: str = "output/eval",
    slice_depth: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    threshold: float = 0.5,
):
    device = get_device()
    model = model.to(device)

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Evaluation dataset
    eval_dataset = ClassificationDataset(
        # Directory containing the dataset
        data_dir,
        # Expected slices per fragment
        slice_depth=slice_depth,
        # Size of an individual patch
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
        # Image resize ratio
        resize_ratio=resize_ratio,
        # Training vs Testing mode
        train=False,
    )
    # import pdb; pdb.set_trace()
    eval_dataset.mask.shape
    eval_dataset.mask_indices

    # Make a blank prediction image
    pred_image = np.zeros(eval_dataset.mask.shape[1:], dtype=np.uint8)

    # DataLoaders
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        # Shuffle does NOT work
        shuffle=False,
        sampler=SequentialSampler(eval_dataset),
        num_workers=16,
        # This will make it go faster if it is loaded into a GPU
        pin_memory=True,
    )

    for i, patch in enumerate(tqdm(eval_dataloader)):
        patch = patch.to(device)
        pixel_index = eval_dataset.mask_indices[i]
        with torch.no_grad():
            pred = model(patch)
            pred = torch.sigmoid(pred)

        pred_image[pixel_index[0], pixel_index[1]] = pred

        # if pred > threshold:
        #     pred_image[pixel_index[0], pixel_index[1]] = 1

    # Save the prediction image
    _img = Image.fromarray(pred_image * 255)
    _img.save(f"{output_dir}/pred_image.png")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"\tImage shape: {pred_image.shape}")
        log.debug(f"\tImage type: {pred_image.dtype}")
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.title(f'Prediction for {output_dir}')
        # Remove the channel dimension for grayscale
        plt.imshow(pred_image, cmap='gray', vmin=0, vmax=1)
        plt.subplot(122)
        plt.title('Histogram of Pixel Values')
        plt.hist(pred_image.flatten(), bins=256, range=(0, 1))
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

    # valid_loss /= len(valid_dataloader)  # Calculate the average validation loss
    # writer.add_scalar('Loss/valid', valid_loss, epoch)  # Log the average validation loss


if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    slice_depth = 65
    patch_size_x = 64
    patch_size_y = 16
    resize_ratio = 0.10

    trained_model = train_valid_loop(
        train_dir="data/train/1",
        slice_depth=slice_depth,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
        resize_ratio=resize_ratio,
        # train_dataset_size=1000000,
        # valid_dataset_size=10000,
        batch_size=128,
        lr=0.001,
        epochs=10,
        num_workers=16,
    )
    evaluate(
        model=trained_model,
        data_dir="data/test/a",
        slice_depth=slice_depth,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
        resize_ratio=resize_ratio,
    )
