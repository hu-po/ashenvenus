import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from tqdm import tqdm

from dataset import PatchDataset
from model import PreTrainNet, SimpleNet, SimpleNetNorm
from utils import get_device

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train_valid_loop(
    train_dir: str = "data/train/",
    model: str = "simplenet",
    freeze_backbone: bool = False,
    optimizer: str = "adam",
    curriculum: str = "1",
    max_samples_per_dataset: int = 100,
    output_dir: str = "output/train",
    image_augs: bool = False,
    slice_depth: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    batch_size: int = 16,
    lr: float = 0.001,
    lr_scheduling_gamma: float = None,
    num_epochs: int = 2,
    num_workers: int = 16,
):
    device = get_device()

    # Load the model, try to fit on GPU
    if model == "simplenet":
        model = SimpleNet(
            slice_depth=slice_depth,
        )
    elif model == 'convnext_tiny':
        model = PreTrainNet(
            slice_depth=slice_depth,
            pretrained_model='convnext_tiny',
            freeze_backbone=freeze_backbone,
        )
    elif model == 'vit_b_32':
        model = PreTrainNet(
            slice_depth=slice_depth,
            pretrained_model='vit_b_32',
            freeze_backbone=freeze_backbone,
        )
    elif model == 'swin_t':
        model = PreTrainNet(
            slice_depth=slice_depth,
            pretrained_model='swin_t',
            freeze_backbone=freeze_backbone,
        )
    elif model == 'resnext50_32x4d':
        model = PreTrainNet(
            slice_depth=slice_depth,
            pretrained_model='resnext50_32x4d',
            freeze_backbone=freeze_backbone,
        )
    elif model == "simplenet_norm":
        model = SimpleNetNorm(
            slice_depth=slice_depth,
        )
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    # Create optimizers
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if lr_scheduling_gamma is not None:
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_scheduling_gamma)

    # Writer for Tensorboard
    writer = SummaryWriter(output_dir)

    # Train the model
    best_loss = 0
    step = 0
    for epoch in range(num_epochs):
        log.info(f"Epoch: {epoch}")

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
                # Training vs Testing mode
                train=True,
            )
            total_dataset_size = len(train_dataset)
            log.debug(f"Raw train dataset size: {total_dataset_size}")

            # Add augmentations
            img_transform_list = [
                transforms.Normalize(train_dataset.mean, train_dataset.std)
            ]
            if image_augs:
                img_transform_list += [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            img_transform = transforms.Compose(img_transform_list)

            # DataLoaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=RandomSampler(
                    train_dataset, num_samples=max_samples_per_dataset),
                num_workers=num_workers,
                # This will make it go faster if it is loaded into a GPU
                pin_memory=True,
            )

            log.info(f"Training...")
            train_loss = 0
            for patch, label in tqdm(train_dataloader):
                # for patch, label in train_dataloader:
                optimizer.zero_grad()
                # writer.add_histogram('patch_input', patch, step)
                # writer.add_histogram('label_input', label, step)
                patch = patch.to(device)
                patch = img_transform(patch)
                pred = model(patch)
                label = label.to(device)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                step += 1
                train_loss += loss.item()
            train_loss /= max_samples_per_dataset
            writer.add_scalar(
                f'{loss_fn.__class__.__name__}/{current_dataset_id}/train', train_loss, step)

            if train_loss < best_loss:
                best_loss = train_loss
                # torch.save(model.state_dict(), f"{output_dir}/model.pth")

        if lr_scheduling_gamma is not None:
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print("Epoch %d: SGD lr %.4f -> %.4f" %
                  (epoch, before_lr, after_lr))

    evaluate(
        model,
        data_dir="data/test/a",
        output_dir=output_dir,
        slice_depth=slice_depth,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
        resize_ratio=resize_ratio,
    )

    evaluate(
        model,
        data_dir="data/test/b",
        output_dir=output_dir,
        slice_depth=slice_depth,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
        resize_ratio=resize_ratio,
    )

    writer.close()  # Close the SummaryWriter
    return best_loss


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
    model.eval()

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Evaluation dataset
    eval_dataset = PatchDataset(
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

    # Make a blank prediction image
    pred_image = np.zeros(eval_dataset.resized_size, dtype=np.uint8).T
    print(f"Prediction image shape: {pred_image.shape}")

    # DataLoaders
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        sampler=SequentialSampler(eval_dataset),
        num_workers=16,
        # This will make it go faster if it is loaded into a GPU
        pin_memory=True,
    )
    img_transform = transforms.Compose([
        transforms.Normalize(eval_dataset.mean, eval_dataset.std)
    ])

    for i, patch in enumerate(tqdm(eval_dataloader)):
        patch = patch.to(device)
        patch = img_transform(patch)
        pixel_index = eval_dataset.mask_indices[i]
        with torch.no_grad():
            pred = model(patch)
            pred = torch.sigmoid(pred)

        pred_image[pixel_index[0], pixel_index[1]] = pred

        if pred > threshold:
            pred_image[pixel_index[0], pixel_index[1]] = 1

    # Save the prediction image
    _img = Image.fromarray(pred_image * 255)
    _img.save(f"{output_dir}/pred_image.png")

    # if log.isEnabledFor(logging.DEBUG):
    #     log.debug(f"\tImage shape: {pred_image.shape}")
    #     log.debug(f"\tImage type: {pred_image.dtype}")
    #     plt.figure(figsize=(12, 5))
    #     plt.subplot(121)
    #     plt.title(f'Prediction for {output_dir}')
    #     # Remove the channel dimension for grayscale
    #     plt.imshow(pred_image, cmap='gray', vmin=0, vmax=1)
    #     plt.subplot(122)
    #     plt.title('Histogram of Pixel Values')
    #     plt.hist(pred_image.flatten(), bins=256, range=(0, 1))
    #     plt.xlabel('Pixel Value')
    #     plt.ylabel('Frequency')
    #     plt.show()


if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    slice_depth = 2
    patch_size_x = 128
    patch_size_y = 32
    resize_ratio = 0.25

    trained_model = train_valid_loop(
        train_dir="data/train/",
        slice_depth=slice_depth,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
        resize_ratio=resize_ratio,
        curriculum='123',
        max_samples_per_dataset=100,
        batch_size=128,
        lr=0.001,
        num_epochs=2,
        num_workers=16,
    )
