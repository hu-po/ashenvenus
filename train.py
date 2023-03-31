import os
import time

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
from utils import clear_gpu_memory, get_device, rle


def train_loop(
    train_dir: str = "data/train/",
    eval_dir: str = "data/eval/",
    model: str = "simplenet",
    freeze_backbone: bool = False,
    pretrained_weights_filepath: str = None,
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
    num_workers: int = 1,
    write_logs: bool = False,
    run_eval_sweep: bool = False,
    run_eval_submit: bool = False,
    max_time_hours: float = 8,
):
    device = get_device()
    clear_gpu_memory()

    # Notebook will only run for this amount of time
    time_train_max_seconds = max_time_hours * 60 * 60
    time_start = time.time()
    time_elapsed = 0

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
            pretrained_weights_filepath=pretrained_weights_filepath,
        )
    elif model == 'vit_b_32':
        model = PreTrainNet(
            slice_depth=slice_depth,
            pretrained_model='vit_b_32',
            freeze_backbone=freeze_backbone,
            pretrained_weights_filepath=pretrained_weights_filepath,
        )
    elif model == 'swin_t':
        model = PreTrainNet(
            slice_depth=slice_depth,
            pretrained_model='swin_t',
            freeze_backbone=freeze_backbone,
            pretrained_weights_filepath=pretrained_weights_filepath,
        )
    elif model == 'resnext50_32x4d':
        model = PreTrainNet(
            slice_depth=slice_depth,
            pretrained_model='resnext50_32x4d',
            freeze_backbone=freeze_backbone,
            pretrained_weights_filepath=pretrained_weights_filepath,
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
        scheduler = lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_scheduling_gamma)

    # Writer for Tensorboard
    if write_logs:
        writer = SummaryWriter(output_dir)

    # Train the model
    best_loss = 0
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
                # Training vs Testing mode
                train=True,
            )
            total_dataset_size = len(train_dataset)
            print(f"Raw train dataset size: {total_dataset_size}")

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

            print(f"Training...")
            train_loss = 0
            for patch, label in tqdm(train_dataloader):
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

                # Check if we have exceeded the time limit
                time_elapsed = time.time() - time_start
                if time_elapsed > time_train_max_seconds:
                    print("Time limit exceeded, stopping batches")
                    break

            train_loss /= max_samples_per_dataset
            if write_logs:
                writer.add_scalar(
                    f'{loss_fn.__class__.__name__}/{current_dataset_id}/train', train_loss, step)

            if train_loss < best_loss:
                best_loss = train_loss
                # torch.save(model.state_dict(), f"{output_dir}/model.pth")

            # Check if we have exceeded the time limit
            time_elapsed = time.time() - time_start
            if time_elapsed > time_train_max_seconds:
                print("Time limit exceeded, stopping curriculum")
                break

        if lr_scheduling_gamma is not None:
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

    if write_logs:
        writer.close()  # Close the SummaryWriter

    del train_dataloader, train_dataset
    clear_gpu_memory()

    if run_eval_submit:
        # Create submission file
        submission_filepath = 'submission.csv'
        with open(submission_filepath, 'w') as f:
            # Write header
            f.write("Id,Predicted\n")

    # Baseline is to use image mask to create guess submission
    for subtest_name in os.listdir(eval_dir):

        # Name of sub-directory inside test dir
        subtest_filepath = os.path.join(eval_dir, subtest_name)

        if run_eval_sweep:
            evaluate_sweep(
                model,
                data_dir=subtest_filepath,
                output_dir=output_dir,
                subtest_name=subtest_name,
                slice_depth=slice_depth,
                patch_size_x=patch_size_x,
                patch_size_y=patch_size_y,
                resize_ratio=resize_ratio,
                num_workers=num_workers,
            )

        if run_eval_submit:
            evaluate_submit(
                model,
                data_dir=subtest_filepath,
                submission_filepath=submission_filepath,
                subtest_name=subtest_name,
                slice_depth=slice_depth,
                patch_size_x=patch_size_x,
                patch_size_y=patch_size_y,
                resize_ratio=resize_ratio,
                num_workers=num_workers,
            )
    return best_loss


def evaluate_sweep(
    model: nn.Module,
    data_dir="data/test/a",
    subtest_name: str = "a",
    output_dir: str = "output/eval",
    slice_depth: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    num_workers: int = 1,
    threshold: float = 0.5,
):
    device = get_device()
    model = model.to(device)
    model.eval()

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
    print(f"Prediction image {subtest_name} shape: {pred_image.shape}")

    # DataLoaders
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(eval_dataset),
        num_workers=num_workers,
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
    _img.save(f"{output_dir}/pred_image_{subtest_name}.png")


def evaluate_submit(
    model: nn.Module,
    data_dir="data/test/a",
    subtest_name: str = "a",
    submission_filepath: str = "submission.csv",
    slice_depth: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    num_workers: int = 1,
    threshold: float = 0.5,
):
    device = get_device()
    model = model.to(device)
    model.eval()

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
        batch_size=batch_size,
        sampler=SequentialSampler(eval_dataset),
        num_workers=num_workers,
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

        if pred > threshold:
            pred_image[pixel_index[0], pixel_index[1]] = 1

    # Resize pred_image to original size
    _img = Image.fromarray(pred_image)
    _img = _img.resize((
        eval_dataset.original_size[0],
        eval_dataset.original_size[1],
    ))
    pred_image = np.array(_img)

    starts_ix, lengths = rle(pred_image)
    inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
    with open(submission_filepath, 'a') as f:
        f.write(f"{subtest_name},{inklabels_rle}\n")
