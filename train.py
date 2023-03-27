import torch

from model import BinaryCNNClassifier
from utils import get_device
from dataset import ClassificationDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train_valid_loop(
    train_dir: str = "data/train/1",
    output_dir: str = "output",
    slice_depth: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    train_dataset_size: int = 100,
    eval_dataset_size: int = 64,
    batch_size: int = 16,
    lr: float = 0.001,
    epochs: int = 2,
    num_workers: int = 16,
) -> float:
    device = get_device()
    
    # Load the model, try to fit on GPU
    model = BinaryCNNClassifier(
        slice_depth=slice_depth,
    )
    model = model.to(device)

    # Writer for Tensorboard
    writer = SummaryWriter(output_dir, comment="test")

    # Training dataset
    train_dataset = ClassificationDataset(
        # Directory containing the dataset
        train_dir,
        # Expected slices per fragment
        slice_depth = slice_depth,
        # Size of an individual patch
        patch_size_x = patch_size_x,
        patch_size_y = patch_size_y,
        # Image resize ratio
        resize_ratio = resize_ratio,
        # Training vs Testing mode
        train = True,
    )
    total_dataset_size = len(train_dataset)

    # Split indices into train and validation
    start_idx_train = 0
    end_idx_train =  int(0.7 * total_dataset_size)
    start_idx_valid = int(0.8 * total_dataset_size)
    end_idx_valid = total_dataset_size
    train_idx = [i for i in range(start_idx_train, end_idx_train)]
    valid_idx = [i for i in range(start_idx_valid, end_idx_valid)]
    log.debug(f"Raw train dataset size: {len(train_idx)}")
    log.debug(f"Raw eval dataset size: {len(valid_idx)}")

    # Reduce dataset size based on max values
    train_idx = train_idx[:train_dataset_size]
    valid_idx = valid_idx[:eval_dataset_size]
    log.debug(f"Reduced train dataset size: {len(train_idx)}")
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
    loss_fn = torch.nn.BCELoss()

    # Train the model
    best_valid_loss = 0
    for epoch in range(epochs):
        log.info(f"Epoch {epoch + 1} of {epochs}")

        # Train
        train_loss = 0
        for patch, label in tqdm(train_dataloader):
            patch = patch.to(device)
            label = label.to(device).unsqueeze(1).to(torch.float32)
            pred = model(patch)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()  # Accumulate the training loss

        train_loss /= len(train_dataloader)  # Calculate the average training loss
        writer.add_scalar('Loss/train', train_loss, epoch)  # Log the average training loss

        # Test
        valid_loss = 0
        for patch, label in tqdm(valid_dataloader):
            patch = patch.to(device)
            label = label.to(device).unsqueeze(1).to(torch.float32)
            with torch.no_grad():
                pred = model(patch)
                loss = loss_fn(pred, label)
                valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)  # Calculate the average validation loss
        writer.add_scalar('Loss/valid', valid_loss, epoch)  # Log the average validation loss

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{output_dir}/model.pth")

    writer.close()  # Close the SummaryWriter
    return best_valid_loss


def evaluate(
    model,
    loss_fn,
    device,
    data_dir="data/test/a",
    batch_size: int = 32,
):
    pass


if __name__ == '__main__':

    train_valid_loop(
        train_dir="data/train/1",
        slice_depth = 3,
        patch_size_x = 512,
        patch_size_y = 128,
        resize_ratio = 1.0,
        train_dataset_size = 1000,
        eval_dataset_size = 200,
        batch_size = 16,
        lr = 0.01,
        epochs = 10,
        num_workers = 16,
    )
