import torch

from model import UNet3D
from utils import get_device
from dataset import FragmentPatchesDataset
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train_valid_eval_loop(
    train_dir: str = "data/train/1",
    eval_dir: str = "data/test/a",
    output_dir: str = "output",
    train_dataset_size: int = 100,
    eval_dataset_size: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    epochs: int = 2,
) -> float:
    # Get the device
    device = get_device()

    # Load the model, try to fit on GPU
    # model = UNet3D()
    # model = model.to(device)

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=32, out_channels=1, init_features=32, pretrained=False)
    model.to(device)

    # Training dataset
    train_dataset = FragmentPatchesDataset(train_dir, viz=False, train=True)

    # Sampler for Train and Validation
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=train_dataset_size)
    valid_sampler = RandomSampler(
        train_dataset, replacement=False, num_samples=eval_dataset_size)

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # Shuffle does NOT work
        shuffle=False,
        sampler=train_sampler,
        num_workers=16,
        # This will make it go faster if it is loaded into a GPU
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # Shuffle does NOT work
        shuffle=False,
        sampler=valid_sampler,
        num_workers=16,
        # This will make it go faster if it is loaded into a GPU
        pin_memory=True,
    )

    # Create optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # UNet loss over 3 dimmensions can be MSE?
    loss_fn = torch.nn.MSELoss()

    # Train the model
    best_eval_score = 0
    for epoch in range(epochs):
        log.info(f"Epoch {epoch + 1} of {epochs}")

        # Train
        for patch, mask, label in tqdm(train_dataloader):
            patch = patch.to(device)
            label = label.to(device).unsqueeze(1).to(torch.float32)
            pred = model(patch)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Test
        eval_score = 0
        for patch, mask, label in tqdm(valid_dataloader):
            patch = patch.to(device)
            label = label.to(device).unsqueeze(1).to(torch.float32)
            with torch.no_grad():
                pred = model(patch)
                loss = loss_fn(pred, label)
                eval_score += loss.item()

        if eval_score > best_eval_score:
            best_eval_score = eval_score
            torch.save(model.state_dict(), f"{output_dir}/model.pth")

    return best_eval_score


def evaluate(
    model,
    loss_fn,
    device,
    data_dir="data/test/a",
    batch_size: int = 32,
):

    eval_dataset = FragmentPatchesDataset(data_dir, viz=False, train=False)
    # Sampler that goes through the entire dataset once with appropraite stride
    eval_sampler = RandomSampler(eval_dataset, replacement=False)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        # Shuffle does NOT work
        shuffle=False,
        sampler=eval_sampler,
        num_workers=16,
        # This will make it go faster if it is loaded into a GPU
        pin_memory=True,
    )

    for patch, mask in tqdm(eval_dataloader):
        patch = patch.to(device)
        with torch.no_grad():
            pred = model(patch)

    # TODO: RLE dump to file
            


if __name__ == '__main__':
    train_eval_loop()
