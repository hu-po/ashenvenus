import torch

from model import My3DCNN
from utils import get_device
from dataset import FragmentPatchesDataset
import torch.utils.data as data
import tqdm

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train_eval_loop(
    data_dir: str = "data/train/1/",
    batch_size: int = 1,
    lr: float = 0.001,
    epochs: int = 2,
    **kwargs,
) -> float:
    # Get the device
    device = get_device()

    # Load the model
    model = My3DCNN(**kwargs).to(device)

    # Load the dataset
    dataset = FragmentPatchesDataset(data_dir, **kwargs)

    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(
        dataset, [train_size, test_size])

    # Create the dataloaders
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Create optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Binary classification loss
    loss_fn = torch.nn.BCELoss()

    # Train the model
    best_score = 0
    for epoch in range(epochs):
        log.info(f"Epoch {epoch + 1} of {epochs}")

        # Train

        # Test
        model.eval()
        for patch, label in tqdm(test_loader):
            pred = model(patch)
            loss = loss_fn(pred, label)
    

    return best_score


if __name__ == '__main__':
    train_eval_loop()
