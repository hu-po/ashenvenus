import torch
import torch.nn as nn

from utils import get_device
from dataset import FragmentPatchesDataset


class SmolCNN(nn.Module):
    def __init__(
        self,
        kernel_size_3d: int = 3,
    ):
        super(SmolCNN, self).__init__()
        # 1D kernel in depth
        self.conv1 = nn.Conv3d(1, 8, (kernel_size_3d, 1, 1), 1)
        self.conv1 = nn.Conv3d(1, 8, kernel_size_3d, 1)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.LazyLinear(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.middle = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3


if __name__ == '__main__':

    device = get_device()

    # Test the model
    model = SmolCNN()
    model.to(device)

    # Test with some data
    data_dir = 'data/train/1/'
    dataset = FragmentPatchesDataset(data_dir)
    for i in range(3):
        patch, label = dataset[i]
        output = model(patch.unsqueeze(0).unsqueeze(0).to(device))
        print(output)
