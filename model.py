import logging

import torch
import torch.nn as nn

from utils import get_device

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, stride, padding)
        self.bn1 = nn.LazyBatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size, 1, padding)
        self.bn2 = nn.LazyBatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, padding=0),
                nn.LazyBatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.gelu(out)
        return out


class BinaryCNNClassifier(nn.Module):
    def __init__(self):
        super(BinaryCNNClassifier, self).__init__()
        self.layer1 = ResidualBlock(65, 128, 3, 2, 1)
        self.layer2 = ResidualBlock(128, 256, 3, 2, 1)
        self.layer3 = ResidualBlock(256, 512, 3, 2, 1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc = nn.LazyLinear(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Model Body
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.max_pool(out)
        # Flatten
        out = out.view(out.size(0), -1)
        # Model Head
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    from torchviz import make_dot
    from GPUtil import showUtilization

    log.setLevel(logging.DEBUG)
    device = get_device()

    INPUT_SHAPE = (32, 65, 1028, 512)
    OUTPUT_SHAPE = (32, 1)

    # Attributes to print out about GPU
    attrList = [[
        {'attr':'id','name':'ID'},
        {'attr':'load','name':'GPU','suffix':'%','transform': lambda x: x*100,'precision':0},
        {'attr':'memoryUtil','name':'MEM','suffix':'%','transform': lambda x: x*100,'precision':0},
        {'attr':'memoryTotal','name':'Memory total','suffix':'MB','precision':0},
        {'attr':'memoryUsed','name':'Memory used','suffix':'MB','precision':0},
        {'attr':'memoryFree','name':'Memory free','suffix':'MB','precision':0},
    ]]

    log.debug("\n\n GPU usage after data is loaded")
    showUtilization(attrList=attrList)

    # Test the model fits into the GPU
    model = BinaryCNNClassifier()
    model.to(device)

    log.debug("\n\n GPU usage after model is loaded")
    showUtilization(attrList=attrList)

    # Fake data to test the model
    x = torch.randn(INPUT_SHAPE, dtype=torch.float32).to(device)
    
    log.debug("\n\n GPU usage after data is loaded")
    showUtilization(attrList=attrList)

    y = model(x)
    
    # Print out the model information
    log.debug(model)

    def get_bytes_per_parameter(model: torch.nn.Module):
        for parameter in model.parameters():
            if parameter.requires_grad:
                dtype = parameter.dtype
                break
        return torch.tensor(1, dtype=dtype).element_size()

    # Print out the size of the model
    _model_size: int = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    # Get bytes per parameter
    _bytes_per_parameter = get_bytes_per_parameter(model)
    # Model size in GB
    _model_size_gb = _model_size * _bytes_per_parameter / 1024 / 1024 / 1024
    log.debug(f"Model Size: {_model_size_gb} GB")

    # Print out the model graph
    make_dot(y, params=dict(model.named_parameters())).render("model_graph", format="png")
