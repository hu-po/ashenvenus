import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    vit_b_32, ViT_B_32_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
    swin_t, Swin_T_Weights,
)

from utils import get_device

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class PreTrainNet(nn.Module):
    def __init__(
        self,
        slice_depth: int = 65,
        pretrained_model: str = 'convnext_tiny',
        freeze_backbone: bool = False,
        # pretrained_weights_filepath = '/kaggle/input/convnextimagenet/convnext_tiny-983f1562.pth',
        # pretrained_weights_filepath='/home/tren/dev/ashenvenus/notebooks/convnext_tiny-983f1562.pth',

    ):
        super().__init__()
        self.conv = nn.Conv2d(slice_depth, 3, 3)
        # Load pretrained model
        if pretrained_model == 'convnext_tiny':
            _weights = ConvNeXt_Tiny_Weights.DEFAULT
            self.pre_trained_model = convnext_tiny(weights=_weights)
        elif pretrained_model == 'vit_b_32':
            _weights = ViT_B_32_Weights.DEFAULT
            self.pre_trained_model = vit_b_32(weights=_weights)
        elif pretrained_model == 'swin_t':
            _weights = Swin_T_Weights.DEFAULT
            self.pre_trained_model = swin_t(weights=_weights)
        elif pretrained_model == 'resnext50_32x4d':
            _weights = ResNeXt50_32X4D_Weights.DEFAULT
            self.pre_trained_model = resnext50_32x4d(weights=_weights)
        # self.model.load_state_dict(pretrained_weights_filepath)
        # Put model in training mode
        if freeze_backbone:
            self.pre_trained_model.eval()
        else:
            self.pre_trained_model.train()
        # Binary classification head on top
        self.fc = nn.LazyLinear(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pre_trained_model(x)
        x = self.fc(x)
        return x


class SimpleNet(nn.Module):
    def __init__(
        self,
        slice_depth: int = 65,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(slice_depth, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleNetNorm(nn.Module):
    def __init__(
        self,
        slice_depth: int = 65,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(slice_depth, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.LazyLinear(120)
        self.ln1 = nn.LayerNorm(120)
        self.fc2 = nn.LazyLinear(84)
        self.ln2 = nn.LayerNorm(84)
        self.fc3 = nn.LazyLinear(1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    from torchviz import make_dot
    from GPUtil import showUtilization

    log.setLevel(logging.DEBUG)
    device = get_device()

    INPUT_SHAPE = (32, 65, 1028, 512)
    OUTPUT_SHAPE = (32, 1)

    # Attributes to print out about GPU
    attrList = [[
        {'attr': 'id', 'name': 'ID'},
        {'attr': 'load', 'name': 'GPU', 'suffix': '%',
            'transform': lambda x: x*100, 'precision': 0},
        {'attr': 'memoryUtil', 'name': 'MEM', 'suffix': '%',
            'transform': lambda x: x*100, 'precision': 0},
        {'attr': 'memoryTotal', 'name': 'Memory total',
            'suffix': 'MB', 'precision': 0},
        {'attr': 'memoryUsed', 'name': 'Memory used',
            'suffix': 'MB', 'precision': 0},
        {'attr': 'memoryFree', 'name': 'Memory free',
            'suffix': 'MB', 'precision': 0},
    ]]

    log.debug("\n\n GPU usage after data is loaded")
    showUtilization(attrList=attrList)

    # Test the model fits into the GPU
    # model = BinaryCNNClassifier()
    model = SimpleNet()
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
    make_dot(y, params=dict(model.named_parameters())
             ).render("model_graph", format="png")
