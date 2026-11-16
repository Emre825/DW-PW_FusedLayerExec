import pytest
import torch
import torch.nn as nn
import torchvision

from sw.capture import INPUT_SHAPE, fold_bn, layer_table, trace


@pytest.fixture(scope="session")
def traced():
    """Untrained MobileNetV2 with non-trivial BN parameters and statistics (no download).

    The statistics are measured from one train-mode pass rather than drawn at random:
    random statistics kill the signal, and the output then no longer depends on the input.
    """
    torch.manual_seed(0)
    model = torchvision.models.mobilenet_v2(weights=None)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0.5, 1.5)
            nn.init.normal_(m.bias, 0.0, 0.5)
            m.momentum = 1.0  # running statistics := statistics of the next batch
    with torch.no_grad():
        model.train()(torch.randn(4, *INPUT_SHAPE[1:]))
    return trace(model)


@pytest.fixture(scope="session")
def resnet_layers():
    """Layer table of an untrained ResNet-18 (no download)."""
    return layer_table(fold_bn(trace(torchvision.models.resnet18(weights=None))))
