"""
models/baselines/resnet.py
==========================
Hand-designed 3-block ResNet baseline for comparison against GA-NAS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with two 3×3 convolutions and a skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class BaselineResNet(nn.Module):
    """
    Simple 3-block ResNet for CIFAR-10.
    Architecture: init conv → block1(64) → block2(128) → block3(256)
                 → global avg pool → FC(10)
    ~2.8 M parameters.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock(64,  64,  stride=1)
        self.block2 = ResidualBlock(64,  128, stride=2)
        self.block3 = ResidualBlock(128, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_baseline_resnet(num_classes: int = 10) -> nn.Module:
    return BaselineResNet(num_classes)
