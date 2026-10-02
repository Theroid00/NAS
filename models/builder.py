"""
models/builder.py
=================
Dynamically builds a PyTorch CNN from a decoded chromosome architecture dict.

Input shape: (batch, 3, 32, 32)  [CIFAR-10]
Output shape: (batch, num_classes)
"""

import torch
import torch.nn as nn
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _get_activation(name: str) -> nn.Module:
    activations = {
        "relu":       nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "elu":        nn.ELU(inplace=True),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


def _get_pool(name: str, stride: int = 2) -> nn.Module:
    if name == "max":
        return nn.MaxPool2d(kernel_size=2, stride=stride)
    elif name == "avg":
        return nn.AvgPool2d(kernel_size=2, stride=stride)
    elif name == "mixed":
        # Mixed: average pooling (a common alternative explored in NAS)
        return nn.AvgPool2d(kernel_size=2, stride=stride)
    else:
        raise ValueError(f"Unknown pooling type: {name}")


# ---------------------------------------------------------------------------
# Dynamic CNN builder
# ---------------------------------------------------------------------------

FILTER_KEYS = ["filters_1", "filters_2", "filters_3", "filters_4", "filters_5"]


def estimate_parameters(arch, num_classes=10):
    """Exact count for this builder, before allocating candidate weights."""
    in_ch, total = 3, 0
    kernel = arch["kernel_size"]
    for i in range(arch["num_blocks"]):
        out_ch = arch.get(FILTER_KEYS[i], arch["filters_3"])
        total += in_ch * out_ch * kernel ** 2 + out_ch
        if arch["batch_norm"]:
            total += 2 * out_ch
        if arch.get("use_residual", False):
            total += out_ch ** 2 * kernel ** 2 + out_ch
            if arch["batch_norm"]:
                total += 2 * out_ch
            if in_ch != out_ch:
                total += in_ch * out_ch
                if arch["batch_norm"]:
                    total += 2 * out_ch
        in_ch = out_ch
    spatial = 32 // (2 ** arch["num_blocks"])
    hidden = arch["fc_hidden"]
    return total + in_ch * spatial ** 2 * hidden + hidden + hidden * num_classes + num_classes

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation_name, use_bn, pool_name):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(_get_activation(activation_name))
        layers.append(_get_pool(pool_name))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation_name, use_bn, pool_name):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(_get_activation(activation_name))
        
        layers.append(
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
            
        self.main_path = nn.Sequential(*layers)
        
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
            
        self.final_activation = _get_activation(activation_name)
        self.pool = _get_pool(pool_name)
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main_path(x)
        out += residual
        out = self.final_activation(out)
        return self.pool(out)


def build_model(arch: Dict[str, Any], num_classes: int = 10) -> nn.Module:
    """
    Build a CNN from a decoded chromosome architecture dict.

    Args:
        arch:        Decoded architecture dict from chromosome.decode()
        num_classes: Number of output classes (10 for CIFAR-10)

    Returns:
        nn.Sequential model ready for training
    """
    layers = []
    in_ch = 3
    kernel = arch["kernel_size"]
    activation_name = arch["activation"]
    use_bn = arch["batch_norm"]
    pool_name = arch["pooling"]
    num_blocks = arch["num_blocks"]
    use_res = arch.get("use_residual", False)

    for i in range(num_blocks):
        # Use available filter key; repeat last if chromosome only has 3
        fkey = FILTER_KEYS[min(i, len(FILTER_KEYS) - 1)]
        out_ch = arch.get(fkey, arch.get("filters_3", 128))

        if use_res:
            layers.append(ResidualBlock(in_ch, out_ch, kernel, activation_name, use_bn, pool_name))
        else:
            layers.append(ConvBlock(in_ch, out_ch, kernel, activation_name, use_bn, pool_name))

        in_ch = out_ch

    # Flatten + classifier head
    layers.append(nn.Flatten())
    layers.append(nn.Dropout(p=arch["dropout"]))
    # Use LazyLinear so we don't need to compute spatial dims manually
    layers.append(nn.LazyLinear(arch["fc_hidden"]))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(arch["fc_hidden"], num_classes))

    return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_model(arch: Dict[str, Any], num_classes: int = 10) -> str:
    """
    Build a model, run a dummy forward pass, and return a summary string.
    Useful for smoke-testing chromosomes.
    """
    model = build_model(arch, num_classes)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    params = count_parameters(model)
    return (
        f"in={tuple(x.shape)} → out={tuple(out.shape)} | "
        f"params={params:,}"
    )
