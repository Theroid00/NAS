"""
data/cifar.py
=============
CIFAR-10 data loading with:
  - get_proxy_loaders()  — small subset for fast fitness evaluation
  - get_full_loaders()   — full training/validation/test splits
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as T
import random

from training.config import (
    CIFAR10_MEAN, CIFAR10_STD,
    PROXY_SUBSET_SIZE, BATCH_SIZE, NUM_WORKERS, RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Shared transforms
# ---------------------------------------------------------------------------

TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

TEST_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

_DATA_ROOT = "./data/cifar10_data"  # download cache


def _get_raw_datasets():
    train_set = torchvision.datasets.CIFAR10(
        root=_DATA_ROOT, train=True, download=True, transform=TRAIN_TRANSFORM
    )
    test_set = torchvision.datasets.CIFAR10(
        root=_DATA_ROOT, train=False, download=True, transform=TEST_TRANSFORM
    )
    return train_set, test_set


# ---------------------------------------------------------------------------
# Proxy loaders (small subset for GA fitness evaluation)
# ---------------------------------------------------------------------------

def get_proxy_loaders(
    proxy_size: int = PROXY_SUBSET_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
):
    """
    Returns (proxy_train_loader, proxy_val_loader).

    Proxy train: proxy_size samples, proxy val: 2000 held-out samples.
    These are used exclusively during the GA search for fast fitness scoring.
    """
    train_set_raw, test_set_raw = _get_raw_datasets()
    
    # We need a pristine validation set that does not undergo random crops/flips.
    # So we load the train split again with TEST_TRANSFORM.
    val_set_raw = torchvision.datasets.CIFAR10(
        root=_DATA_ROOT, train=True, download=True, transform=TEST_TRANSFORM
    )

    generator = torch.Generator().manual_seed(seed)
    n_val = 2_000
    n_train = proxy_size
    n_rest = len(train_set_raw) - n_train - n_val

    # random_split just creates Subsets. We can extract indices.
    train_sub, val_sub, _ = random_split(
        train_set_raw, [n_train, n_val, n_rest], generator=generator
    )
    
    # Map the validation indices to the pristine val_set_raw
    proxy_train = train_sub
    proxy_val = Subset(val_set_raw, val_sub.indices)

    proxy_train_loader = DataLoader(
        proxy_train, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    proxy_val_loader = DataLoader(
        proxy_val, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return proxy_train_loader, proxy_val_loader


# ---------------------------------------------------------------------------
# Full loaders (for training the best architecture to convergence)
# ---------------------------------------------------------------------------

def get_full_loaders(
    val_split: float = 0.1,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
):
    """
    Returns (train_loader, val_loader, test_loader) for full training.

    val_split: fraction of training data held out for validation.
    """
    train_set_raw, test_set = _get_raw_datasets()
    
    val_set_raw = torchvision.datasets.CIFAR10(
        root=_DATA_ROOT, train=True, download=True, transform=TEST_TRANSFORM
    )

    n_val = int(len(train_set_raw) * val_split)
    n_train = len(train_set_raw) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(train_set_raw, [n_train, n_val], generator=generator)
    
    # Apply clean transforms to validation subset
    clean_val_sub = Subset(val_set_raw, val_sub.indices)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        clean_val_sub, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
