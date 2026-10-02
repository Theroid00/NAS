"""Offline, stratified tabular splits with preprocessing fitted on training rows only."""
from functools import lru_cache
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_NAME = "breast_cancer_wisconsin"
INPUT_FEATURES = 30
NUM_CLASSES = 2


@lru_cache(maxsize=8)
def _split(seed):
    raw = load_breast_cancer()
    indices = np.arange(len(raw.target))
    train, held = train_test_split(indices, test_size=0.4, stratify=raw.target, random_state=seed)
    val, test = train_test_split(held, test_size=0.5, stratify=raw.target[held], random_state=seed)
    scaler = StandardScaler().fit(raw.data[train])
    datasets = tuple(TensorDataset(torch.tensor(scaler.transform(raw.data[idx]), dtype=torch.float32),
                                  torch.tensor(raw.target[idx], dtype=torch.long)) for idx in (train, val, test))
    metadata = {"name": DATASET_NAME, "input_features": INPUT_FEATURES, "num_classes": NUM_CLASSES,
                "split_seed": seed, "train_samples": len(train), "validation_samples": len(val),
                "test_samples": len(test), "split_indices": {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()},
                "preprocessing": "StandardScaler fitted exclusively on training rows",
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "data_sha256": hashlib.sha256(raw.data.tobytes() + raw.target.tobytes()).hexdigest()}
    return datasets, metadata


def dataset_metadata(seed=42):
    return _split(seed)[1]


def get_full_loaders(batch_size=32, seed=42, training_seed=42):
    datasets, _ = _split(seed)
    generator = torch.Generator().manual_seed(training_seed)
    return tuple(DataLoader(ds, batch_size=batch_size, shuffle=i == 0, num_workers=0,
                            generator=generator if i == 0 else None) for i, ds in enumerate(datasets))


def get_proxy_loaders(proxy_size=0, batch_size=32, seed=42, training_seed=42):
    """Zero selects all training rows. Positive sizes select a fixed stratified subset."""
    datasets, _ = _split(seed)
    train, val, _ = datasets
    if type(proxy_size) is not int or not 0 <= proxy_size <= len(train):
        raise ValueError(f"Proxy size must be 0 (all training rows) or 1..{len(train)}")
    if 0 < proxy_size < len(train):
        try:
            selected, _ = train_test_split(np.arange(len(train)), train_size=proxy_size,
                                          stratify=train.tensors[1].numpy(), random_state=seed)
        except ValueError as error:
            raise ValueError("Proxy subset must be large enough for a stratified binary split") from error
        train = TensorDataset(*(tensor[selected] for tensor in train.tensors))
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0,
                       generator=torch.Generator().manual_seed(training_seed)),
            DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0))
