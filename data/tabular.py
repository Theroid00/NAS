"""Cached tabular splits and fixed subsets; preprocessing never sees held-out rows."""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.specs import DEFAULT_DATASET, INPUT_FEATURES, NUM_CLASSES, dataset_spec, search_sizes

# Library defaults retain compatibility with the original tabular API.
# Experiment CLIs select Covertype explicitly by default.
DATASET_NAME = DEFAULT_DATASET
DATA_HOME = Path(__file__).resolve().parent / "tabular_cache"


def prepare_dataset(dataset="covertype"):
    spec = dataset_spec(dataset)
    if dataset == "covertype":
        raw = fetch_covtype(data_home=str(DATA_HOME), download_if_missing=True)
    else:
        raw = load_breast_cancer()
    return {"dataset": dataset, "rows": len(raw.target), "input_features": spec["input_features"],
            "num_classes": spec["num_classes"], "cache_dir": str(DATA_HOME)}


@lru_cache(maxsize=2)
def _raw(dataset):
    dataset_spec(dataset)
    if dataset == "covertype":
        try:
            raw = fetch_covtype(data_home=str(DATA_HOME), download_if_missing=False)
        except PermissionError:
            raise
        except OSError as error:
            raise OSError(f"Cannot load cached Covertype: {error}. Run python prepare_dataset.py --dataset covertype first") from error
        return raw.data, raw.target.astype(np.int64) - 1
    raw = load_breast_cancer()
    return raw.data, raw.target


@lru_cache(maxsize=3)
def _split(seed, dataset=DATASET_NAME):
    spec = dataset_spec(dataset)
    x, y = _raw(dataset)
    if x.shape[1] != spec["input_features"] or set(np.unique(y)) != set(range(spec["num_classes"])):
        raise ValueError("Dataset dimensions or class labels do not match the selected task")
    indices = np.arange(len(y))
    train, held = train_test_split(indices, test_size=0.4, stratify=y, random_state=seed)
    val, test = train_test_split(held, test_size=0.5, stratify=y[held], random_state=seed)
    numeric = 10 if dataset == "covertype" else x.shape[1]
    scaler = StandardScaler().fit(x[train, :numeric])
    datasets = []
    for idx in (train, val, test):
        features = x[idx].astype(np.float32)
        features[:, :numeric] = scaler.transform(x[idx, :numeric]).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(features), torch.tensor(y[idx], dtype=torch.long))
        ds.row_indices = idx
        datasets.append(ds)
    metadata = {"name": dataset, "input_features": spec["input_features"], "num_classes": spec["num_classes"],
                "split_seed": seed, "train_samples": len(train), "validation_samples": len(val),
                "test_samples": len(test), "split_policy": "stratified 60/20/20, fixed dataset ordering",
                "split_index_sha256": {name: hashlib.sha256(idx.tobytes()).hexdigest()
                                       for name, idx in zip(("train", "validation", "test"), (train, val, test))},
                "preprocessing": "StandardScaler fitted exclusively on training rows; binary columns unchanged",
                "scaled_feature_columns": list(range(numeric)),
                "class_counts": {name: np.bincount(y[idx], minlength=spec["num_classes"]).tolist()
                                 for name, idx in zip(("train", "validation", "test"), (train, val, test))},
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "data_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()}
    if dataset == DATASET_NAME:
        metadata["split_indices"] = {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}
    return tuple(datasets), metadata


def dataset_metadata(seed=42, dataset=DATASET_NAME):
    return _split(seed, dataset)[1]


def get_full_loaders(batch_size=None, seed=42, training_seed=42, dataset=DATASET_NAME):
    datasets, _ = _split(seed, dataset)
    batch_size = batch_size or dataset_spec(dataset)["batch_size"]
    generator = torch.Generator().manual_seed(training_seed)
    return tuple(DataLoader(ds, batch_size=batch_size, shuffle=i == 0, num_workers=0,
                            generator=generator if i == 0 else None) for i, ds in enumerate(datasets))


def _subset(ds, size, seed):
    if size == 0 or size == len(ds):
        return ds
    if not 0 < size < len(ds):
        raise ValueError(f"Subset size must be 0 (all rows) or 1..{len(ds)}")
    try:
        selected, _ = train_test_split(np.arange(len(ds)), train_size=size,
                                      stratify=ds.tensors[1].numpy(), random_state=seed)
    except ValueError as error:
        raise ValueError("Subset must be large enough for stratification across all classes") from error
    result = TensorDataset(*(tensor[selected] for tensor in ds.tensors))
    result.row_indices = ds.row_indices[selected]
    return result


@lru_cache(maxsize=6)
def _proxy_datasets(seed, dataset, proxy_size, validation_size):
    datasets, _ = _split(seed, dataset)
    return _subset(datasets[0], proxy_size, seed), _subset(datasets[1], validation_size, seed)


def get_proxy_loaders(proxy_size=None, batch_size=None, seed=42, training_seed=42,
                      dataset=DATASET_NAME, validation_size=None):
    proxy_size, validation_size = search_sizes(dataset, proxy_size, validation_size)
    train, val = _proxy_datasets(seed, dataset, proxy_size, validation_size)
    batch_size = batch_size or dataset_spec(dataset)["batch_size"]
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0,
                       generator=torch.Generator().manual_seed(training_seed)),
            DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0))
