"""Dataset dimensions and practical defaults; no training dependencies needed."""
DATASETS = {
    "breast_cancer_wisconsin": {"input_features": 30, "num_classes": 2, "train_size": 341,
                               "proxy_size": 0, "validation_size": 0, "batch_size": 32},
    "covertype": {"input_features": 54, "num_classes": 7, "train_size": 348607,
                 "proxy_size": 20000, "validation_size": 10000, "batch_size": 512},
}

# Small offline library default; command-line experiments select Covertype.
DEFAULT_DATASET = "breast_cancer_wisconsin"
INPUT_FEATURES = DATASETS[DEFAULT_DATASET]["input_features"]
NUM_CLASSES = DATASETS[DEFAULT_DATASET]["num_classes"]


def dataset_spec(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    return DATASETS[name]


def search_sizes(name, proxy_size=None, validation_size=None):
    spec = dataset_spec(name)
    proxy_size = spec["proxy_size"] if proxy_size is None else proxy_size
    validation_size = spec["validation_size"] if validation_size is None else validation_size
    if type(proxy_size) is not int or not 0 <= proxy_size <= spec["train_size"]:
        raise ValueError("Proxy size must be zero (all training rows) or within the training split")
    if type(validation_size) is not int or validation_size < 0:
        raise ValueError("Validation size must be zero (all validation rows) or a positive integer")
    return proxy_size, validation_size
