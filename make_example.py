"""Save one raw training row as a prediction request; never sample the test set."""
import argparse
import json
from pathlib import Path
from data.tabular import _raw, get_full_loaders, dataset_metadata
from utils.persistence import atomic_json


def make_example(results_path, output):
    source = json.loads(Path(results_path).read_text(encoding="utf-8"))
    dataset, seed = source["dataset"]["name"], source["split_seed"]
    current = dataset_metadata(seed, dataset)
    if current["data_sha256"] != source["dataset"]["data_sha256"]:
        raise ValueError("Example dataset differs from model training data")
    train, _, _ = get_full_loaders(seed=seed, dataset=dataset)
    index = int(train.dataset.row_indices[0])
    x, _ = _raw(dataset)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"features": [x[index].tolist()]})
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    index = make_example(args.results, args.out)
    print(f"Saved raw training row {index} to {args.out}")
