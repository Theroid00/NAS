"""Plot measured results from explicitly supplied full-training JSON files."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--metric", choices=["test_accuracy", "best_val_accuracy"], default="test_accuracy")
    p.add_argument("--out", default="experiments/tabular/comparisons/measured_comparison.png")
    args = p.parse_args()
    labels = args.labels or [Path(path).stem for path in args.results]
    if len(labels) != len(args.results) or len(set(labels)) != len(labels):
        p.error("Provide one distinct label per results file")
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.results]
    datasets = {record.get("dataset", {}).get("name", "historical_unspecified") for record in records}
    if len(datasets) != 1:
        p.error("Results belong to different datasets; tabular and historical CIFAR scores cannot be compared")
    measurements = [record[args.metric] for record in records]
    if any(value is None for value in measurements):
        p.error("Requested metric has not been evaluated in every result")
    from utils.visualiser import plot_comparison_bar
    plot_comparison_bar(dict(zip(labels, measurements)), args.out, metric_label=args.metric)


if __name__ == "__main__":
    main()
