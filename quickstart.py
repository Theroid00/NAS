"""Small real offline workflow; its scores are integration checks, not benchmarks."""
import argparse
from pathlib import Path
from uuid import uuid4

from run_comparison import compare
from utils.records import read_record
from serving.artifact import export_artifact
from make_example import make_example


def run(device="cuda", output="experiments/tabular/quickstart"):
    root = Path(output).resolve() / uuid4().hex[:8]
    compare(["ga", "aging", "random"], [7], budget=4, population=2,
            proxy_epochs=1, full_epochs=2, full_seeds=[101], device=device,
            dataset="breast_cancer_wisconsin", out_dir=root)
    path = next(root.glob("*/comparison.json"))
    comparison = read_record(path)
    selected = max((full for row in comparison["runs"] for full in row["full_training"]),
                   key=lambda full: full["best_val_accuracy"])
    example = path.parent / "example.json"
    make_example(selected["results_path"], example)
    artifact = path.parent / "artifact"
    export_artifact(selected["results_path"], artifact, example)
    print("Integration workflow completed; no test evaluation or algorithm superiority claim.")
    print(f"Comparison: {path}\nArtifact: {artifact}")
    return path, artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="experiments/tabular/quickstart")
    args = parser.parse_args()
    run(args.device, args.out_dir)
