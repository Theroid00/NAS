"""Independently verify archived search winners and replay their GPU metrics."""
import argparse
import json
import math
from pathlib import Path
import statistics
from unittest.mock import patch
from utils.records import read_record


def audit(comparison_path, output_path):
    import numpy as np
    import torch
    from scipy.special import logsumexp
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from training.evaluator import evaluate_trial, validate

    manifest = read_record(comparison_path)
    assert not manifest["smoke"]
    report = {"comparison": str(comparison_path), "device": torch.cuda.get_device_name(0),
              "test_set_evaluated": False, "runs": []}
    subset_hashes = set()
    for run in manifest["runs"]:
        winner = read_record(run["winner_path"])
        trials = [json.loads(line) for line in Path(winner["trial_path"]).read_text(encoding="utf-8").splitlines()]
        assert len(trials) == manifest["evaluation_budget"] == run["evaluation_count"]
        best = max((trial for trial in trials if trial["status"] == "ok"), key=lambda trial: trial["fitness"])
        assert best["trial_id"] == winner["best_trial_id"]
        assert best["chromosome"] == winner["best_chromosome"]
        assert best["fitness"] == winner["best_fitness"] == run["best_proxy_fitness"]
        assert best["validation_loss"] == winner["best_validation_loss"] == run["best_proxy_validation_loss"]
        metadata = read_record(winner["metadata_path"])
        protocol = metadata["proxy_protocol"]
        subset_hashes.add((protocol["train_indices_sha256"], protocol["validation_indices_sha256"]))
        independent = {}

        def independent_validate(model, loader, device, return_metrics=False):
            metrics = validate(model, loader, device, return_metrics=True)
            logits, targets = [], []
            with torch.no_grad():
                for x, y in loader:
                    logits.append(model(x.to(device)).cpu().numpy())
                    targets.append(y.numpy())
            values = np.concatenate(logits).astype(np.float64)
            labels = np.concatenate(targets)
            predictions = values.argmax(axis=1)
            accuracy = float(accuracy_score(labels, predictions))
            loss = float((logsumexp(values, axis=1) - values[np.arange(len(labels)), labels]).mean())
            assert accuracy == metrics["accuracy"]
            assert math.isclose(loss, metrics["loss"], abs_tol=1e-6)
            assert len(labels) == metrics["samples"] == protocol["validation_samples"]
            independent.update(accuracy=accuracy, loss=loss, samples=len(labels),
                               balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
                               confusion_matrix=confusion_matrix(labels, predictions, labels=np.arange(values.shape[1])).tolist())
            return metrics if return_metrics else metrics["accuracy"]

        with patch("training.evaluator.validate", side_effect=independent_validate):
            replay = evaluate_trial(best["chromosome"], device="cuda:0", proxy_epochs=best["proxy_epochs"],
                                    seed=best["seed"], split_seed=best["split_seed"], proxy_size=best["proxy_size"],
                                    dataset=best["dataset_name"], validation_size=best["validation_size"],
                                    max_params=winner["hyperparams"].get("max_params"))
        assert replay["status"] == "ok"
        assert replay["fitness"] == best["fitness"]
        assert math.isclose(replay["validation_loss"], best["validation_loss"], abs_tol=1e-6)
        report["runs"].append({"method": run["method"], "seed": run["seed"], "trial_id": best["trial_id"],
                               "budget_verified": len(trials), "archive_winner_verified": True,
                               "replayed_accuracy_matches": True, "metrics": independent})
        print(f'{run["method"]} seed {run["seed"]}: accuracy {independent["accuracy"]:.4f}, '
              f'balanced accuracy {independent["balanced_accuracy"]:.4f}; archive and replay agree', flush=True)
    assert len(subset_hashes) == 1
    report["identical_proxy_subsets_verified"] = True
    for method, summary in manifest["summary"].items():
        scores = [run["best_proxy_fitness"] for run in manifest["runs"] if run["method"] == method]
        assert statistics.mean(scores) == summary["proxy_mean"]
        assert statistics.stdev(scores) == summary["proxy_std"]
    report["comparison_summary_verified"] = True
    Path(output_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/scoring-audit.json"))
    args = parser.parse_args()
    audit(args.comparison, args.output)
