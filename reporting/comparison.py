"""Build a portable report from recorded trials and full-training results."""
import json
import math
from pathlib import Path
import statistics

from utils.persistence import atomic_json
from utils.records import read_record


def read(path):
    return read_record(path)


def build_report(comparison_path):
    manifest = read(comparison_path)
    report = {"report_version": 1, "dataset": manifest["dataset_name"], "status": manifest["status"],
              "smoke": manifest["smoke"], "config": manifest["config"],
              "budget": manifest["evaluation_budget"], "search_seeds": manifest["search_seeds"],
              "split_seed": manifest["split_seed"], "runs": [], "summary": {}, "selected_model": None,
              "selection_policy": "Highest full-training validation accuracy; earliest result breaks ties",
              "metric_policy": "Accuracy selects winners; balanced accuracy and macro F1 are diagnostics"}
    for row in manifest["runs"]:
        winner = read(row["winner_path"])
        records = [json.loads(line) for line in Path(winner["trial_path"]).read_text(encoding="utf-8").splitlines()]
        points, best, elapsed, valid, failures = [], None, 0., 0, 0
        for record in records:
            elapsed += record.get("elapsed_s", 0.)
            if record["status"] == "error":
                failures += 1
                continue
            if record["status"] in ("ok", "smoke"):
                valid += 1
                best = record["fitness"] if best is None else max(best, record["fitness"])
            else:
                failures += 1
            points.append({"evaluations": record["trial_id"], "elapsed_s": elapsed,
                           "best_accuracy": best, "trial_accuracy": record["fitness"]})
        full = [read(item["results_path"]) for item in row["full_training"]]
        run = {"method": row["method"], "seed": row["seed"], "accuracy": row["best_proxy_fitness"],
               "metrics": winner.get("best_validation_metrics"), "num_params": winner.get("num_params"),
               "architecture": winner["best_arch"], "search_elapsed_s": row["search_elapsed_s"],
               "evaluation_count": row["evaluation_count"], "valid_trials": valid, "failed_attempts": failures,
               "progress": points, "full_training": []}
        for result in full:
            item = {"model_id": result["run_id"], "seed": result["seed"], "accuracy": result["best_val_accuracy"],
                    "metrics": result.get("best_val_metrics"), "test_metrics": result.get("test_metrics"),
                    "num_params": result["num_params"], "elapsed_s": result["elapsed_s"],
                    "epochs": result["protocol"]["epochs"], "results_path": result["results_path"]}
            run["full_training"].append(item)
            if report["selected_model"] is None or item["accuracy"] > report["selected_model"]["accuracy"]:
                report["selected_model"] = {**item, "method": row["method"], "search_seed": row["seed"],
                                            "architecture": winner["best_arch"]}
        report["runs"].append(run)
    for method in manifest["config"]["methods"]:
        runs = [r for r in report["runs"] if r["method"] == method]
        summary = {"searches_completed": len(runs)}
        values = {"proxy_accuracy": [r["accuracy"] for r in runs],
                  "search_seconds": [r["search_elapsed_s"] for r in runs]}
        for metric in ("balanced_accuracy", "macro_f1"):
            values["proxy_" + metric] = [(r["metrics"] or {}).get(metric) for r in runs]
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values["full_validation_" + metric] = [statistics.mean((f["metrics"] or {}).get(metric, f["accuracy"] if metric == "accuracy" else float("nan"))
                                                                 for f in r["full_training"])
                                                   for r in runs if r["full_training"]]
        for name, measurements in values.items():
            if measurements and all(v is not None and isinstance(v, (float, int)) and math.isfinite(v) for v in measurements):
                summary[name] = {"mean": statistics.mean(measurements),
                                 "std": statistics.stdev(measurements) if len(measurements) > 1 else None,
                                 "n": len(measurements)}
        report["summary"][method] = summary
    return report


def save_report(comparison_path, output):
    report = build_report(comparison_path)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report
