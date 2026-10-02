"""Repeat GA, aging evolution, and random search with identical trial budgets."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
from uuid import uuid4

from ga.engine import run_nas
from ga.aging import run_aging_evolution
from models.baselines.random_nas import run_random_search
from utils.search_runtime import validate_budget
from ga.chromosome import SCHEMA_VERSION


def compare(methods, seeds, budget=300, population=20, proxy_epochs=20, device="cpu",
            split_seed=42, proxy_size=0, smoke=False, max_params=None,
            out_dir="experiments/tabular/comparisons", full_epochs=None, full_seeds=(101,)):
    validate_budget(budget, proxy_epochs, population)
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= {"ga", "aging", "random"}:
        raise ValueError("Choose distinct methods from ga, aging, random")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choose at least one distinct search seed")
    if not full_seeds or len(set(full_seeds)) != len(full_seeds):
        raise ValueError("Choose distinct final training seeds")
    if full_epochs is not None and (full_epochs < 1 or smoke):
        raise ValueError("Full training requires positive epochs and real search results")
    root = Path(out_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    root.mkdir(parents=True)
    manifest = {"dataset_name": "breast_cancer_wisconsin", "schema_version": SCHEMA_VERSION,
                "config": {"methods": list(methods), "population": population,
                           "proxy_epochs": proxy_epochs, "proxy_size": proxy_size,
                           "device": device, "max_params": max_params, "full_epochs": full_epochs},
                "budget_kind": "candidate_evaluations", "evaluation_budget": budget,
                "smoke": smoke, "split_seed": split_seed, "search_seeds": list(seeds),
                "full_training_seeds": list(full_seeds), "runs": [], "status": "running"}
    path = root / "comparison.json"
    def save():
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    try:
        for seed in seeds:
            for method in methods:
                directory = root / method / str(seed)
                common = dict(device=device, proxy_epochs=proxy_epochs, seed=seed, split_seed=split_seed,
                              proxy_size=proxy_size, smoke=smoke, max_params=max_params,
                               log_dir=str(directory), save_dir=str(directory))
                if method == "ga":
                    result = run_nas(population_size=population, evaluation_budget=budget, **common)
                elif method == "aging":
                    result = run_aging_evolution(n_evaluations=budget, population_size=population, **common)
                else:
                    result = run_random_search(n_evaluations=budget, **common)
                row = {"method": method, "seed": seed, "best_proxy_fitness": result["best_fitness"],
                       "evaluation_count": result["evaluation_count"], "search_elapsed_s": result["elapsed_s"],
                       "total_evaluation_seconds": result["total_evaluation_seconds"],
                       "winner_path": result["save_path"], "full_training": []}
                manifest["runs"].append(row)
                save()
                if full_epochs is not None:
                    from training.trainer import full_train
                    for full_seed in full_seeds:
                        full = full_train(result["best_chromosome"], device=device, epochs=full_epochs,
                                          seed=full_seed, split_seed=split_seed, save_dir=str(directory),
                                          run_id=f"{result['run_id']}_seed{full_seed}", evaluate_test=False)
                        row["full_training"].append({"seed": full_seed, "best_val_accuracy": full["best_val_accuracy"],
                                                      "num_params": full["num_params"], "results_path": full["results_path"]})
                        save()
        manifest["summary"] = {}
        for method in methods:
            values = [r["best_proxy_fitness"] for r in manifest["runs"] if r["method"] == method]
            times = [r["search_elapsed_s"] for r in manifest["runs"] if r["method"] == method]
            full_values = [statistics.mean(f["best_val_accuracy"] for f in r["full_training"])
                           for r in manifest["runs"] if r["method"] == method and r["full_training"]]
            manifest["summary"][method] = {"n_searches": len(values), "proxy_mean": statistics.mean(values),
                                            "proxy_std": statistics.stdev(values) if len(values) > 1 else None,
                                            "mean_search_elapsed_s": statistics.mean(times)}
            if full_values:
                manifest["summary"][method].update(full_validation_mean=statistics.mean(full_values),
                                                   full_validation_std=statistics.stdev(full_values) if len(full_values) > 1 else None)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save()
    print(f"Comparison saved: {path}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=["ga", "aging", "random"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--proxy-epochs", type=int, default=20)
    p.add_argument("--proxy-size", type=int, default=0, help="0 uses all training rows")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-params", type=int)
    p.add_argument("--full-epochs", type=int)
    p.add_argument("--full-seeds", nargs="+", type=int, default=[101])
    p.add_argument("--out-dir", default="experiments/tabular/comparisons")
    args = vars(p.parse_args())
    compare(**args)


if __name__ == "__main__":
    main()
