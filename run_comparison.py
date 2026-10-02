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
from utils.persistence import atomic_json, RunLock
from utils.provenance import source_fingerprint


def compare(methods, seeds, budget=300, population=20, proxy_epochs=20, device="cpu",
            split_seed=42, proxy_size=None, smoke=False, max_params=None,
            out_dir="experiments/tabular/comparisons", full_epochs=None, full_seeds=(101,), dataset="breast_cancer_wisconsin", validation_size=None, resume=None):
    from data.specs import search_sizes
    proxy_size, validation_size = search_sizes(dataset, proxy_size, validation_size)
    validate_budget(budget, proxy_epochs, population)
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= {"ga", "aging", "random"}:
        raise ValueError("Choose distinct methods from ga, aging, random")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choose at least one distinct search seed")
    if not full_seeds or len(set(full_seeds)) != len(full_seeds):
        raise ValueError("Choose distinct final training seeds")
    if full_epochs is not None and (full_epochs < 1 or smoke):
        raise ValueError("Full training requires positive epochs and real search results")
    root = Path(resume).resolve().parent if resume else Path(out_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=bool(resume))
    manifest = {"dataset_name": dataset, "schema_version": SCHEMA_VERSION,
                "config": {"methods": list(methods), "population": population,
                           "proxy_epochs": proxy_epochs, "proxy_size": proxy_size, "validation_size": validation_size,
                           "device": device, "max_params": max_params, "full_epochs": full_epochs},
                "budget_kind": "candidate_evaluations", "evaluation_budget": budget,
                "smoke": smoke, "split_seed": split_seed, "search_seeds": list(seeds),
                "full_training_seeds": list(full_seeds), "runs": [], "status": "running", "resume_version": 2,
                "source_fingerprint": source_fingerprint()}
    path = root / "comparison.json"
    lock = RunLock(path.with_suffix(".lock"))
    lock.acquire()
    if resume:
        try:
            saved = json.loads(Path(resume).read_text(encoding="utf-8"))
            if saved.get("resume_version") != 2:
                raise ValueError("Comparison predates source-verified recovery; use its original implementation")
            if saved.get("source_fingerprint") != manifest["source_fingerprint"]:
                raise ValueError("Source files changed; use the original implementation to resume the comparison")
            for key in ("dataset_name", "schema_version", "config", "evaluation_budget", "smoke",
                        "split_seed", "search_seeds", "full_training_seeds"):
                if saved[key] != manifest[key]:
                    raise ValueError(f"Comparison resume configuration differs: {key}")
            manifest = saved
            manifest["status"] = "running"
            manifest.pop("error", None)
        except BaseException:
            lock.release()
            raise
    def save():
        atomic_json(path, manifest)
    try:
        save()
        for seed in seeds:
            for method in methods:
                if source_fingerprint() != manifest["source_fingerprint"]:
                    raise ValueError("Source files changed during the comparison; restore its original implementation")
                directory = root / method / str(seed)
                common = dict(device=device, proxy_epochs=proxy_epochs, seed=seed, split_seed=split_seed,
                              proxy_size=proxy_size, dataset=dataset, validation_size=validation_size, smoke=smoke, max_params=max_params,
                               log_dir=str(directory), save_dir=str(directory))
                row = next((r for r in manifest["runs"] if r["method"] == method and r["seed"] == seed), None)
                saved_searches = list(directory.glob("metadata_*.json")) if resume and row is None else []
                if row is not None:
                    result = json.loads(Path(row["winner_path"]).read_text(encoding="utf-8"))
                    if result["evaluation_count"] != budget or result["best_fitness"] != row["best_proxy_fitness"]:
                        raise ValueError("Saved comparison winner is inconsistent")
                elif saved_searches:
                    if len(saved_searches) != 1:
                        raise ValueError("Ambiguous search metadata in comparison directory")
                    from utils.search_runtime import resume_search
                    result = resume_search(saved_searches[0])
                elif method == "ga":
                    result = run_nas(population_size=population, evaluation_budget=budget, **common)
                elif method == "aging":
                    result = run_aging_evolution(n_evaluations=budget, population_size=population, **common)
                else:
                    result = run_random_search(n_evaluations=budget, **common)
                if row is None:
                    row = {"method": method, "seed": seed, "best_proxy_fitness": result["best_fitness"], "best_proxy_validation_loss": result["best_validation_loss"],
                           "best_proxy_metrics": result.get("best_validation_metrics"), "num_params": result.get("num_params"),
                           "evaluation_count": result["evaluation_count"], "search_elapsed_s": result["elapsed_s"],
                           "total_evaluation_seconds": result["total_evaluation_seconds"],
                           "winner_path": result["save_path"], "full_training": []}
                    manifest["runs"].append(row)
                save()
                if full_epochs is not None:
                    from training.trainer import full_train
                    for full_seed in full_seeds:
                        if source_fingerprint() != manifest["source_fingerprint"]:
                            raise ValueError("Source files changed during the comparison; restore its original implementation")
                        if any(f["seed"] == full_seed for f in row["full_training"]):
                            continue
                        full_run_id = f"{result['run_id']}_seed{full_seed}"
                        completed_full = directory / f"full_train_{full_run_id}.json"
                        if completed_full.exists():
                            full = json.loads(completed_full.read_text(encoding="utf-8"))
                            if (full["seed"] != full_seed or full["chromosome"] != result["best_chromosome"]
                                    or full["split_seed"] != split_seed or full["protocol"]["epochs"] != full_epochs
                                    or full["dataset"]["name"] != dataset or full["test_accuracy"] is not None):
                                raise ValueError("Saved final training result is inconsistent")
                        else:
                            full = full_train(result["best_chromosome"], device=device, epochs=full_epochs,
                                          seed=full_seed, split_seed=split_seed, save_dir=str(directory),
                                          run_id=full_run_id, evaluate_test=False, dataset=dataset)
                        if full.get("source_fingerprint") != manifest["source_fingerprint"]:
                            raise ValueError("Final training source differs from the comparison implementation")
                        row["full_training"].append({"seed": full_seed, "best_val_accuracy": full["best_val_accuracy"], "best_val_loss": full["best_val_loss"],
                                                      "best_val_metrics": full.get("best_val_metrics"), "elapsed_s": full["elapsed_s"],
                                                      "num_params": full["num_params"], "results_path": full["results_path"]})
                        save()
        manifest["summary"] = {}
        for method in methods:
            values = [r["best_proxy_fitness"] for r in manifest["runs"] if r["method"] == method]
            times = [r["search_elapsed_s"] for r in manifest["runs"] if r["method"] == method]
            losses = [r["best_proxy_validation_loss"] for r in manifest["runs"]
                      if r["method"] == method and r["best_proxy_validation_loss"] is not None]
            full_values = [statistics.mean(f["best_val_accuracy"] for f in r["full_training"])
                           for r in manifest["runs"] if r["method"] == method and r["full_training"]]
            manifest["summary"][method] = {"n_searches": len(values), "proxy_mean": statistics.mean(values),
                                            "proxy_std": statistics.stdev(values) if len(values) > 1 else None,
                                            "mean_search_elapsed_s": statistics.mean(times)}
            if losses:
                manifest["summary"][method]["proxy_validation_loss_mean"] = statistics.mean(losses)
            for metric in ("balanced_accuracy", "macro_f1"):
                scores = [(r.get("best_proxy_metrics") or {}).get(metric) for r in manifest["runs"] if r["method"] == method]
                if all(score is not None for score in scores):
                    manifest["summary"][method]["proxy_" + metric + "_mean"] = statistics.mean(scores)
            if full_values:
                manifest["summary"][method].update(full_validation_mean=statistics.mean(full_values),
                                                   full_validation_std=statistics.stdev(full_values) if len(full_values) > 1 else None)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        try:
            save()
        finally:
            lock.release()
    print(f"Comparison saved: {path}")
    return manifest


def resume_comparison(path):
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    return compare(**saved["config"], seeds=saved["search_seeds"], budget=saved["evaluation_budget"],
                   split_seed=saved["split_seed"], smoke=saved["smoke"],
                   full_seeds=saved["full_training_seeds"], dataset=saved["dataset_name"], resume=path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=["ga", "aging", "random"])
    p.add_argument("--resume", help="Comparison JSON; restores all saved suite settings")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--proxy-epochs", type=int, default=20)
    p.add_argument("--validation-size", type=int, default=None, help="Default depends on dataset; 0 uses all validation rows")
    p.add_argument("--proxy-size", type=int, default=None, help="Default depends on dataset; 0 uses all training rows")
    p.add_argument("--split-seed", type=int, default=42)
    from data.specs import DATASETS
    p.add_argument("--dataset", choices=list(DATASETS), default="covertype")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-params", type=int)
    p.add_argument("--full-epochs", type=int)
    p.add_argument("--full-seeds", nargs="+", type=int, default=[101])
    p.add_argument("--out-dir", default="experiments/tabular/comparisons")
    args = vars(p.parse_args())
    if args["resume"]:
        resume_comparison(args["resume"])
    else:
        compare(**args)


if __name__ == "__main__":
    main()
