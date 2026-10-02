"""Repeated operator ablations with a common candidate-evaluation budget."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
from uuid import uuid4
from ga.engine import run_nas

CONFIGS = {"baseline": {}, "no_crossover": {"crossover_prob": 0.0},
           "no_mutation": {"mutation_prob": 0.0}, "no_elitism": {"n_elites": 0},
           "small_pop": {"population_size": 10}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--budget", type=int, default=200)
    from data.specs import DATASETS
    p.add_argument("--dataset", choices=list(DATASETS), default="covertype")
    p.add_argument("--device", default="cpu")
    p.add_argument("--proxy-epochs", type=int, default=20)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-dir", default="experiments/tabular/comparisons")
    args = p.parse_args()
    if args.budget < 20 or len(set(args.seeds)) != len(args.seeds):
        p.error("Budget must cover population 20; seeds must be distinct")
    root = Path(args.out_dir) / ("ablations_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    root.mkdir(parents=True)
    result = {"config": vars(args), "budget_kind": "candidate_evaluations", "runs": [], "status": "running"}
    path = root / "ablations.json"
    try:
        for seed in args.seeds:
            for name, settings in CONFIGS.items():
                directory = root / name / str(seed)
                run = run_nas(evaluation_budget=args.budget, proxy_epochs=args.proxy_epochs,
                              device=args.device, dataset=args.dataset, seed=seed, split_seed=args.split_seed,
                              smoke=args.smoke, log_dir=str(directory), save_dir=str(directory), **settings)
                result["runs"].append({"name": name, "seed": seed, "best_fitness": run["best_fitness"],
                                       "evaluation_count": run["evaluation_count"], "winner_path": run["save_path"]})
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["summary"] = {name: statistics.mean(r["best_fitness"] for r in result["runs"] if r["name"] == name)
                              for name in CONFIGS}
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
