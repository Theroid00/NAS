"""Repeated population/proxy-epoch sweeps; record unequal training costs explicitly."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4
from ga.engine import run_nas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--budget", type=int, default=150)
    p.add_argument("--populations", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--proxy-epochs", nargs="+", type=int, default=[5, 10, 20])
    from data.specs import DATASETS
    p.add_argument("--dataset", choices=list(DATASETS), default="covertype")
    p.add_argument("--device", default="cuda")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-dir", default="experiments/tabular/comparisons")
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.populations) < 2 or args.budget < max(args.populations) or min(args.proxy_epochs) < 1:
        p.error("Use distinct seeds, positive epochs, populations >=2, and sufficient budget")
    root = Path(args.out_dir) / ("sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    root.mkdir(parents=True)
    result = {"config": vars(args), "budget_kind": "candidate_evaluations; proxy epochs vary",
              "runs": [], "status": "running"}
    path = root / "sweep.json"
    try:
        for seed in args.seeds:
            for population in args.populations:
                for epochs in args.proxy_epochs:
                    directory = root / f"pop{population}_epochs{epochs}" / str(seed)
                    run = run_nas(evaluation_budget=args.budget, population_size=population,
                                  proxy_epochs=epochs, device=args.device, dataset=args.dataset, seed=seed,
                                  split_seed=args.split_seed, smoke=args.smoke,
                                  log_dir=str(directory), save_dir=str(directory))
                    result["runs"].append({"population": population, "proxy_epochs": epochs, "seed": seed,
                                           "best_fitness": run["best_fitness"], "elapsed_s": run["elapsed_s"],
                                           "evaluation_count": run["evaluation_count"], "winner_path": run["save_path"]})
                    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
