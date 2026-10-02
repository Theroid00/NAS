"""Measure proxy/long-training rank agreement without evaluating the test set."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import statistics
from uuid import uuid4
from ga.chromosome import random_chromosome, decode


def ranks(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for i in ordered[start:end]:
            result[i] = (start + end - 1) / 2 + 1
        start = end
    return result


def ranking_metrics(proxy, longer, top_k=3):
    if len(proxy) != len(longer) or len(proxy) < 3 or not 1 <= top_k <= len(proxy):
        raise ValueError("Need at least three paired observations and a valid top-k")
    x, y = ranks(proxy), ranks(longer)
    correlation = statistics.correlation(x, y) if len(set(x)) > 1 and len(set(y)) > 1 else None
    top_x = set(sorted(range(len(x)), key=lambda i: (-proxy[i], i))[:top_k])
    top_y = set(sorted(range(len(y)), key=lambda i: (-longer[i], i))[:top_k])
    return {"spearman": correlation, "top_k": top_k, "top_k_overlap_fraction": len(top_x & top_y) / top_k,
            "tie_policy": "average_ranks; shortlist ties resolved by candidate order"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--seeds", nargs="+", type=int, default=[101, 102])
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--proxy-epochs", type=int, default=20)
    p.add_argument("--validation-size", type=int, default=None, help="Default depends on dataset; 0 uses all validation rows")
    p.add_argument("--proxy-size", type=int, default=None, help="Default depends on dataset; 0 uses all training rows")
    p.add_argument("--long-epochs", type=int, default=100)
    p.add_argument("--top-k", type=int, default=3)
    from data.specs import DATASETS
    p.add_argument("--dataset", choices=list(DATASETS), default="covertype")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="experiments/tabular/proxy_calibration")
    args = p.parse_args()
    if not 3 <= args.samples <= 1000 or not 1 <= args.top_k <= args.samples or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        p.error("Use at least three samples, distinct training seeds, and top-k within sample count")
    from data.specs import search_sizes
    args.proxy_size, args.validation_size = search_sizes(args.dataset, args.proxy_size, args.validation_size)
    if args.proxy_epochs < 1 or args.long_epochs <= args.proxy_epochs:
        p.error("Use positive proxy epochs, longer reference training, and a valid proxy size")
    from training.evaluator import evaluate_trial
    from training.trainer import full_train
    rng = random.Random(args.sample_seed)
    chromosomes, seen = [], set()
    while len(chromosomes) < args.samples:
        chromosome = random_chromosome(rng)
        arch = decode(chromosome)
        active = {k: v for k, v in arch.items() if not k.startswith("width_") or int(k.split("_")[1]) <= arch["num_layers"]}
        widths = [30] + [arch[f"width_{j}"] for j in range(1, arch["num_layers"] + 1)]
        if not any(a == b for a, b in zip(widths, widths[1:])):
            active["use_residual"] = False
        key = json.dumps(active, sort_keys=True)
        if key not in seen:
            chromosomes.append(chromosome)
            seen.add(key)
    root = Path(args.out_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    root.mkdir(parents=True)
    result = {"config": vars(args), "test_evaluated": False, "status": "running", "candidates": []}
    path = root / "calibration.json"
    def save():
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save()
    try:
        for index, chromosome in enumerate(chromosomes):
            row = {"candidate_id": index + 1, "chromosome": chromosome, "arch": decode(chromosome), "repeats": []}
            result["candidates"].append(row)
            for seed in args.seeds:
                proxy = evaluate_trial(chromosome, args.device, args.proxy_epochs, seed=seed,
                                       split_seed=args.split_seed, proxy_size=args.proxy_size, dataset=args.dataset, validation_size=args.validation_size)
                if proxy["status"] != "ok":
                    row["repeats"].append({"seed": seed, "proxy": proxy, "reference": None})
                    save()
                    continue
                full = full_train(chromosome, device=args.device, epochs=args.long_epochs, seed=seed,
                                  split_seed=args.split_seed, evaluate_test=False, dataset=args.dataset, save_dir=str(root),
                                  run_id=f"candidate{index + 1}_seed{seed}")
                row["repeats"].append({"seed": seed, "proxy": proxy,
                                       "reference": {"validation_accuracy": full["best_val_accuracy"],
                                                     "results_path": full["results_path"]}})
                save()
        paired = []
        for row in result["candidates"]:
            valid = [r for r in row["repeats"] if r["reference"] is not None]
            # Compare only candidates with complete repeated measurements.
            if len(valid) == len(args.seeds):
                paired.append((statistics.mean(r["proxy"]["fitness"] for r in valid),
                               statistics.mean(r["reference"]["validation_accuracy"] for r in valid)))
        result["valid_candidates"] = len(paired)
        result["metrics"] = ranking_metrics([r[0] for r in paired], [r[1] for r in paired],
                                             min(args.top_k, len(paired))) if len(paired) >= 3 else None
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        save()
    print(f"Proxy calibration saved: {path}")


if __name__ == "__main__":
    main()
