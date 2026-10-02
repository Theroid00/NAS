"""Offline NAS-Bench-201 topology comparisons through its NATS-Bench successor."""
import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import random
import statistics
from uuid import uuid4

OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")


def architecture_string(genes):
    if len(genes) != 6 or any(type(g) is not int or not 0 <= g < 5 for g in genes):
        raise ValueError("Topology requires six operation indices within 0..4")
    operations = [OPS[g] for g in genes]
    return (f"|{operations[0]}~0|+|{operations[1]}~0|{operations[2]}~1|+"
            f"|{operations[3]}~0|{operations[4]}~1|{operations[5]}~2|")


def search(method, evaluate, seed, budget=300, population_size=20, tournament_k=5):
    """Apply the three search policies to the benchmark's six-edge encoding."""
    if method not in ("ga", "aging", "random") or budget < population_size or population_size < 2 or tournament_k < 1:
        raise ValueError("Invalid method or benchmark budget/population")
    rng = random.Random(seed)
    history = []
    def sample():
        return [rng.randrange(5) for _ in range(6)]
    def score(genes):
        info = evaluate(genes)
        value = info["fitness"]
        if not 0 <= value <= 1:
            raise ValueError("Benchmark validation fitness must be within 0..1")
        history.append({**info, "genes": genes[:], "trial_id": len(history) + 1})
        return (genes[:], value)
    population = deque(score(sample()) for _ in range(population_size))
    stagnant, best_seen = 0, max(p[1] for p in population)
    while len(history) < budget:
        if method == "random":
            score(sample())
        elif method == "aging":
            parent = max(rng.choices(list(population), k=tournament_k), key=lambda p: p[1])[0]
            child = parent[:]
            position = rng.randrange(6)
            child[position] = rng.choice([v for v in range(5) if v != child[position]])
            population.append(score(child))
            population.popleft()
        else:
            elites = [p[0][:] for p in sorted(population, key=lambda p: p[1], reverse=True)[:2]]
            offspring = []
            while len(elites) + len(offspring) < population_size:
                parents = [max(rng.choices(list(population), k=tournament_k), key=lambda p: p[1])[0][:] for _ in range(2)]
                if rng.random() < 0.8:
                    point = rng.randrange(1, 6)
                    parents = [parents[0][:point] + parents[1][point:], parents[1][:point] + parents[0][point:]]
                mutation = 0.2 if stagnant >= 3 else 0.1
                for child in parents:
                    for i in range(6):
                        if rng.random() < mutation:
                            child[i] = rng.randrange(5)
                offspring.extend(parents)
            population = deque(score(c) for c in (elites + offspring)[:min(population_size, budget - len(history))])
            best = max(p[1] for p in population)
            if best > best_seen:
                best_seen, stagnant = best, 0
            else:
                stagnant += 1
    winner = max(history, key=lambda r: r["fitness"])
    return {"method": method, "seed": seed, "evaluation_count": len(history),
            "winner": winner, "history": history,
            "simulated_training_seconds": sum(r.get("training_seconds", 0) or 0 for r in history)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, help="Local NATS-tss benchmark file or unpacked simple directory")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(30)))
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--hp", choices=["12", "200"], default="12")
    p.add_argument("--out-dir", default="experiments/comparisons")
    args = p.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or args.population < 2 or args.budget < args.population:
        p.error("Use distinct seeds and a budget covering population >=2")
    from nats_bench import create
    api = create(args.benchmark, "tss", fast_mode=Path(args.benchmark).is_dir(), verbose=False)
    def evaluate(genes):
        index = api.query_index_by_arch(architecture_string(genes))
        if index < 0:
            raise ValueError("Architecture is absent from the benchmark")
        info = api.get_more_info(index, "cifar10-valid", hp=args.hp, is_random=False)
        return {"fitness": info["valid-accuracy"] / 100,
                "training_seconds": info["train-all-time"], "architecture_index": index}
    root = Path(args.out_dir) / ("benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    root.mkdir(parents=True)
    result = {"config": vars(args), "benchmark": "NATS-Bench tss / NAS-Bench-201 topology",
              "fitness_policy": "mean_stored_validation_accuracy", "status": "running", "runs": []}
    path = root / "benchmark.json"
    try:
        for seed in args.seeds:
            for method in ("ga", "aging", "random"):
                run = search(method, evaluate, seed, args.budget, args.population)
                result["runs"].append(run)
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["summary"] = {}
        for method in ("ga", "aging", "random"):
            scores = [r["winner"]["fitness"] for r in result["runs"] if r["method"] == method]
            result["summary"][method] = {"mean": statistics.mean(scores),
                                         "std": statistics.stdev(scores) if len(scores) > 1 else None}
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
