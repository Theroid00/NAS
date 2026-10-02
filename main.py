"""CLI for reproducible tabular MLP NAS experiments."""
import argparse
from training.config import RANDOM_SEED


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["nas", "aging", "random-search", "train-best", "train-mlp", "plot"], default="nas")
    p.add_argument("--pop", type=int, default=20)
    p.add_argument("--gen", type=int, default=15)
    p.add_argument("--proxy-epochs", type=int, default=20)
    p.add_argument("--proxy-size", type=int, default=0, help="0 uses all training rows")
    p.add_argument("--crossover-p", type=float, default=0.8)
    p.add_argument("--mutation-p", type=float, default=0.1)
    p.add_argument("--tournament-k", type=int, default=5)
    p.add_argument("--elites", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-eval", type=int, default=300)
    p.add_argument("--max-params", type=int)
    p.add_argument("--best-json")
    p.add_argument("--full-epochs", type=int, default=100)
    p.add_argument("--validation-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--split-seed", type=int, default=RANDOM_SEED)
    p.add_argument("--log-dir", default="experiments/tabular/generation_logs")
    p.add_argument("--save-dir", default="experiments/tabular/best_architectures")
    return p.parse_args(argv)


def main():
    args = parse_args()
    common = dict(device=args.device, proxy_epochs=args.proxy_epochs, smoke=args.smoke,
                  seed=args.seed, split_seed=args.split_seed, proxy_size=args.proxy_size,
                  max_params=args.max_params,
                  log_dir=args.log_dir, save_dir=args.save_dir)
    if args.mode == "nas":
        from ga.engine import run_nas
        result = run_nas(population_size=args.pop, n_generations=args.gen,
                         crossover_prob=args.crossover_p, mutation_prob=args.mutation_p,
                         tournament_k=args.tournament_k, n_elites=args.elites, **common)
    elif args.mode == "aging":
        from ga.aging import run_aging_evolution
        result = run_aging_evolution(n_evaluations=args.n_eval, population_size=args.pop,
                                    tournament_k=args.tournament_k, **common)
    elif args.mode == "random-search":
        from models.baselines.random_nas import run_random_search
        result = run_random_search(n_evaluations=args.n_eval, **common)
    elif args.mode in ("train-best", "train-mlp"):
        from utils.results import load_winner, latest_winner
        from training.trainer import full_train
        record = {} if args.mode == "train-mlp" else load_winner(args.best_json or latest_winner(args.save_dir))
        result = full_train(record.get("best_chromosome"), baseline=args.mode == "train-mlp",
                            device=args.device, epochs=args.full_epochs, seed=args.seed,
                            split_seed=args.split_seed, save_dir=args.save_dir,
                            run_id=record.get("run_id", "mlp"),
                            evaluate_test=not args.validation_only)
        print(result["results_path"])
        return
    else:
        from utils.visualiser import plot_convergence
        print(plot_convergence(args.log_dir))
        return
    print(f"Best observed proxy fitness: {result['best_fitness']:.4f}")
    print(f"Winner saved: {result['save_path']}")
    try:
        from pathlib import Path
        from utils.visualiser import plot_convergence, plot_architecture
        if result["log_path"]:
            plot_convergence(result["log_path"])
        plot_architecture(result["best_arch"], str(Path(args.save_dir) / f"architecture_{result['run_id']}.png"))
    except ImportError as error:
        print(f"Plotting dependencies unavailable: {error}")


if __name__ == "__main__":
    main()
