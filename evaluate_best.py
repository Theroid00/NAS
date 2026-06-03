"""
evaluate_best.py
================
Standalone script to fully train the best architecture found by the GA.

Usage:
  python evaluate_best.py                        # auto-finds most recent best_*.json
  python evaluate_best.py --json path/to/best.json
  python evaluate_best.py --device cuda --epochs 100
"""

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Full training for GA-NAS best architecture")
    p.add_argument("--json",    default=None, help="Path to best_*.json (auto-found if omitted)")
    p.add_argument("--device",  default="cpu")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--save-dir", default="experiments/best_architectures")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve JSON path
    if args.json:
        json_path = args.json
    else:
        d = args.save_dir
        if not os.path.isdir(d):
            print(f"[Error] {d} does not exist. Run main.py --mode nas first.")
            sys.exit(1)
        candidates = sorted(
            [f for f in os.listdir(d) if f.startswith("best_") and f.endswith(".json")],
            reverse=True,
        )
        if not candidates:
            print(f"[Error] No best_*.json found in {d}.")
            sys.exit(1)
        json_path = os.path.join(d, candidates[0])

    print(f"Loading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    chromosome = data["best_chromosome"]
    run_id = data.get("run_id", "best")

    print(f"Chromosome : {chromosome}")
    print(f"GA fitness : {data.get('best_fitness', 'N/A')}")
    print(f"Architecture:")
    for k, v in data.get("best_arch", {}).items():
        print(f"  {k:14s}: {v}")

    # Full train
    import torch
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable, falling back to CPU.")
        device = "cpu"

    from training.trainer import full_train
    results = full_train(
        chromosome=chromosome,
        device=device,
        epochs=args.epochs,
        save_dir=args.save_dir,
        run_id=run_id,
    )

    print(f"\nFinal Test Accuracy : {results['test_accuracy']:.4f}")
    print(f"Parameters          : {results['num_params']:,}")

    # Plot comparison if baseline results exist
    try:
        from utils.visualiser import plot_architecture, plot_comparison_bar
        from ga.chromosome import decode
        arch = decode(chromosome)
        plot_architecture(arch)

        compare = {"GA-NAS": results["test_accuracy"]}
        plot_comparison_bar(compare)
    except Exception as e:
        print(f"[Visualiser] {e}")


if __name__ == "__main__":
    main()
