"""
main.py
=======
Entry point for the NAS with Genetic Algorithm project.

Usage examples:
  # Full run on GPU:
  python main.py --pop 20 --gen 15 --proxy-epochs 5 --device cuda

  # Dual-GPU parallel mode:
  python main.py --pop 20 --gen 15 --device cuda --parallel

  # Smoke test (no real training, quick sanity check):
  python main.py --pop 5 --gen 2 --smoke

  # Run random search baseline instead of GA:
  python main.py --mode random-search --n-eval 300 --device cuda
"""

import argparse
import os
import sys
import random
import numpy as np
import torch

from training.config import RANDOM_SEED


def set_seeds(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="NAS with Genetic Algorithm | CIFAR-10")

    # Mode
    p.add_argument(
        "--mode",
        choices=["nas", "random-search", "train-best", "plot", "train-resnet"],
        default="nas",
        help="Run mode: 'nas' (default), 'random-search', 'train-best', 'plot', or 'train-resnet'",
    )

    # GA hyperparameters
    p.add_argument("--pop",          type=int,   default=20,   help="Population size")
    p.add_argument("--gen",          type=int,   default=15,   help="Number of generations")
    p.add_argument("--proxy-epochs", type=int,   default=5,    help="Proxy training epochs per architecture")
    p.add_argument("--crossover-p",  type=float, default=0.8,  help="Crossover probability")
    p.add_argument("--mutation-p",   type=float, default=0.1,  help="Per-gene mutation probability")
    p.add_argument("--tournament-k", type=int,   default=3,    help="Tournament selection size")
    p.add_argument("--elites",       type=int,   default=2,    help="Number of elite individuals")

    # Device
    p.add_argument("--device",   default="cpu",  help="PyTorch device: cpu | cuda | cuda:0")
    p.add_argument("--parallel", action="store_true", help="Evaluate across cuda:0 and cuda:1")

    # Random search
    p.add_argument("--n-eval", type=int, default=300, help="Number of evaluations for random search")

    # Full training
    p.add_argument("--best-json",    default=None,   help="Path to best_*.json to full-train")
    p.add_argument("--full-epochs",  type=int, default=50, help="Epochs for full training")

    # Misc
    p.add_argument("--smoke",  action="store_true", help="Smoke test — skip real training")
    p.add_argument("--seed",   type=int, default=RANDOM_SEED)
    p.add_argument("--log-dir",  default="experiments/generation_logs")
    p.add_argument("--save-dir", default="experiments/best_architectures")

    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    # Auto-select device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU.")
        device = "cpu"

    # ------------------------------------------------------------------
    if args.mode == "nas":
        from ga.engine import run_nas
        result = run_nas(
            device=device,
            population_size=args.pop,
            n_generations=args.gen,
            proxy_epochs=args.proxy_epochs,
            crossover_prob=args.crossover_p,
            mutation_prob=args.mutation_p,
            tournament_k=args.tournament_k,
            n_elites=args.elites,
            use_parallel_gpus=args.parallel,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            smoke=args.smoke,
        )

        print("\nBest architecture found:")
        for k, v in result["best_arch"].items():
            print(f"  {k:14s}: {v}")
        print(f"\nBest proxy accuracy : {result['best_fitness']:.4f}")

        # Automatically plot convergence if log exists
        try:
            from utils.visualiser import plot_convergence, plot_architecture
            plot_convergence(result["log_path"])
            plot_architecture(result["best_arch"])
        except Exception as e:
            print(f"[Visualiser] Could not generate plots: {e}")

    # ------------------------------------------------------------------
    elif args.mode == "random-search":
        from models.baselines.random_nas import random_search
        chrom, fit, arch = random_search(
            n_evaluations=args.n_eval,
            device=device,
            proxy_epochs=args.proxy_epochs,
            smoke=args.smoke,
        )
        print(f"\nRandom Search best fitness: {fit:.4f}")
        print("Best arch:", arch)

    # ------------------------------------------------------------------
    elif args.mode == "train-best":
        import json
        from training.trainer import full_train

        if args.best_json is None:
            # Auto-find most recent best_*.json
            d = args.save_dir
            jsons = sorted(
                [f for f in os.listdir(d) if f.startswith("best_") and f.endswith(".json")],
                reverse=True,
            )
            if not jsons:
                print(f"[Error] No best_*.json found in {d}. Run --mode nas first.")
                sys.exit(1)
            json_path = os.path.join(d, jsons[0])
        else:
            json_path = args.best_json

        with open(json_path) as f:
            data = json.load(f)

        print(f"Loading best chromosome from: {json_path}")
        full_train(
            chromosome=data["best_chromosome"],
            device=device,
            epochs=args.full_epochs,
            save_dir=args.save_dir,
            run_id=data.get("run_id", "best"),
        )

    # ------------------------------------------------------------------
    elif args.mode == "train-resnet":
        import time
        from models.baselines.resnet import build_baseline_resnet
        from training.trainer import full_train
        from data.cifar import get_full_loaders
        import torch.optim as optim
        import torch.nn as nn
        from training.evaluator import validate
        
        model = build_baseline_resnet().to(device)
        train_loader, val_loader, test_loader = get_full_loaders(batch_size=128)
        
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.full_epochs)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Training Baseline ResNet on {device} for {args.full_epochs} epochs...")
        best_val = 0.0
        
        for ep in range(args.full_epochs):
            t0 = time.time()
            model.train()
            for x, y in train_loader:
                out = model(x.to(device))
                loss = criterion(out, y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            val_acc = validate(model, val_loader, device)
            print(f"  Epoch {ep+1}: val={val_acc:.4f} ({time.time()-t0:.1f}s)")
            if val_acc > best_val:
                best_val = val_acc
                
        test_acc = validate(model, test_loader, device)
        print(f"\nResNet Baseline Final Test Accuracy: {test_acc:.4f}")

    # ------------------------------------------------------------------
    elif args.mode == "plot":
        from utils.visualiser import plot_convergence
        log = args.log_dir
        plot_convergence(log, show=False)
        print("Done.")


if __name__ == "__main__":
    main()
