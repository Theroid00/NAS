"""Full-train a compatible real winner or explicitly test a frozen checkpoint."""
import argparse
from utils.results import latest_winner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None, help="Use the saved winner's split unless explicitly overridden")
    p.add_argument("--validation-only", action="store_true")
    p.add_argument("--test-checkpoint", help="Full-training JSON to evaluate on test, without retraining")
    p.add_argument("--save-dir", default="experiments/tabular/best_architectures")
    args = p.parse_args()
    from training.trainer import train_winner, evaluate_checkpoint
    if args.test_checkpoint:
        result = evaluate_checkpoint(args.test_checkpoint, args.device)
    else:
        result = train_winner(args.json or latest_winner(args.save_dir), device=args.device,
                              epochs=args.epochs, seed=args.seed, split_seed=args.split_seed,
                              evaluate_test=not args.validation_only, save_dir=args.save_dir)
    print(f"Validation: {result['best_val_accuracy']:.4f}; test: {result['test_accuracy']}")
    print(result["results_path"])


if __name__ == "__main__":
    main()
