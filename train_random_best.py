import torch
from training.trainer import full_train

# The best chromosome found by random search:
# Best arch: {'num_blocks': 5, 'filters_1': 128, 'filters_2': 128, 'filters_3': 128, 'filters_4': 256, 'filters_5': 512, 'kernel_size': 3, 'activation': 'leaky_relu', 'dropout': 0.0, 'batch_norm': True, 'fc_hidden': 512, 'pooling': 'max', 'use_residual': True}
chrom = [3, 2, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0]

print(f"Training best random architecture: {chrom}")
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    results = full_train(
        chromosome=chrom,
        device=device,
        epochs=50,
        save_dir="experiments/best_architectures",
        run_id="random_baseline",
    )
    print(f"\nRandom Search Final Test Accuracy : {results['test_accuracy']:.4f}")
