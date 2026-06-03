"""
models/baselines/random_nas.py
==============================
Random search NAS baseline.

Evaluates a fixed budget of random architectures and returns the best one.
This serves as a compute-matched baseline against the GA — both use the same
number of total evaluations (300 = 15 gen × 20 pop).
"""

import random
from typing import Dict, Any, Tuple


def random_search(
    n_evaluations: int = 300,
    device: str = "cpu",
    proxy_epochs: int = 5,
    smoke: bool = False,
) -> Tuple[list, float, Dict[str, Any]]:
    """
    Evaluate `n_evaluations` random architectures and return the best.

    Args:
        n_evaluations: Total random architectures to try
        device:        PyTorch device
        proxy_epochs:  Proxy training epochs per architecture
        smoke:         If True, return random fitness without training

    Returns:
        (best_chromosome, best_fitness, best_arch)
    """
    from ga.chromosome import random_chromosome, decode
    from training.evaluator import evaluate_architecture
    from tqdm import tqdm

    best_chrom = None
    best_fit = -1.0

    print(f"\nRandom Search: evaluating {n_evaluations} architectures...")
    for i in tqdm(range(n_evaluations), desc="Random Search"):
        chrom = random_chromosome()

        if smoke:
            fit = random.uniform(0.1, 0.9)
        else:
            fit = evaluate_architecture(chrom, device, proxy_epochs)

        if fit > best_fit:
            best_fit = fit
            best_chrom = chrom

    best_arch = decode(best_chrom)
    print(f"Random Search best fitness: {best_fit:.4f}")
    return best_chrom, best_fit, best_arch
