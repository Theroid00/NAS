"""
ga/operators.py
===============
Genetic operators: selection, crossover, mutation.
"""

import random
from typing import List, Tuple
from ga.chromosome import GENE_VALUES, NUM_GENES


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def tournament_select(
    population: List[List[int]],
    fitnesses: List[float],
    k: int = 3,
    rng=None,
) -> List[int]:
    """
    Tournament selection: randomly choose k individuals and return the best.

    Args:
        population: list of chromosomes
        fitnesses:  corresponding fitness values (higher = better)
        k:          tournament size

    Returns:
        A single chromosome (winner).
    """
    rng = rng or random
    contestants = rng.choices(range(len(population)), k=k)
    winner = max(contestants, key=lambda idx: fitnesses[idx])
    return list(population[winner])  # return a copy


def select_parents(
    population: List[List[int]],
    fitnesses: List[float],
    n: int,
    k: int = 3,
    rng=None,
) -> List[List[int]]:
    """Select `n` parents via repeated tournament selection."""
    return [tournament_select(population, fitnesses, k, rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def single_point_crossover(
    parent_a: List[int],
    parent_b: List[int],
    prob: float = 0.8,
    rng=None,
) -> Tuple[List[int], List[int]]:
    """
    Single-point crossover.

    Returns two offspring. With probability (1-prob) the parents are returned
    unchanged (no crossover occurs).
    """
    rng = rng or random
    if rng.random() >= prob or NUM_GENES <= 1:
        return list(parent_a), list(parent_b)

    point = rng.randint(1, NUM_GENES - 1)
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(chromosome: List[int], prob: float = 0.1, rng=None) -> List[int]:
    """
    Per-gene mutation: each gene is independently replaced with a random
    valid value with probability `prob`.

    Args:
        chromosome: integer-index chromosome to mutate
        prob:       per-gene mutation probability

    Returns:
        Mutated chromosome (new list, original unchanged).
    """
    mutant = list(chromosome)
    rng = rng or random
    for i in range(NUM_GENES):
        if rng.random() < prob:
            mutant[i] = rng.randint(0, len(GENE_VALUES[i]) - 1)
    return mutant


def mutate_active(chromosome, rng=None):
    """Change exactly one expressed MLP choice."""
    from ga.chromosome import decode, GENE_NAMES
    rng = rng or random
    arch = decode(chromosome)
    active = [i for i, name in enumerate(GENE_NAMES)
              if not name.startswith("width_") or int(name.split("_")[1]) <= arch["num_layers"]]
    # Both supported input widths (30 and 54) differ from every hidden-width choice.
    widths = [arch[f"width_{j}"] for j in range(1, arch["num_layers"] + 1)]
    if not any(a == b for a, b in zip(widths, widths[1:])):
        active.remove(GENE_NAMES.index("use_residual"))
    i = rng.choice(active)
    choices = [v for v in range(len(GENE_VALUES[i])) if v != chromosome[i]]
    child = list(chromosome)
    child[i] = rng.choice(choices)
    return child
