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
    contestants = random.choices(range(len(population)), k=k)
    winner = max(contestants, key=lambda idx: fitnesses[idx])
    return list(population[winner])  # return a copy


def select_parents(
    population: List[List[int]],
    fitnesses: List[float],
    n: int,
    k: int = 3,
) -> List[List[int]]:
    """Select `n` parents via repeated tournament selection."""
    return [tournament_select(population, fitnesses, k) for _ in range(n)]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def single_point_crossover(
    parent_a: List[int],
    parent_b: List[int],
    prob: float = 0.8,
) -> Tuple[List[int], List[int]]:
    """
    Single-point crossover.

    Returns two offspring. With probability (1-prob) the parents are returned
    unchanged (no crossover occurs).
    """
    if random.random() > prob or NUM_GENES <= 1:
        return list(parent_a), list(parent_b)

    point = random.randint(1, NUM_GENES - 1)
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(chromosome: List[int], prob: float = 0.1) -> List[int]:
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
    for i in range(NUM_GENES):
        if random.random() < prob:
            mutant[i] = random.randint(0, len(GENE_VALUES[i]) - 1)
    return mutant
