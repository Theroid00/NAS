"""
ga/population.py
================
Population-level utilities: initialisation, ranking, top-k selection.
"""

from typing import List, Tuple
from ga.chromosome import random_chromosome


def init_population(size: int, rng=None) -> List[List[int]]:
    """Create an initial population of `size` random chromosomes."""
    return [random_chromosome(rng) for _ in range(size)]


def get_ranked(
    population: List[List[int]],
    fitnesses: List[float],
) -> List[Tuple[float, List[int]]]:
    """Return (fitness, chromosome) pairs sorted descending by fitness."""
    paired = list(zip(fitnesses, population))
    paired.sort(key=lambda x: x[0], reverse=True)
    return paired


def get_top_k(
    population: List[List[int]],
    fitnesses: List[float],
    k: int,
) -> List[List[int]]:
    """Return the top-k chromosomes by fitness."""
    ranked = get_ranked(population, fitnesses)
    return [chrom for _, chrom in ranked[:k]]


def get_best(
    population: List[List[int]],
    fitnesses: List[float],
) -> Tuple[List[int], float]:
    """Return (best_chromosome, best_fitness)."""
    best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    return population[best_idx], fitnesses[best_idx]
