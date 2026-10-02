"""
ga/engine.py
============
Main GA loop for Neural Architecture Search.

run_nas() coordinates selection, crossover, mutation, elitism, and
delegates fitness evaluation to training/evaluator.py.
"""

import time
import json
import os
from typing import List, Optional

from ga.chromosome import decode, SEARCH_SPACE, SCHEMA_VERSION
from ga.operators import select_parents, single_point_crossover, mutate
from ga.population import init_population, get_top_k, get_best


def _evaluate_sequential(population, device, proxy_epochs):
    """Evaluate all chromosomes one at a time on a single device."""
    from training.evaluator import evaluate_architecture
    from tqdm import tqdm
    fitnesses = []
    for chrom in tqdm(population, desc="Evaluating"):
        acc = evaluate_architecture(chrom, device, proxy_epochs)
        fitnesses.append(acc)
    return fitnesses


def _evaluate_parallel(population, proxy_epochs):
    """Evaluate population split across available GPUs."""
    import concurrent.futures
    import torch
    from training.evaluator import evaluate_architecture

    mid = len(population) // 2
    gpu0_pop = population[:mid]
    gpu1_pop = population[mid:]

    dev0 = "cuda:0"
    dev1 = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"

    def eval_batch(batch, device):
        return [evaluate_architecture(c, device, proxy_epochs) for c in batch]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(eval_batch, gpu0_pop, dev0)
        f1 = executor.submit(eval_batch, gpu1_pop, dev1)
        return f0.result() + f1.result()


def run_nas(
    device: str = "cpu",
    population_size: int = 20,
    n_generations: int = 15,
    proxy_epochs: int = 5,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.1,
    tournament_k: int = 5,
    n_elites: int = 2,
    use_parallel_gpus: bool = False,
    log_dir: str = "experiments/generation_logs",
    save_dir: str = "experiments/best_architectures",
    smoke: bool = False,
) -> dict:
    """
    Run the full genetic algorithm NAS loop.

    Args:
        device:            PyTorch device string ('cpu', 'cuda', 'cuda:0')
        population_size:   Number of architectures per generation
        n_generations:     Number of GA generations to run
        proxy_epochs:      Epochs for proxy training each architecture
        crossover_prob:    Probability of crossover between two parents
        mutation_prob:     Per-gene mutation probability
        tournament_k:      Tournament size for selection
        n_elites:          Number of elite chromosomes carried forward
        use_parallel_gpus: If True, split evaluation across cuda:0 and cuda:1
        log_dir:           Directory for generation CSV logs
        save_dir:          Directory to save best chromosome JSON
        smoke:             If True, skip real training (return random fitness)

    Returns:
        dict with keys: best_chromosome, best_fitness, best_arch, log_path
    """
    from utils.logger import GenerationLogger

    if population_size < 1 or n_generations < 1 or proxy_epochs < 1:
        raise ValueError("Population, generations, and proxy epochs must be positive")
    if not 0 <= n_elites <= population_size or tournament_k < 1:
        raise ValueError("Elite count must be within population size; tournament size must be positive")
    if not 0 <= crossover_prob <= 1 or not 0 <= mutation_prob <= 1:
        raise ValueError("Operator probabilities must be between zero and one")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    logger = GenerationLogger(log_dir, run_id)

    print(f"\n{'='*60}")
    print(f"  NAS with Genetic Algorithm")
    print(f"  population={population_size}  generations={n_generations}")
    print(f"  proxy_epochs={proxy_epochs}  device={device}")
    print(f"  parallel_gpus={use_parallel_gpus}")
    print(f"{'='*60}\n")

    population = init_population(population_size)
    fitnesses = [0.0] * population_size  # placeholder for first gen display

    best_historical_fitness = -1.0
    best_historical_chromosome = None
    stagnation_counter = 0

    for gen in range(n_generations):
        gen_start = time.time()
        print(f"\n[Gen {gen+1:02d}/{n_generations}] Evaluating {population_size} architectures...")

        # ---- Fitness evaluation ----
        if smoke:
            import random
            fitnesses = [random.uniform(0.1, 0.9) for _ in population]
        elif use_parallel_gpus:
            fitnesses = _evaluate_parallel(population, proxy_epochs)
        else:
            fitnesses = _evaluate_sequential(population, device, proxy_epochs)

        # ---- Logging ----
        best_chrom, best_fit = get_best(population, fitnesses)
        if best_fit > best_historical_fitness:
            best_historical_chromosome = list(best_chrom)
        mean_fit = sum(fitnesses) / len(fitnesses)
        
        import math
        variance = sum((f - mean_fit) ** 2 for f in fitnesses) / len(fitnesses)
        std_fit = math.sqrt(variance)
        
        min_fit = min(fitnesses)
        elapsed = time.time() - gen_start
        
        # Adaptive Mutation Logic
        current_mutation_prob = mutation_prob
        if best_fit > best_historical_fitness:
            best_historical_fitness = best_fit
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if stagnation_counter >= 3:
            current_mutation_prob = min(0.3, mutation_prob * 2)
            print(f"  [!] Stagnation detected ({stagnation_counter} gens). Temporarily boosting mutation to {current_mutation_prob:.2f}")

        logger.log(
            gen=gen + 1,
            best_fitness=best_fit,
            mean_fitness=mean_fit,
            std_fitness=std_fit,
            min_fitness=min_fit,
            best_chromosome=best_chrom,
            elapsed=elapsed,
        )

        print(
            f"  best={best_fit:.4f}  mean={mean_fit:.4f}  "
            f"min={min_fit:.4f}  time={elapsed:.1f}s"
        )

        # ---- Build next generation ----
        if gen == n_generations - 1:
            break
        # 1. Elitism — carry top-n_elites forward unchanged
        elites = get_top_k(population, fitnesses, n_elites)

        # 2. Selection — pick parents for the remaining slots
        n_offspring = population_size - n_elites
        parents = select_parents(population, fitnesses, n=n_offspring, k=tournament_k)

        # 3. Crossover — pair up parents
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            c1, c2 = single_point_crossover(parents[i], parents[i + 1], prob=crossover_prob)
            offspring.extend([c1, c2])
        # Handle odd n_offspring
        if len(offspring) < n_offspring:
            offspring.append(list(parents[-1]))

        # 4. Mutation
        offspring = [mutate(c, prob=current_mutation_prob) for c in offspring[:n_offspring]]

        # 5. New population
        population = elites + offspring

    # ---- Final best ----
    best_chrom = best_historical_chromosome
    best_fit = best_historical_fitness
    best_arch = decode(best_chrom)

    # Save best architecture
    result = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "search_space": SEARCH_SPACE,
        "smoke": smoke,
        "winner_policy": "best_observed_validation_accuracy",
        "best_chromosome": best_chrom,
        "best_fitness": best_fit,
        "best_arch": best_arch,
        "hyperparams": {
            "population_size": population_size,
            "n_generations": n_generations,
            "proxy_epochs": proxy_epochs,
            "crossover_prob": crossover_prob,
            "mutation_prob": mutation_prob,
            "tournament_k": tournament_k,
            "n_elites": n_elites,
            "use_parallel_gpus": use_parallel_gpus,
            "device": device,
            "adaptive_mutation": {"patience": 3, "multiplier": 2, "cap": 0.3},
        },
    }
    save_path = os.path.join(save_dir, f"best_{run_id}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    log_path = logger.close()

    print(f"\n{'='*60}")
    print(f"  NAS Complete!")
    print(f"  Best validation accuracy : {best_fit:.4f}")
    print(f"  Best architecture saved  : {save_path}")
    print(f"  Generation log saved     : {log_path}")
    print(f"{'='*60}\n")

    result["log_path"] = log_path
    return result
