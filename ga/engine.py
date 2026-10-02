"""Generational NAS with paired trial records and an evaluated winner archive."""
import random
import statistics
import time
from ga.operators import select_parents, single_point_crossover, mutate
from ga.population import init_population, get_top_k, get_best
from utils.logger import GenerationLogger
from utils.search_runtime import SearchSession, validate_budget


def run_nas(device="cpu", population_size=20, n_generations=15, proxy_epochs=20,
            crossover_prob=0.8, mutation_prob=0.1, tournament_k=5, n_elites=2,
            log_dir="experiments/tabular/generation_logs",
            save_dir="experiments/tabular/best_architectures", smoke=False, seed=42,
            split_seed=42, proxy_size=None, max_params=None, evaluator=None,
            evaluation_budget=None, dataset="breast_cancer_wisconsin", validation_size=None, resume=None):
    budget = evaluation_budget if evaluation_budget is not None else population_size * n_generations
    validate_budget(budget, proxy_epochs, population_size, tournament_k)
    if n_generations < 1 or not 0 <= n_elites <= population_size:
        raise ValueError("Generations must be positive and elites within the population size")
    if not 0 <= crossover_prob <= 1 or not 0 <= mutation_prob <= 1:
        raise ValueError("Operator probabilities must be between zero and one")
    from data.specs import search_sizes
    proxy_size, validation_size = search_sizes(dataset, proxy_size, validation_size)
    if max_params is not None and max_params < 1:
        raise ValueError("Parameter limit must be positive")
    config = dict(device=device, population_size=population_size, n_generations=(budget + population_size - 1) // population_size,
                  proxy_epochs=proxy_epochs, crossover_prob=crossover_prob, mutation_prob=mutation_prob,
                  tournament_k=tournament_k, n_elites=n_elites,
                  seed=seed, split_seed=split_seed, proxy_size=proxy_size, validation_size=validation_size, dataset=dataset, max_params=max_params,
                  evaluation_budget=budget, injected_evaluator=evaluator is not None,
                  adaptive_mutation={"patience": 3, "multiplier": 2, "cap": 0.3})
    rng = random.Random(seed)
    population = init_population(population_size, rng)
    historical_best, stagnant = -1.0, 0
    with SearchSession("ga", config, log_dir, save_dir, smoke, evaluator, resume) as session:
        logger = GenerationLogger(session.log_dir, session.run_id, resume=bool(resume))
        try:
            generation = 0
            while session.count < budget:
                generation += 1
                t0 = time.perf_counter()
                evaluated = population[:budget - session.count]
                fitnesses = session.evaluate(evaluated, generation)
                chrom, best = get_best(evaluated, fitnesses)
                if best > historical_best:
                    historical_best, stagnant = best, 0
                else:
                    stagnant += 1
                mutation = min(0.3, mutation_prob * 2) if stagnant >= 3 else mutation_prob
                logger.log(generation, best, statistics.mean(fitnesses), statistics.pstdev(fitnesses),
                           min(fitnesses), chrom, time.perf_counter() - t0)
                print(f"Generation {generation}: best={best:.4f}, evaluations={session.count}/{budget}")
                if session.count == budget:
                    break
                elites = get_top_k(population, fitnesses, n_elites)
                parents = select_parents(population, fitnesses, population_size - n_elites, tournament_k, rng)
                offspring = []
                for i in range(0, len(parents) - 1, 2):
                    offspring.extend(single_point_crossover(parents[i], parents[i + 1], crossover_prob, rng))
                if len(offspring) < len(parents):
                    offspring.append(parents[-1][:])
                population = elites + [mutate(c, mutation, rng) for c in offspring]
            result = session.finish(logger.path)
        finally:
            logger.close()
    return result
