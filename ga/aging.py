"""Mutation-only tournament evolution with FIFO population replacement."""
from collections import deque
import random
import statistics
import time
from ga.chromosome import random_chromosome
from ga.operators import mutate_active
from utils.logger import GenerationLogger
from utils.search_runtime import SearchSession, validate_budget


def run_aging_evolution(n_evaluations=300, population_size=20, tournament_k=5,
                       proxy_epochs=20, device="cpu", smoke=False, seed=42, split_seed=42,
                       proxy_size=None, max_params=None,
                       log_dir="experiments/tabular/generation_logs", save_dir="experiments/tabular/best_architectures",
                       evaluator=None, dataset="breast_cancer_wisconsin", validation_size=None, resume=None):
    validate_budget(n_evaluations, proxy_epochs, population_size, tournament_k)
    from data.specs import search_sizes
    proxy_size, validation_size = search_sizes(dataset, proxy_size, validation_size)
    if max_params is not None and max_params < 1:
        raise ValueError("Invalid proxy size or parameter limit")
    config = dict(evaluation_budget=n_evaluations, population_size=population_size,
                  tournament_k=tournament_k, proxy_epochs=proxy_epochs, device=device,
                  seed=seed, split_seed=split_seed, proxy_size=proxy_size, validation_size=validation_size, dataset=dataset, max_params=max_params,
                  injected_evaluator=evaluator is not None,
                  mutation_policy="one_active_choice", replacement_policy="oldest_first")
    rng = random.Random(seed)
    population = deque()
    with SearchSession("aging", config, log_dir, save_dir, smoke, evaluator, resume) as session:
        logger = GenerationLogger(session.log_dir, session.run_id, resume=bool(resume))
        try:
            initial = [random_chromosome(rng) for _ in range(population_size)]
            t0 = time.perf_counter()
            initial_fitness = session.evaluate(initial, 1)
            population.extend(zip(initial, initial_fitness))
            cycle = 1
            while True:
                best = max(population, key=lambda item: item[1])
                scores = [fitness for _, fitness in population]
                logger.log(cycle, best[1], statistics.mean(scores), statistics.pstdev(scores),
                           min(scores), best[0], time.perf_counter() - t0)
                if session.count >= n_evaluations:
                    break
                cycle += 1
                parent = max(rng.choices(list(population), k=tournament_k), key=lambda item: item[1])[0]
                child = mutate_active(parent, rng)
                t0 = time.perf_counter()
                score = session.evaluate([child], cycle)[0]
                population.append((child, score))
                population.popleft()
            return session.finish(logger.path)
        finally:
            logger.close()
