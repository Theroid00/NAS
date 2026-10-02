"""Evaluation-matched random search using the common trial protocol."""
import random
from ga.chromosome import random_chromosome
from utils.search_runtime import SearchSession, validate_budget


def run_random_search(n_evaluations=300, device="cpu", proxy_epochs=5, smoke=False,
                      seed=42, split_seed=42, proxy_size=10000, max_params=None,
                      use_parallel_gpus=False, log_dir="experiments/generation_logs",
                      save_dir="experiments/best_architectures", evaluator=None):
    validate_budget(n_evaluations, proxy_epochs)
    if not 1 <= proxy_size <= 48000 or max_params is not None and max_params < 1:
        raise ValueError("Invalid proxy size or parameter limit")
    config = dict(evaluation_budget=n_evaluations, proxy_epochs=proxy_epochs, device=device,
                  seed=seed, split_seed=split_seed, proxy_size=proxy_size, max_params=max_params,
                  parallel=use_parallel_gpus, injected_evaluator=evaluator is not None)
    rng = random.Random(seed)
    with SearchSession("random", config, log_dir, save_dir, smoke, evaluator) as session:
        # Batches enable the same worker/device allocation used by the other searchers.
        for start in range(0, n_evaluations, 20):
            batch = [random_chromosome(rng) for _ in range(min(20, n_evaluations - start))]
            session.evaluate(batch)
        return session.finish()


def random_search(n_evaluations=300, device="cpu", proxy_epochs=5, smoke=False, **kwargs):
    """Preserve the historical tuple interface while persisting the winning record."""
    result = run_random_search(n_evaluations, device, proxy_epochs, smoke, **kwargs)
    return result["best_chromosome"], result["best_fitness"], result["best_arch"]
