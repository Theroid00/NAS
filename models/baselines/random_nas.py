"""Evaluation-matched random search using the common trial protocol."""
import random
from ga.chromosome import random_chromosome
from utils.search_runtime import SearchSession, validate_budget


def run_random_search(n_evaluations=300, device="cpu", proxy_epochs=20, smoke=False,
                      seed=42, split_seed=42, proxy_size=None, max_params=None,
                      log_dir="experiments/tabular/generation_logs",
                      save_dir="experiments/tabular/best_architectures", evaluator=None, dataset="breast_cancer_wisconsin", validation_size=None, resume=None):
    validate_budget(n_evaluations, proxy_epochs)
    from data.specs import search_sizes
    proxy_size, validation_size = search_sizes(dataset, proxy_size, validation_size)
    if max_params is not None and max_params < 1:
        raise ValueError("Invalid proxy size or parameter limit")
    config = dict(evaluation_budget=n_evaluations, proxy_epochs=proxy_epochs, device=device,
                  seed=seed, split_seed=split_seed, proxy_size=proxy_size, validation_size=validation_size, dataset=dataset, max_params=max_params,
                  injected_evaluator=evaluator is not None)
    rng = random.Random(seed)
    with SearchSession("random", config, log_dir, save_dir, smoke, evaluator, resume) as session:
        # Evaluate each candidate sequentially on the selected device.
        for start in range(0, n_evaluations, 20):
            batch = [random_chromosome(rng) for _ in range(min(20, n_evaluations - start))]
            session.evaluate(batch)
        return session.finish()


def random_search(n_evaluations=300, device="cpu", proxy_epochs=20, smoke=False, **kwargs):
    """Preserve the historical tuple interface while persisting the winning record."""
    result = run_random_search(n_evaluations, device, proxy_epochs, smoke, **kwargs)
    return result["best_chromosome"], result["best_fitness"], result["best_arch"]
