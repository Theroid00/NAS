import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ga.chromosome import decode, GENE_VALUES
from ga.engine import run_nas


class SearchCorrectnessTests(unittest.TestCase):
    def test_winner_is_an_evaluated_chromosome_and_historical_best(self):
        population = [[0] * 13, [1] * 13, [2 if len(v) > 2 else 0 for v in GENE_VALUES]]
        with tempfile.TemporaryDirectory() as directory:
            scores = iter([0.1, 0.9, 0.2, 0.4, 0.3, 0.2])
            with patch("ga.engine.init_population", return_value=population), patch(
                "ga.engine.select_parents", return_value=[population[0], population[0]]):
                result = run_nas(population_size=3, n_generations=2, n_elites=1,
                                 mutation_prob=0, crossover_prob=0,
                                 log_dir=directory, save_dir=directory, evaluator=lambda *args: next(scores))
            self.assertEqual(result["best_chromosome"], population[1])
            self.assertEqual(result["best_fitness"], 0.9)
            saved = json.loads(next(Path(directory).glob("best_*.json")).read_text())
            self.assertEqual(saved["best_arch"], decode(population[1]))
            self.assertEqual(saved["schema_version"], 2)
            records = [json.loads(line) for line in Path(result["trial_path"]).read_text().splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual(records[1]["chromosome"], result["best_chromosome"])

    def test_search_rng_is_independent_of_evaluator_randomness(self):
        import random
        with tempfile.TemporaryDirectory() as directory:
            def evaluator(*args):
                random.seed(args[3])
                random.random()
                return 0.5
            results = [run_nas(population_size=3, n_generations=2, n_elites=1, seed=7,
                               log_dir=directory, save_dir=directory, evaluator=evaluator) for _ in range(2)]
            chromosomes = [[json.loads(line)["chromosome"] for line in Path(r["trial_path"]).read_text().splitlines()]
                           for r in results]
            self.assertEqual(chromosomes[0], chromosomes[1])

    def test_fatal_trial_error_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "broken loader"):
                run_nas(population_size=2, n_generations=1, log_dir=directory, save_dir=directory,
                        evaluator=lambda *args: (_ for _ in ()).throw(RuntimeError("broken loader")))
            metadata = json.loads(next(Path(directory).glob("metadata_*.json")).read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertFalse(list(Path(directory).glob("best_*.json")))

    def test_bad_budgets_fail_before_creating_outputs(self):
        for kwargs in ({"population_size": 0}, {"n_generations": 0},
                       {"n_elites": 21}, {"mutation_prob": -1}, {"tournament_k": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_nas(**kwargs)

    def test_chromosome_validation(self):
        for chromosome in ([0] * 10, [-1] * 13, [True] * 13):
            with self.subTest(chromosome=chromosome), self.assertRaises(ValueError):
                decode(chromosome)


if __name__ == "__main__":
    unittest.main()
