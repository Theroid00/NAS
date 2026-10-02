from ga.chromosome import NUM_GENES, SCHEMA_VERSION
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from ga.aging import run_aging_evolution
from ga.operators import mutate_active
from ga.chromosome import decode
from run_comparison import compare


class AgingTests(unittest.TestCase):
    def test_oldest_winner_dies_but_archive_keeps_it(self):
        scores = iter([0.9, 0.2, 0.1, 0.3])
        parents = []
        def mutation(parent, rng):
            parents.append(parent[:])
            return [1] * NUM_GENES
        with tempfile.TemporaryDirectory() as directory, patch("ga.aging.mutate_active", side_effect=mutation):
            result = run_aging_evolution(n_evaluations=4, population_size=2, tournament_k=100,
                                        log_dir=directory, save_dir=directory, evaluator=lambda *args: next(scores))
            records = [json.loads(line) for line in Path(result["trial_path"]).read_text().splitlines()]
            self.assertEqual(result["best_trial_id"], 1)
            self.assertEqual(parents[0], records[0]["chromosome"])
            self.assertEqual(parents[1], records[1]["chromosome"])
            self.assertEqual(result["evaluation_count"], 4)

    def test_mutation_changes_one_expressed_choice(self):
        parent = [0] * NUM_GENES
        for seed in range(100):
            child = mutate_active(parent, random.Random(seed))
            changes = [i for i, (a, b) in enumerate(zip(parent, child)) if a != b]
            self.assertEqual(len(changes), 1)
            self.assertNotIn(changes[0], [2, 3, 4, 8])
            self.assertNotEqual(decode(parent), decode(child))

    def test_comparison_obeys_same_budget_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            results = [compare(["ga", "aging", "random"], [1, 2], budget=7, population=3,
                               smoke=True, out_dir=directory) for _ in range(2)]
            for manifest in results:
                self.assertEqual(manifest["status"], "completed")
                self.assertTrue(all(row["evaluation_count"] == 7 for row in manifest["runs"]))
            self.assertEqual([r["best_proxy_fitness"] for r in results[0]["runs"]],
                             [r["best_proxy_fitness"] for r in results[1]["runs"]])
