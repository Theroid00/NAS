import unittest
from benchmark_search import architecture_string, search


class BenchmarkTests(unittest.TestCase):
    def test_official_encoding_and_repeatable_budgets(self):
        self.assertEqual(architecture_string([0] * 6), "|none~0|+|none~0|none~1|+|none~0|none~1|none~2|")
        evaluate = lambda genes: {"fitness": sum(genes) / 24, "training_seconds": 2.0}
        for method in ("ga", "aging", "random"):
            a = search(method, evaluate, 42, budget=17, population_size=4)
            b = search(method, evaluate, 42, budget=17, population_size=4)
            self.assertEqual(a, b)
            self.assertEqual(a["evaluation_count"], 17)
            self.assertEqual(a["simulated_training_seconds"], 34)
            self.assertEqual(a["winner"]["fitness"], max(r["fitness"] for r in a["history"]))
