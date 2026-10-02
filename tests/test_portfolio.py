from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PortfolioTests(unittest.TestCase):
    def test_end_to_end_export_test_evaluation_and_repeated_finish(self):
        from run_comparison import compare
        from run_portfolio import finalize_comparison
        from training.trainer import evaluate_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compare(["random"], [1], budget=3, population=2, proxy_epochs=1, full_epochs=1,
                    device="cpu", out_dir=root / "runs")
            comparison = next(root.glob("runs/*/comparison.json"))
            with patch("training.trainer.evaluate_checkpoint", wraps=evaluate_checkpoint) as evaluate:
                report = finalize_comparison(comparison, root / "report.json", root / "artifacts")
                repeated = finalize_comparison(comparison, root / "report.json", root / "artifacts")
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(report["selected_model"]["model_id"], repeated["selected_model"]["model_id"])
            self.assertIsNotNone(report["selected_model"]["test_metrics"])
            self.assertTrue((Path(report["artifact_dir"]) / "example.json").exists())
            self.assertEqual(len(report["inference_benchmark"]["measurements"]), 2)

    def test_smoke_results_cannot_be_packaged_as_measured_models(self):
        from run_comparison import compare
        from run_portfolio import finalize_comparison
        with tempfile.TemporaryDirectory() as directory:
            compare(["random"], [1], budget=3, population=2, smoke=True, out_dir=directory)
            with self.assertRaisesRegex(ValueError, "real comparison"):
                finalize_comparison(next(Path(directory).glob("*/comparison.json")))
