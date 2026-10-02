from pathlib import Path
import tempfile
import unittest
from run_comparison import compare
from reporting.comparison import build_report, save_report


class ReportingTests(unittest.TestCase):
    def test_report_preserves_measured_progress_and_marks_smoke_scores(self):
        with tempfile.TemporaryDirectory() as root:
            compare(["ga", "aging", "random"], [1, 2], budget=7, population=3, smoke=True, out_dir=root)
            path = next(Path(root).glob("*/comparison.json"))
            report = build_report(path)
            self.assertTrue(report["smoke"])
            self.assertEqual(len(report["runs"]), 6)
            self.assertIsNone(report["selected_model"])
            for run in report["runs"]:
                scores = [p["best_accuracy"] for p in run["progress"]]
                self.assertEqual(scores, sorted(scores))
                self.assertEqual(scores[-1], run["accuracy"])
                self.assertIsNone(run["metrics"])
            for method in ("ga", "aging", "random"):
                self.assertEqual(report["summary"][method]["proxy_accuracy"]["n"], 2)
            self.assertEqual(save_report(path, Path(root) / "report.json"), report)

    def test_deployment_selection_uses_validation_and_never_test_scores(self):
        from unittest.mock import patch
        from ga.chromosome import NUM_GENES
        with tempfile.TemporaryDirectory() as root:
            compare(["random"], [1, 2], budget=3, population=2, proxy_epochs=1, full_epochs=1, out_dir=root)
            report = build_report(next(Path(root).glob("*/comparison.json")))
            self.assertIsNotNone(report["selected_model"])
            self.assertEqual(report["selected_model"]["accuracy"],
                             max(run["full_training"][0]["accuracy"] for run in report["runs"]))
            self.assertIsNone(report["selected_model"]["test_metrics"])
