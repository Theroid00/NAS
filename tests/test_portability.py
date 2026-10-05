import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from utils.records import read_record, write_record


class PortabilityTests(unittest.TestCase):
    def test_record_relative_paths_and_explicit_legacy_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "record.json"
            value = {"checkpoint_path": str(root / "weights.pt"), "runs": [{"winner_path": str(root / "best.json")}]}
            write_record(path, value)
            stored = json.loads(path.read_text())
            self.assertEqual(stored["checkpoint_path"], "weights.pt")
            self.assertEqual(read_record(path), value)
            self.assertNotIn("path_format", value)
            path.write_text(json.dumps({"checkpoint_path": "C:\\old\\weights.pt"}))
            self.assertEqual(read_record(path, "C:\\old", root)["checkpoint_path"], str(root / "weights.pt"))
            with self.assertRaisesRegex(ValueError, "outside"):
                read_record(path, "C:\\other", root)

    def test_real_experiment_moves_without_original_directory(self):
        from run_comparison import compare, resume_comparison
        from relocate_experiment import relocate
        from reporting.comparison import build_report
        from training.trainer import train_winner
        from serving.artifact import export_artifact, Predictor
        from make_example import make_example
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compare(["ga", "aging", "random"], [1], budget=3, population=2,
                    proxy_epochs=1, full_epochs=1, out_dir=root / "original")
            source = next((root / "original").iterdir())
            destination = relocate(source, root / "moved")
            shutil.rmtree(root / "original")  # Owned temporary test fixture only.
            path = destination / "comparison.json"
            with patch("run_comparison.run_nas", side_effect=AssertionError("Reran search")), patch(
                    "training.trainer.full_train", side_effect=AssertionError("Retrained result")):
                recovered = resume_comparison(path)
            self.assertEqual(len(build_report(path)["runs"]), 3)
            row = recovered["runs"][0]
            full = train_winner(row["winner_path"], epochs=1, evaluate_test=False, save_dir=destination / "retrain")
            example = destination / "example.json"
            make_example(full["results_path"], example)
            export_artifact(full["results_path"], destination / "artifact", example)
            prediction = Predictor(destination / "artifact").predict(json.loads(example.read_text())["features"])
            self.assertEqual(len(prediction["predictions"]), 1)
            self.assertIsNone(full["test_accuracy"])
            with self.assertRaises(ValueError):
                relocate(destination, destination)

    def test_missing_and_unknown_record_formats_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            with self.assertRaises(FileNotFoundError):
                read_record(path)
            path.write_text('{"path_format": 99}')
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                read_record(path)
