"""A failed or killed export never publishes a partially ready artifact."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ga.chromosome import NUM_GENES
from serving.artifact import export_artifact, Predictor
from training.trainer import full_train
from utils.persistence import atomic_json


class ExportRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.result = full_train([0] * NUM_GENES, epochs=1, save_dir=self.root / "training", evaluate_test=False)
        from make_example import make_example
        self.example = self.root / "request.json"
        make_example(self.result["results_path"], self.example)

    def tearDown(self):
        self.temporary.cleanup()

    def test_failure_cleans_staging_and_same_destination_can_be_retried(self):
        destination = self.root / "artifact"
        def interrupted(path, value):
            if Path(path).name == "example.json":
                self.assertFalse(destination.exists())
                raise KeyboardInterrupt()
            return atomic_json(path, value)
        with patch("serving.artifact.atomic_json", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                export_artifact(self.result["results_path"], destination, self.example)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".artifact.export-*/")), [])
        export_artifact(self.result["results_path"], destination, self.example)
        self.assertTrue(Predictor(destination).manifest["example_available"])

    def test_invalid_example_cannot_publish_and_existing_artifact_is_preserved(self):
        destination = self.root / "artifact"
        self.example.write_text(json.dumps({"features": [[0] * 29]}))
        with self.assertRaises(ValueError):
            export_artifact(self.result["results_path"], destination, self.example)
        self.assertFalse(destination.exists())
        export_artifact(self.result["results_path"], destination)
        before = (destination / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            export_artifact(self.result["results_path"], destination)
        self.assertEqual((destination / "manifest.json").read_bytes(), before)

    def test_missing_or_corrupted_promised_example_fails_startup(self):
        destination = self.root / "artifact"
        export_artifact(self.result["results_path"], destination, self.example)
        example = destination / "example.json"
        original = example.read_bytes()
        example.write_text(json.dumps({"features": [[1] * 30]}))
        with self.assertRaisesRegex(ValueError, "checksum"):
            Predictor(destination)
        example.write_bytes(original)
        example.unlink()
        with self.assertRaisesRegex(ValueError, "example.json is missing"):
            Predictor(destination)

    def test_process_death_leaves_only_staging_and_os_releases_export_lock(self):
        destination = self.root / "artifact"
        script = '''
import os, sys
from unittest.mock import patch
from serving.artifact import export_artifact
from utils.persistence import atomic_json
def killed(path, value):
    atomic_json(path, value)
    if path.name == "manifest.json": os._exit(73)
with patch("serving.artifact.atomic_json", side_effect=killed):
    export_artifact(sys.argv[1], sys.argv[2], sys.argv[3])
'''
        result = subprocess.run([sys.executable, "-c", script, self.result["results_path"],
                                 str(destination), str(self.example)],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 73, result.stderr.decode())
        self.assertFalse(destination.exists())
        self.assertEqual(len(list(self.root.glob(".artifact.export-*/"))), 1)
        export_artifact(self.result["results_path"], destination, self.example)
        self.assertTrue((destination / "model.pt").exists())
        self.assertEqual(Predictor(destination).manifest["model_id"], self.result["run_id"])
