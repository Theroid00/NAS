"""Startup source snapshots detect changed, added, and removed implementation files."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.provenance import source_fingerprint


class ProvenanceTests(unittest.TestCase):
    def test_fingerprint_tracks_source_but_ignores_docs_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training").mkdir()
            module = root / "training" / "module.py"
            module.write_text("value = 1\n")
            first = source_fingerprint(root)
            self.assertEqual(first, source_fingerprint(root))
            (root / "README.md").write_text("changed documentation")
            self.assertEqual(first, source_fingerprint(root))
            module.write_text("value = 2\n")
            self.assertNotEqual(first, source_fingerprint(root))
            second = source_fingerprint(root)
            (root / "training" / "extra.py").write_text("value = 3\n")
            self.assertNotEqual(second, source_fingerprint(root))
            module.unlink()
            self.assertNotIn("training/module.py", source_fingerprint(root)["files"])

    def test_search_winner_and_metadata_record_startup_snapshot(self):
        from models.baselines.random_nas import run_random_search
        with tempfile.TemporaryDirectory() as directory:
            result = run_random_search(n_evaluations=2, proxy_epochs=1, smoke=True,
                                       log_dir=directory, save_dir=directory)
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(metadata["resume_version"], 3)
            self.assertEqual(result["source_fingerprint"], source_fingerprint())
            self.assertEqual(metadata["source_fingerprint"], result["source_fingerprint"])

    def test_changed_source_blocks_search_resume_before_rewriting_records(self):
        from models.baselines.random_nas import run_random_search
        from utils.search_runtime import resume_search
        with tempfile.TemporaryDirectory() as directory:
            result = run_random_search(n_evaluations=2, proxy_epochs=1, smoke=True,
                                       log_dir=directory, save_dir=directory)
            metadata = Path(result["metadata_path"])
            before = metadata.read_bytes()
            journal = Path(result["trial_path"]).read_bytes()
            changed = {**source_fingerprint(), "sha256": "changed"}
            with patch("utils.search_runtime.source_fingerprint", return_value=changed):
                with self.assertRaisesRegex(ValueError, "Source files changed"):
                    resume_search(metadata)
            self.assertEqual(metadata.read_bytes(), before)
            self.assertEqual(Path(result["trial_path"]).read_bytes(), journal)

    def test_changed_source_blocks_completed_comparison_resume(self):
        from run_comparison import compare, resume_comparison
        with tempfile.TemporaryDirectory() as directory:
            manifest = compare(["random"], [1], budget=2, population=2, smoke=True, out_dir=directory)
            path = next(Path(directory).glob("*/comparison.json"))
            self.assertEqual(manifest["resume_version"], 2)
            before = path.read_bytes()
            with patch("run_comparison.source_fingerprint", return_value={"changed": True}):
                with self.assertRaisesRegex(ValueError, "Source files changed"):
                    resume_comparison(path)
            self.assertEqual(path.read_bytes(), before)

    def test_mid_suite_source_change_stops_before_next_search(self):
        from run_comparison import compare
        original = source_fingerprint()
        with tempfile.TemporaryDirectory() as directory:
            with patch("run_comparison.source_fingerprint", side_effect=[original, {"changed": True}]):
                with self.assertRaisesRegex(ValueError, "Source files changed during"):
                    compare(["random"], [1], budget=2, population=2, smoke=True, out_dir=directory)
            manifest = json.loads(next(Path(directory).glob("*/comparison.json")).read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["runs"], [])
