import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from serving.bundle import package, install, download


class BundleTests(unittest.TestCase):
    def test_release_round_trip_and_checksum_preserve_predictions(self):
        from training.trainer import full_train
        from ga.chromosome import NUM_GENES
        from make_example import make_example
        from serving.artifact import export_artifact, Predictor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = full_train([0] * NUM_GENES, epochs=1, evaluate_test=False, save_dir=root / "training")
            example = root / "example.json"
            make_example(full["results_path"], example)
            export_artifact(full["results_path"], root / "artifact", example)
            release = package(root / "artifact", root / "model.zip")
            package(root / "artifact", root / "repeat.zip")
            self.assertEqual((root / "model.zip").read_bytes(), (root / "repeat.zip").read_bytes())
            install(root / "model.zip", root / "installed", release["sha256"])
            rows = json.loads(example.read_text())["features"]
            self.assertEqual(Predictor(root / "artifact").predict(rows)["predictions"],
                             Predictor(root / "installed").predict(rows)["predictions"])
            with self.assertRaises(FileExistsError):
                install(root / "model.zip", root / "installed", release["sha256"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                install(root / "model.zip", root / "bad", "0" * 64)
            self.assertFalse((root / "bad").exists())

    def test_untrusted_archive_names_are_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "outside")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "exactly"):
                install(archive, root / "installed", digest)
            self.assertFalse((root / "escape.txt").exists())
            self.assertFalse((root / "installed").exists())

    def test_download_requires_https_and_pin(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            download("http://example.com/latest.zip", "unused", "0" * 64)
