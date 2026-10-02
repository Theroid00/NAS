"""Shared retraining preserves winner provenance across command-line entry points."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ga.chromosome import NUM_GENES, SCHEMA_VERSION
from main import parse_args
from training.trainer import train_winner


class EntrypointTests(unittest.TestCase):
    def winner(self, directory):
        path = Path(directory) / "best.json"
        path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "run_id": "saved-run",
                                   "dataset_name": "covertype", "best_chromosome": [0] * NUM_GENES,
                                   "hyperparams": {"split_seed": 77}}))
        return path

    def test_saved_winner_dataset_and_split_override_cli_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("training.trainer.full_train", return_value={"saved": True}) as train:
                result = train_winner(self.winner(directory), device="cuda", epochs=7,
                                      seed=101, evaluate_test=False)
            self.assertEqual(result, {"saved": True})
            self.assertEqual(train.call_args.args, ([0] * NUM_GENES,))
            self.assertEqual(train.call_args.kwargs["dataset"], "covertype")
            self.assertEqual(train.call_args.kwargs["split_seed"], 77)
            self.assertEqual(train.call_args.kwargs["device"], "cuda")
            self.assertFalse(train.call_args.kwargs["evaluate_test"])

    def test_explicit_split_override_and_invalid_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.winner(directory)
            with patch("training.trainer.full_train") as train:
                train_winner(path, split_seed=9)
                self.assertEqual(train.call_args.kwargs["split_seed"], 9)
                path.write_text(json.dumps({"smoke": True}))
                train.reset_mock()
                with self.assertRaises(ValueError):
                    train_winner(path)
                train.assert_not_called()

    def test_main_and_evaluate_cli_delegate_to_same_retraining_function(self):
        from main import main
        from evaluate_best import main as evaluate_main
        for entrypoint, arguments in ((main, ["main.py", "--mode", "train-best", "--best-json", "saved.json"]),
                                      (evaluate_main, ["evaluate_best.py", "--json", "saved.json"])):
            with self.subTest(entrypoint=entrypoint.__module__):
                with patch("sys.argv", arguments + ["--validation-only"]), patch(
                        "training.trainer.train_winner", return_value={"results_path": "full.json",
                                                                       "best_val_accuracy": .9,
                                                                       "test_accuracy": None}) as train:
                    entrypoint()
                self.assertEqual(train.call_args.args, ("saved.json",))
                self.assertEqual(train.call_args.kwargs["device"], "cuda")
                self.assertFalse(train.call_args.kwargs["evaluate_test"])

    def test_gpu_default_and_explicit_cpu_option(self):
        self.assertEqual(parse_args([]).device, "cuda")
        self.assertEqual(parse_args(["--device", "cpu"]).device, "cpu")
