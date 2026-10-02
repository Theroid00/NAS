from ga.chromosome import NUM_GENES, SCHEMA_VERSION
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def loaders(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(torch.randn(8, 30), torch.arange(8) % 2)
        loader = DataLoader(dataset, batch_size=4)
        return loader, loader, loader

    def test_supported_depths_have_valid_forward_and_backward(self):
        import torch
        from ga.chromosome import decode
        from models.mlp import build_model, estimate_parameters, count_parameters
        for depth in range(4):
            for residual in (0, 1):
                chromosome = [0] * NUM_GENES
                chromosome[0], chromosome[8] = depth, residual
                model = build_model(decode(chromosome))
                logits = model(torch.randn(2, 30))
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertEqual(count_parameters(model), estimate_parameters(decode(chromosome)))
                logits.sum().backward()

    def test_baseline_parameter_count(self):
        from models.mlp import build_baseline_mlp
        from models.mlp import count_parameters
        self.assertEqual(count_parameters(build_baseline_mlp()), 4130)

    def test_device_requests_are_validated_before_training(self):
        from utils.search_runtime import resolve_device
        self.assertEqual(resolve_device("cpu"), "cpu")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            self.assertEqual(resolve_device("cuda"), "cuda:0")
            with self.assertRaises(ValueError):
                resolve_device("cuda:1")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(ValueError):
                resolve_device("cuda")

    def test_baseline_uses_shared_optimizer_and_checkpoint_protocol(self):
        from training.trainer import full_train
        with tempfile.TemporaryDirectory() as directory, patch("data.tabular.get_full_loaders", return_value=self.loaders()):
            result = full_train(baseline=True, epochs=1, save_dir=directory, evaluate_test=False)
            self.assertEqual(result["protocol"]["optimizer"], "Adam")
            self.assertEqual(result["protocol"]["checkpoint_policy"], "best_validation")
            self.assertEqual(result["num_params"], 4130)

    def test_parameter_limit_does_not_load_training_data(self):
        from training.evaluator import evaluate_trial
        with patch("data.tabular.get_proxy_loaders") as loader, patch("models.mlp.build_model") as builder:
            result = evaluate_trial([0] * NUM_GENES, max_params=1)
            self.assertEqual(result["status"], "parameter_limit")
            loader.assert_not_called()
            builder.assert_not_called()

    def test_infrastructure_failure_is_not_zero_fitness(self):
        from training.evaluator import evaluate_trial
        with patch("data.tabular.get_proxy_loaders", side_effect=RuntimeError("dataset unavailable")):
            with self.assertRaisesRegex(RuntimeError, "dataset unavailable"):
                evaluate_trial([0] * NUM_GENES)

    def test_proxy_actually_trains_on_synthetic_data(self):
        from training.evaluator import evaluate_trial
        with patch("data.tabular.get_proxy_loaders", return_value=self.loaders()[:2]):
            result = evaluate_trial([0] * NUM_GENES, proxy_epochs=1)
            self.assertEqual(result["status"], "ok")
            self.assertGreater(result["num_params"], 0)

    def test_zero_validation_saves_checkpoint_and_test_is_opt_in(self):
        from training.trainer import full_train, evaluate_checkpoint
        with tempfile.TemporaryDirectory() as directory, patch("data.tabular.get_full_loaders", return_value=self.loaders()), patch(
            "training.trainer.validate", return_value=0.0
        ) as validate:
            result = full_train([0] * NUM_GENES, epochs=1, save_dir=directory, evaluate_test=False)
            self.assertTrue(Path(result["checkpoint_path"]).exists())
            self.assertIsNone(result["test_accuracy"])
            self.assertEqual(validate.call_count, 1)
            result = evaluate_checkpoint(result["results_path"])
            self.assertEqual(result["test_accuracy"], 0.0)
            with self.assertRaisesRegex(ValueError, "already"):
                evaluate_checkpoint(result["results_path"])

    def test_full_training_restores_best_validation_weights(self):
        import torch
        from training.trainer import train_model
        model = torch.nn.Sequential(torch.nn.Linear(30, 2))
        with tempfile.TemporaryDirectory() as directory, patch("data.tabular.get_full_loaders", return_value=self.loaders()), patch(
            "training.trainer.validate", side_effect=[0.8, 0.3]
        ):
            result = train_model(model, "cpu", 2, directory, "restore", 42, 42, evaluate_test=False)
            checkpoint = torch.load(result["checkpoint_path"], weights_only=True)
            self.assertEqual(result["best_val_accuracy"], 0.8)
            self.assertTrue(all(torch.equal(value, checkpoint[key]) for key, value in model.state_dict().items()))

    def test_tabular_split_and_scaling_do_not_leak(self):
        import numpy as np
        from sklearn.datasets import load_breast_cancer
        from data.tabular import get_proxy_loaders, dataset_metadata
        first = get_proxy_loaders(seed=5, training_seed=1)
        second = get_proxy_loaders(seed=5, training_seed=2)
        self.assertTrue(np.array_equal(first[0].dataset.tensors[0], second[0].dataset.tensors[0]))
        metadata = dataset_metadata(5)
        splits = [set(v) for v in metadata["split_indices"].values()]
        self.assertEqual(sum(len(v) for v in splits), 569)
        self.assertFalse(splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
        raw = load_breast_cancer()
        train_indices = metadata["split_indices"]["train"]
        np.testing.assert_allclose(metadata["scaler_mean"], raw.data[train_indices].mean(axis=0))
        np.testing.assert_allclose(first[0].dataset.tensors[0].numpy().mean(axis=0), 0, atol=1e-6)
        for indices in metadata["split_indices"].values():
            self.assertEqual(len(set(raw.target[indices])), 2)
        self.assertIs(first[0].dataset, second[0].dataset)

    def test_real_tabular_proxy_is_repeatable(self):
        from training.evaluator import evaluate_trial
        a = evaluate_trial([0] * NUM_GENES, proxy_epochs=2, seed=7)
        b = evaluate_trial([0] * NUM_GENES, proxy_epochs=2, seed=7)
        self.assertEqual(a["fitness"], b["fitness"])
        self.assertEqual(a["status"], "ok")

    def test_invalid_proxy_size_is_rejected(self):
        from data.tabular import get_proxy_loaders
        for size in (-1, 342, 1):
            with self.assertRaises(ValueError):
                get_proxy_loaders(proxy_size=size)


class RankingTests(unittest.TestCase):
    def test_rank_agreement_and_constant_scores(self):
        from calibrate_proxy import ranking_metrics, ranks
        self.assertEqual(ranks([2, 2, 1]), [2.5, 2.5, 1.0])
        self.assertAlmostEqual(ranking_metrics([1, 2, 3], [3, 2, 1])["spearman"], -1.0)
        self.assertIsNone(ranking_metrics([1, 1, 1], [1, 2, 3])["spearman"])

    def test_saved_smoke_legacy_and_mismatched_architectures_are_rejected(self):
        from utils.results import load_winner
        from ga.chromosome import decode
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "winner.json"
            compatible = {"schema_version": SCHEMA_VERSION, "dataset_name": "breast_cancer_wisconsin",
                          "best_chromosome": [0] * NUM_GENES}
            for record in ({"smoke": True}, {"schema_version": 2, "best_chromosome": [0] * 13},
                           {**compatible, "dataset_name": "cifar10"}, {**compatible, "best_arch": {}}):
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    load_winner(path)
            path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "dataset_name": "breast_cancer_wisconsin", "best_chromosome": [0] * NUM_GENES, "best_arch": decode([0] * NUM_GENES)}))
            self.assertEqual(load_winner(path)["best_chromosome"], [0] * NUM_GENES)
