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
        dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 10)
        loader = DataLoader(dataset, batch_size=4)
        return loader, loader, loader

    def test_supported_depths_have_valid_forward_and_backward(self):
        import torch
        from ga.chromosome import decode
        from models.builder import build_model
        for depth in range(4):
            for residual in (0, 1):
                chromosome = [0] * 13
                chromosome[0], chromosome[12] = depth, residual
                model = build_model(decode(chromosome))
                logits = model(torch.randn(2, 3, 32, 32))
                self.assertEqual(tuple(logits.shape), (2, 10))
                logits.sum().backward()

    def test_baseline_parameter_count(self):
        from models.baselines.resnet import build_baseline_resnet
        from models.builder import count_parameters
        self.assertEqual(count_parameters(build_baseline_resnet()), 1227594)

    def test_parameter_limit_does_not_load_training_data(self):
        from training.evaluator import evaluate_trial
        with patch("data.cifar.get_proxy_loaders") as loader:
            result = evaluate_trial([0] * 13, max_params=1)
            self.assertEqual(result["status"], "parameter_limit")
            loader.assert_not_called()

    def test_infrastructure_failure_is_not_zero_fitness(self):
        from training.evaluator import evaluate_trial
        with patch("data.cifar.get_proxy_loaders", side_effect=RuntimeError("dataset unavailable")):
            with self.assertRaisesRegex(RuntimeError, "dataset unavailable"):
                evaluate_trial([0] * 13)

    def test_proxy_actually_trains_on_synthetic_data(self):
        from training.evaluator import evaluate_trial
        with patch("data.cifar.get_proxy_loaders", return_value=self.loaders()[:2]):
            result = evaluate_trial([0] * 13, proxy_epochs=1)
            self.assertEqual(result["status"], "ok")
            self.assertGreater(result["num_params"], 0)

    def test_zero_validation_saves_checkpoint_and_test_is_opt_in(self):
        from training.trainer import full_train, evaluate_checkpoint
        with tempfile.TemporaryDirectory() as directory, patch("data.cifar.get_full_loaders", return_value=self.loaders()), patch(
            "training.trainer.validate", return_value=0.0
        ) as validate:
            result = full_train([0] * 13, epochs=1, save_dir=directory, evaluate_test=False)
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
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 10))
        with tempfile.TemporaryDirectory() as directory, patch("data.cifar.get_full_loaders", return_value=self.loaders()), patch(
            "training.trainer.validate", side_effect=[0.8, 0.3]
        ):
            result = train_model(model, "cpu", 2, directory, "restore", 42, 42, evaluate_test=False)
            checkpoint = torch.load(result["checkpoint_path"], weights_only=True)
            self.assertEqual(result["best_val_accuracy"], 0.8)
            self.assertTrue(all(torch.equal(value, checkpoint[key]) for key, value in model.state_dict().items()))

    def test_proxy_split_is_clean_disjoint_and_independent_of_training_seed(self):
        import torch
        from data.cifar import get_proxy_loaders
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(torch.zeros(50000, 1), torch.zeros(50000, dtype=torch.long))
        with patch("data.cifar._dataset", return_value=dataset), patch("data.cifar.NUM_WORKERS", 0):
            first = get_proxy_loaders(proxy_size=10, seed=5, training_seed=1)
            second = get_proxy_loaders(proxy_size=10, seed=5, training_seed=2)
            self.assertEqual(first[0].dataset.indices, second[0].dataset.indices)
            self.assertEqual(first[1].dataset.indices, second[1].dataset.indices)
            self.assertFalse(set(first[0].dataset.indices) & set(first[1].dataset.indices))


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
            for record in ({"smoke": True}, {"best_chromosome": [0] * 10},
                           {"best_chromosome": [0] * 13, "best_arch": {}}):
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    load_winner(path)
            path.write_text(json.dumps({"best_chromosome": [0] * 13, "best_arch": decode([0] * 13)}))
            self.assertEqual(load_winner(path)["best_chromosome"], [0] * 13)
