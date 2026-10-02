import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(AVAILABLE, "Training dependencies not installed")
class CovertypeTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from data import tabular
        torch.set_num_threads(1)
        tabular._split.cache_clear()
        tabular._proxy_datasets.cache_clear()
        rng = np.random.default_rng(8)
        self.x = rng.integers(0, 2, size=(210, 54)).astype(float)
        self.x[:, :10] = rng.normal(20, 3, size=(210, 10))
        self.y = np.tile(np.arange(7), 30)
        self.raw = patch("data.tabular._raw", return_value=(self.x, self.y))
        self.raw.start()

    def tearDown(self):
        from data import tabular
        self.raw.stop()
        tabular._split.cache_clear()
        tabular._proxy_datasets.cache_clear()

    def test_stratified_subsets_and_numeric_only_preprocessing(self):
        import numpy as np
        from data.tabular import get_full_loaders, get_proxy_loaders, dataset_metadata
        full = get_full_loaders(seed=9, dataset="covertype")
        train, val = get_proxy_loaders(seed=9, dataset="covertype", proxy_size=70, validation_size=21)
        again = get_proxy_loaders(seed=9, dataset="covertype", proxy_size=70, validation_size=21, training_seed=99)
        self.assertIs(train.dataset, again[0].dataset)
        self.assertIs(val.dataset, again[1].dataset)
        self.assertEqual(len(train.dataset), 70)
        self.assertEqual(len(val.dataset), 21)
        self.assertEqual(train.batch_size, 512)
        self.assertFalse(set(train.dataset.row_indices) & set(val.dataset.row_indices))
        self.assertFalse(set(val.dataset.row_indices) & set(full[2].dataset.row_indices))
        self.assertEqual(set(train.dataset.tensors[1].tolist()), set(range(7)))
        metadata = dataset_metadata(9, "covertype")
        np.testing.assert_allclose(metadata["scaler_mean"], self.x[full[0].dataset.row_indices, :10].mean(axis=0))
        np.testing.assert_array_equal(train.dataset.tensors[0][:, 10:], self.x[train.dataset.row_indices, 10:])

    def test_seven_class_models_have_valid_gradients_and_exact_parameter_estimates(self):
        import torch
        from ga.chromosome import decode, NUM_GENES
        from models.mlp import build_model, estimate_parameters, count_parameters
        for depth in range(4):
            chromosome = [0] * NUM_GENES
            chromosome[0] = depth
            arch = decode(chromosome)
            model = build_model(arch, input_features=54, num_classes=7)
            logits = model(torch.randn(8, 54))
            self.assertEqual(tuple(logits.shape), (8, 7))
            self.assertEqual(count_parameters(model), estimate_parameters(arch, 54, 7))
            torch.nn.functional.cross_entropy(logits, torch.arange(8) % 7).backward()

    def test_covertype_checkpoint_uses_saved_dimensions_and_rejects_changed_data(self):
        from ga.chromosome import NUM_GENES
        from training.trainer import full_train, evaluate_checkpoint
        with tempfile.TemporaryDirectory() as root:
            result = full_train([0] * NUM_GENES, dataset="covertype", epochs=2,
                                save_dir=root, split_seed=9, evaluate_test=False)
            self.assertTrue(math.isfinite(result["best_val_loss"]))
            path = Path(result["results_path"])
            original = path.read_text()
            changed = json.loads(original)
            changed["dataset"]["data_sha256"] = "different"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                evaluate_checkpoint(path)
            path.write_text(original)
            scored = evaluate_checkpoint(path)
            self.assertTrue(0 <= scored["test_accuracy"] <= 1)

    def test_loss_distinguishes_equal_accuracy(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from training.evaluator import validate
        labels = torch.tensor([0, 1])
        confident = DataLoader(TensorDataset(torch.tensor([[2., 0.], [0., 2.]]), labels))
        uncertain = DataLoader(TensorDataset(torch.tensor([[.1, 0.], [0., .1]]), labels))
        a = validate(torch.nn.Identity(), confident, "cpu", return_metrics=True)
        b = validate(torch.nn.Identity(), uncertain, "cpu", return_metrics=True)
        self.assertEqual(a["accuracy"], b["accuracy"])
        self.assertLess(a["loss"], b["loss"])
