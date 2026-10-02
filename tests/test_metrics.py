import unittest


class MetricTests(unittest.TestCase):
    def test_streaming_metrics_match_independent_sklearn_on_imbalanced_uneven_batches(self):
        import numpy as np
        import torch
        from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, recall_score
        from torch.utils.data import DataLoader, TensorDataset
        from training.evaluator import validate
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2])
        predictions = np.array([0, 0, 0, 1, 0, 0, 0, 2, 0])
        logits = torch.eye(3)[predictions] * 3
        loader = DataLoader(TensorDataset(logits, torch.tensor(labels)), batch_size=4)
        result = validate(torch.nn.Identity(), loader, "cpu", return_metrics=True)
        self.assertAlmostEqual(result["accuracy"], float((labels == predictions).mean()))
        self.assertAlmostEqual(result["balanced_accuracy"], balanced_accuracy_score(labels, predictions))
        self.assertAlmostEqual(result["macro_f1"], f1_score(labels, predictions, average="macro", zero_division=0))
        np.testing.assert_array_equal(result["confusion_matrix"], confusion_matrix(labels, predictions))
        np.testing.assert_allclose(result["per_class_recall"], recall_score(labels, predictions, average=None, zero_division=0))

    def test_absent_classes_and_zero_predictions_have_defined_metrics(self):
        from training.metrics import classification_metrics
        result = classification_metrics([[2, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["per_class_recall"], [1., 0., 0.])
        self.assertAlmostEqual(result["macro_f1"], 0.8 / 3)
