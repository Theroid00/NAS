import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "Service dependencies not installed")
class ServingTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def trained_artifact(self):
        from ga.chromosome import NUM_GENES
        from serving.artifact import export_artifact
        from training.trainer import full_train
        result = full_train([0] * NUM_GENES, epochs=1, save_dir=self.root / "training", evaluate_test=False)
        artifact = self.root / "artifact"
        export_artifact(result["results_path"], artifact)
        return result, artifact

    def test_exported_preprocessing_and_api_match_checkpoint_predictions_without_data_cache(self):
        import numpy as np
        import torch
        from sklearn.datasets import load_breast_cancer
        from fastapi.testclient import TestClient
        from models.mlp import build_model
        from serving.artifact import Predictor
        from serving.api import create_app
        result, artifact = self.trained_artifact()
        raw = load_breast_cancer().data[:3]
        metadata = result["dataset"]
        standardized = ((raw - np.array(metadata["scaler_mean"])) / np.array(metadata["scaler_scale"])).astype(np.float32)
        model = build_model(result["arch"])
        model.load_state_dict(torch.load(result["checkpoint_path"], weights_only=True))
        model.eval()
        with torch.no_grad():
            expected = model(torch.from_numpy(standardized)).softmax(1).numpy()
        with patch("data.tabular._raw", side_effect=AssertionError("Inference loaded data")):
            predictor = Predictor(artifact)
            np.testing.assert_array_equal(predictor.transform(raw).numpy(), standardized)
            with TestClient(create_app(artifact)) as client:
                self.assertTrue(client.get("/health").json()["model_loaded"])
                self.assertEqual(client.get("/model").json()["input_features"], 30)
                response = client.post("/predict", json={"features": raw.tolist()})
                self.assertEqual(response.status_code, 200)
                np.testing.assert_allclose([p["probabilities"] for p in response.json()["predictions"]], expected, atol=1e-7)
                self.assertIn("x-response-time-ms", response.headers)
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.get("/docs").status_code, 200)

    def test_bad_requests_and_unconfigured_model_have_clear_errors(self):
        from fastapi.testclient import TestClient
        from serving.api import create_app
        _, artifact = self.trained_artifact()
        invalid = [[], [[0.] * 29], [[0.] * 31], [["1"] * 30], [[True] * 30], [[0.] * 30] * 1025]
        with TestClient(create_app(artifact)) as client:
            for rows in invalid:
                with self.subTest(rows=len(rows)):
                    self.assertEqual(client.post("/predict", json={"features": rows}).status_code, 422)
            self.assertEqual(client.post("/predict", content='{"features": [[NaN]]}',
                                         headers={"Content-Type": "application/json"}).status_code, 422)
        with TestClient(create_app()) as client:
            self.assertFalse(client.get("/health").json()["model_loaded"])
            self.assertEqual(client.post("/predict", json={"features": [[0.] * 30]}).status_code, 503)
            self.assertEqual(client.get("/report").status_code, 404)

    def test_corrupt_weights_and_invalid_preprocessing_are_rejected(self):
        from serving.artifact import Predictor
        _, artifact = self.trained_artifact()
        weights = artifact / "model.pt"
        content = weights.read_bytes()
        weights.write_bytes(content + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Predictor(artifact)
        weights.write_bytes(content)
        path = artifact / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["scaler_scale"][0] = 0
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            Predictor(artifact)

    def test_covertype_binary_columns_and_source_labels_are_preserved(self):
        import hashlib
        import numpy as np
        import torch
        from models.mlp import build_model
        from ga.chromosome import decode, NUM_GENES, SCHEMA_VERSION
        from serving.artifact import Predictor, feature_names
        torch.save(build_model(decode([0] * NUM_GENES), 54, 7).state_dict(), self.root / "model.pt")
        manifest = {"artifact_version": 1, "schema_version": SCHEMA_VERSION, "model_id": "test",
                    "dataset_name": "covertype", "input_features": 54, "num_classes": 7,
                    "arch": decode([0] * NUM_GENES), "feature_names": feature_names("covertype"),
                    "class_names": [str(i) for i in range(7)], "source_labels": list(range(1, 8)),
                    "scaled_feature_columns": list(range(10)), "scaler_mean": [2.] * 10,
                    "scaler_scale": [3.] * 10,
                    "weights_sha256": hashlib.sha256((self.root / "model.pt").read_bytes()).hexdigest()}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        predictor = Predictor(self.root)
        raw = np.ones((2, 54))
        np.testing.assert_array_equal(predictor.transform(raw).numpy()[:, 10:], raw[:, 10:])
        prediction = predictor.predict(raw)["predictions"][0]
        self.assertEqual(prediction["source_label"], prediction["class_id"] + 1)
        raw[0, 10] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            predictor.predict(raw)

    def test_bundled_example_uses_training_rows_and_report_dataset_must_match(self):
        from fastapi.testclient import TestClient
        from serving.api import create_app
        from serving.artifact import export_artifact
        from make_example import make_example
        result, _ = self.trained_artifact()
        example = self.root / "example.json"
        index = make_example(result["results_path"], example)
        self.assertIn(index, result["dataset"]["split_indices"]["train"])
        artifact = self.root / "with-example"
        export_artifact(result["results_path"], artifact, example)
        report = self.root / "report.json"
        report.write_text(json.dumps({"report_version": 1, "dataset": "breast_cancer_wisconsin"}))
        with TestClient(create_app(artifact, report)) as client:
            payload = client.get("/example").json()
            self.assertEqual(client.post("/predict", json=payload).status_code, 200)
            self.assertEqual(client.get("/dashboard.js").status_code, 200)
        report.write_text(json.dumps({"report_version": 1, "dataset": "covertype"}))
        with self.assertRaisesRegex(ValueError, "different datasets"):
            with TestClient(create_app(artifact, report)):
                pass
