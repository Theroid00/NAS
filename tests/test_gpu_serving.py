"""Optional integration check against an already trained artifact, without retraining."""
import json
import os
from pathlib import Path
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("NAS_TEST_ARTIFACT"),
                     "Set NAS_TEST_ARTIFACT to verify an exported model on CUDA")
class GpuServingTests(unittest.TestCase):
    def test_gpu_api_matches_independently_preprocessed_checkpoint(self):
        import numpy as np
        from fastapi.testclient import TestClient
        from models.mlp import build_model
        from serving.api import create_app
        root = Path(os.environ["NAS_TEST_ARTIFACT"])
        manifest = json.loads((root / "manifest.json").read_text())
        rows = json.loads((root / "example.json").read_text())["features"]
        raw = np.asarray(rows, dtype=np.float64)
        standardized = raw.astype(np.float32)
        columns = manifest["scaled_feature_columns"]
        standardized[:, columns] = ((raw[:, columns] - np.asarray(manifest["scaler_mean"])) /
                                     np.asarray(manifest["scaler_scale"])).astype(np.float32)
        model = build_model(manifest["arch"], manifest["input_features"], manifest["num_classes"]).cuda().eval()
        model.load_state_dict(torch.load(root / "model.pt", map_location="cuda", weights_only=True))
        with torch.inference_mode():
            expected = model(torch.from_numpy(standardized).cuda()).softmax(1).cpu().numpy()
        with TestClient(create_app(root, device="cuda")) as client:
            self.assertEqual(client.get("/health").json()["device"], "cuda:0")
            response = client.post("/predict", json={"features": rows})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["device"], "cuda:0")
            np.testing.assert_allclose([row["probabilities"] for row in response.json()["predictions"]],
                                       expected, atol=1e-7)
