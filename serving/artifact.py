"""Export trained models with preprocessing; inference never loads training data."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from data.specs import dataset_spec
from ga.chromosome import SCHEMA_VERSION, decode
from models.mlp import build_model
from utils.persistence import atomic_json


def feature_names(dataset):
    if dataset == "breast_cancer_wisconsin":
        from sklearn.datasets import load_breast_cancer
        return load_breast_cancer().feature_names.tolist()
    return ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"] + [
                f"Wilderness_Area_{i}" for i in range(1, 5)] + [f"Soil_Type_{i}" for i in range(1, 41)]


def export_artifact(results_path, destination, example_path=None):
    source = json.loads(Path(results_path).read_text(encoding="utf-8"))
    if source.get("schema_version") != SCHEMA_VERSION or source.get("baseline"):
        raise ValueError("Export requires a compatible searched model's full-training result")
    if source["arch"] != decode(source["chromosome"]):
        raise ValueError("Training architecture does not match chromosome")
    dataset = source["dataset"]
    spec = dataset_spec(dataset["name"])
    if (dataset["input_features"], dataset["num_classes"]) != (spec["input_features"], spec["num_classes"]):
        raise ValueError("Training dataset dimensions are inconsistent")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    weights = destination / "model.pt"
    shutil.copyfile(source["checkpoint_path"], weights)
    labels = ["malignant", "benign"] if dataset["name"] == "breast_cancer_wisconsin" else [
        "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow",
        "Aspen", "Douglas-fir", "Krummholz"]
    manifest = {"artifact_version": 1, "schema_version": SCHEMA_VERSION, "model_id": source["run_id"],
                "dataset_name": dataset["name"], "input_features": spec["input_features"],
                "num_classes": spec["num_classes"], "arch": source["arch"],
                "feature_names": feature_names(dataset["name"]), "class_names": labels,
                "source_labels": list(range(1, 8)) if dataset["name"] == "covertype" else [0, 1],
                "scaled_feature_columns": dataset["scaled_feature_columns"],
                "scaler_mean": dataset["scaler_mean"], "scaler_scale": dataset["scaler_scale"],
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "data_sha256": dataset["data_sha256"], "split_seed": source["split_seed"],
                "training_seed": source["seed"], "num_params": source["num_params"],
                "example_available": example_path is not None,
                "validation_metrics": source.get("best_val_metrics"), "test_metrics": source.get("test_metrics"),
                "input_contract": "Raw unscaled numeric features in feature_names order; no missing values"}
    atomic_json(destination / "manifest.json", manifest)
    predictor = Predictor(destination)  # Verify weights, dimensions, and preprocessing.
    if example_path:
        example = json.loads(Path(example_path).read_text(encoding="utf-8"))
        predictor.transform(example["features"])
        atomic_json(destination / "example.json", example)
    return manifest


class Predictor:
    def __init__(self, artifact_dir, device="cpu"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; install a CUDA-enabled PyTorch build")
        root = Path(artifact_dir)
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        m = self.manifest
        if m.get("artifact_version") != 1 or m.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported inference artifact")
        spec = dataset_spec(m["dataset_name"])
        if (m["input_features"], m["num_classes"]) != (spec["input_features"], spec["num_classes"]):
            raise ValueError("Artifact dimensions do not match dataset")
        if (len(m["feature_names"]) != m["input_features"] or len(m["class_names"]) != m["num_classes"]
                or len(m["source_labels"]) != m["num_classes"]):
            raise ValueError("Artifact feature/class names are inconsistent")
        self.columns = np.asarray(m["scaled_feature_columns"], dtype=int)
        self.mean = np.asarray(m["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(m["scaler_scale"], dtype=np.float64)
        if (self.mean.shape != self.columns.shape or self.scale.shape != self.columns.shape
                or len(set(self.columns.tolist())) != len(self.columns)
                or (self.columns < 0).any() or (self.columns >= m["input_features"]).any()
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any()):
            raise ValueError("Invalid artifact preprocessing")
        weights = root / "model.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != m["weights_sha256"]:
            raise ValueError("Artifact weights checksum differs")
        self.model = build_model(m["arch"], m["input_features"], m["num_classes"])
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    def transform(self, rows):
        raw = np.asarray(rows, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != self.manifest["input_features"] or not 1 <= len(raw) <= 1024:
            raise ValueError(f"Provide 1..1024 rows with {self.manifest['input_features']} features each")
        if not np.isfinite(raw).all():
            raise ValueError("Features must be finite numbers")
        if self.manifest["dataset_name"] == "covertype" and not np.isin(raw[:, 10:], [0, 1]).all():
            raise ValueError("Covertype wilderness and soil features must be binary (0 or 1)")
        features = raw.astype(np.float32)
        features[:, self.columns] = ((raw[:, self.columns] - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Features exceed supported numeric range")
        return torch.from_numpy(features)

    def predict(self, rows):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        features = self.transform(rows).to(self.device)
        with torch.inference_mode():
            logits = self.model(features)
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced nonfinite predictions")
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        ids = probabilities.argmax(axis=1).tolist()
        return {"model_id": self.manifest["model_id"], "dataset": self.manifest["dataset_name"],
                "device": str(next(self.model.parameters()).device),
                "predictions": [{"class_id": idx, "source_label": self.manifest["source_labels"][idx],
                                 "class_name": self.manifest["class_names"][idx], "probabilities": row.tolist()}
                                for idx, row in zip(ids, probabilities)],
                "elapsed_ms": (time.perf_counter() - started) * 1000}
