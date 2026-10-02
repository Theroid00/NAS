"""Verify pinned packages and run a tiny offline model-training check on CUDA."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def verify_pins(requirements_path):
    """Check the flat pinned environment files without relying on another venv."""
    versions = {}
    for line in Path(requirements_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        name, expected = line.split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, installed {actual}")
        versions[name] = actual
    return versions


def verify_environment(device="cuda", service=False, artifact=None):
    import torch
    from ga.chromosome import NUM_GENES
    from training.evaluator import evaluate_trial
    from utils.search_runtime import resolve_device
    from utils.provenance import source_fingerprint
    root = Path(__file__).resolve().parent
    resolved = resolve_device(device)
    requirements = root / ("requirements-gpu.txt" if resolved.startswith("cuda") else "requirements-tested-cpu.txt")
    versions = verify_pins(requirements)
    if service:
        versions.update(verify_pins(root / "requirements-service.txt"))
    trial = evaluate_trial([0] * NUM_GENES, device=resolved, proxy_epochs=1, seed=7,
                           dataset="breast_cancer_wisconsin")
    if trial["status"] != "ok":
        raise RuntimeError(f"GPU/environment training verification failed: {trial}")
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "isolated_virtual_environment": sys.prefix != sys.base_prefix,
              "device": resolved, "torch_cuda_runtime": torch.version.cuda,
              "gpu_name": torch.cuda.get_device_name(resolved) if resolved.startswith("cuda") else None,
              "packages": versions, "source_fingerprint": source_fingerprint(),
              "training_check": {"dataset": "breast_cancer_wisconsin", "epochs": 1,
                                 "status": trial["status"], "device": resolved,
                                 "validation_accuracy": trial["fitness"], "num_params": trial["num_params"]}}
    if artifact:
        from serving.artifact import Predictor
        artifact = Path(artifact)
        example = json.loads((artifact / "example.json").read_text(encoding="utf-8"))
        predictor = Predictor(artifact, device=resolved)
        prediction = predictor.predict(example["features"])
        if prediction["device"] != resolved:
            raise RuntimeError("Inference ran on an unexpected device")
        result["inference_check"] = {"model_id": prediction["model_id"], "device": prediction["device"],
                                     "rows": len(prediction["predictions"]), "status": "ok"}
        if service:
            from fastapi.testclient import TestClient
            from serving.api import create_app
            with TestClient(create_app(artifact, device=resolved)) as client:
                response = client.post("/predict", json=example)
                if response.status_code != 200 or response.json()["device"] != resolved:
                    raise RuntimeError("API prediction verification failed")
                result["api_check"] = {"status": "ok", "device": response.json()["device"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--service", action="store_true", help="Also verify optional service dependency pins")
    parser.add_argument("--artifact", help="Also check predictions from an existing artifact with a bundled example")
    parser.add_argument("--out", help="Optionally save verification evidence; otherwise print JSON")
    args = parser.parse_args()
    result = verify_environment(args.device, args.service, args.artifact)
    if args.out:
        from utils.persistence import atomic_json
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
        print(f"Verified {result['device']}; evidence saved to {path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
