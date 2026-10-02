"""Run or finish the bounded industry demo: search, evaluate, export, report."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

from reporting.comparison import build_report, save_report
from utils.persistence import atomic_json, RunLock


def benchmark_inference(artifact_dir, example_path, repeats=100, device="cuda"):
    import numpy as np
    import torch
    from serving.artifact import Predictor
    predictor = Predictor(artifact_dir, device=device)
    example = json.loads(Path(example_path).read_text(encoding="utf-8"))["features"][0]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    measurements = []
    try:
        for batch_size in (1, 32):
            rows = [example] * batch_size
            for _ in range(10):
                predictor.predict(rows)
            timings = []
            for _ in range(repeats):
                started = time.perf_counter()
                predictor.predict(rows)
                timings.append((time.perf_counter() - started) * 1000)
            measurements.append({"batch_size": batch_size, "repeats": repeats,
                                 "median_ms": statistics.median(timings),
                                 "p95_ms": float(np.percentile(timings, 95)),
                                 "rows_per_second": batch_size / (statistics.mean(timings) / 1000)})
    finally:
        torch.set_num_threads(previous_threads)
    return {"device": str(next(predictor.model.parameters()).device),
            "gpu_name": torch.cuda.get_device_name(predictor.device) if predictor.device.type == "cuda" else None,
            "threads": 1, "warmup_requests": 10, "measurements": measurements,
            "scope": "In-process preprocessing, forward pass, softmax, and response construction; excludes HTTP",
            "input_policy": "Repeated raw training example, no accuracy claim", "platform": platform.platform()}


def finalize_comparison(comparison_path, report_path="docs/industry-results.json", artifact_root="artifacts/industry"):
    from make_example import make_example
    from serving.artifact import export_artifact, Predictor
    from training.trainer import evaluate_checkpoint
    comparison_path = Path(comparison_path).resolve()
    lock = RunLock(comparison_path.with_name("delivery.lock"))
    lock.acquire()
    try:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison["status"] != "completed" or comparison["smoke"]:
            raise ValueError("Finish a real comparison suite before packaging the industry demo")
        report = build_report(comparison_path)
        selected = report["selected_model"]
        if selected is None:
            raise ValueError("Comparison needs full-trained winners before export")
        result = json.loads(Path(selected["results_path"]).read_text(encoding="utf-8"))
        if result["test_accuracy"] is None:
            result = evaluate_checkpoint(selected["results_path"], comparison["config"]["device"])
        example_path = comparison_path.parent / "example-request.json"
        training_row = make_example(selected["results_path"], example_path)
        artifact_dir = Path(artifact_root).resolve() / selected["model_id"]
        if artifact_dir.exists():
            predictor = Predictor(artifact_dir)
            if predictor.manifest["model_id"] != selected["model_id"]:
                raise ValueError("Artifact directory contains another model")
        else:
            export_artifact(selected["results_path"], artifact_dir, example_path)
        report = save_report(comparison_path, report_path)
        report["inference_benchmark"] = benchmark_inference(artifact_dir, example_path, device=comparison["config"]["device"])
        report["artifact_dir"] = str(artifact_dir)
        report["example_training_row"] = training_row
        report["environment"] = {name: importlib.metadata.version(name)
                                 for name in ("torch", "numpy", "scikit-learn", "fastapi")}
        report["training_code_sha256"] = {}
        import hashlib
        root = Path(__file__).resolve().parent
        for folder in ("ga", "training", "data", "utils", "models"):
            for path in sorted((root / folder).rglob("*.py")):
                report["training_code_sha256"][path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["provenance_note"] = ("Source hashes are captured at packaging; per-search metadata records the original Git revision. "
                                     "Use the same training implementation throughout a suite.")
        atomic_json(report_path, report)
        atomic_json(comparison_path.parent / "delivery.json", {"report": str(Path(report_path).resolve()),
                    "artifact": str(artifact_dir), "selected_model": selected["model_id"], "test_evaluated": True})
        print(f"Report: {Path(report_path).resolve()}")
        print(f"Model artifact: {artifact_dir}")
        print(f"Selected {selected['method']} seed {selected['search_seed']} by full validation: {selected['accuracy']:.4f}")
        print(f"Held-out test accuracy: {result['test_accuracy']:.4f}")
        return report
    finally:
        lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", help="Package an already completed comparison without rerunning searches")
    parser.add_argument("--resume", help="Resume a comparison, then package its completed results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", choices=["covertype", "breast_cancer_wisconsin"], default="covertype")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--proxy-epochs", type=int, default=10)
    parser.add_argument("--full-epochs", type=int, default=20)
    parser.add_argument("--report", default="docs/industry-results.json")
    parser.add_argument("--artifact-root", default="artifacts/industry")
    args = parser.parse_args()
    if args.comparison and args.resume:
        parser.error("Choose --comparison or --resume")
    if args.resume:
        from run_comparison import resume_comparison
        resume_comparison(args.resume)
        path = args.resume
    elif args.comparison:
        path = args.comparison
    else:
        from run_comparison import compare
        from datetime import datetime
        from uuid import uuid4
        root = Path("experiments/tabular/industry-suite") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        compare(["ga", "aging", "random"], args.seeds, budget=args.budget, population=args.population,
                proxy_epochs=args.proxy_epochs, full_epochs=args.full_epochs, full_seeds=[101],
                device=args.device, dataset=args.dataset, out_dir=root)
        path = next(root.glob("*/comparison.json"))
    finalize_comparison(path, args.report, args.artifact_root)


if __name__ == "__main__":
    main()
