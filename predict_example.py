"""Predict from a bundled model example without a dataset download or training."""
import argparse
import json
from pathlib import Path
from serving.artifact import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--input", help="Optional raw request JSON; defaults to the bundled example")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    path = Path(args.input) if args.input else Path(args.artifact) / "example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(Predictor(args.artifact, args.device).predict(payload["features"]), indent=2))
