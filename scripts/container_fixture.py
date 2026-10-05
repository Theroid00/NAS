"""Generate a tiny offline CPU artifact only for container integration testing."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ga.chromosome import NUM_GENES
from training.trainer import full_train
from make_example import make_example
from serving.artifact import export_artifact, Predictor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True)
args = parser.parse_args()
root = Path(args.out)
root.mkdir(parents=True, exist_ok=True)
result = full_train([0] * NUM_GENES, epochs=1, device="cpu", evaluate_test=False, save_dir=root / "training")
example = root / "example.json"
make_example(result["results_path"], example)
artifact = root / "artifact"
export_artifact(result["results_path"], artifact, example)
prediction = Predictor(artifact).predict(json.loads(example.read_text())["features"])
(root / "expected.json").write_text(json.dumps(prediction), encoding="utf-8")
