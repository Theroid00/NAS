"""Exercise the running container through HTTP using its fixture's expected output."""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fixture", required=True)
parser.add_argument("--url", default="http://127.0.0.1:8000")
args = parser.parse_args()
root = Path(args.fixture)

def request(route, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    query = urllib.request.Request(args.url + route, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(query, timeout=5) as response:
        return json.load(response)

for attempt in range(30):
    try:
        health = request("/health")
        assert health["model_loaded"] and health["device"] == "cpu"
        break
    except (OSError, AssertionError):
        if attempt == 29:
            raise
        time.sleep(1)
expected = json.loads((root / "expected.json").read_text())
example = json.loads((root / "example.json").read_text())
actual = request("/predict", example)
assert request("/model")["model_id"] == expected["model_id"] == actual["model_id"]
assert actual["device"] == "cpu"
for got, want in zip(actual["predictions"], expected["predictions"], strict=True):
    assert got["class_id"] == want["class_id"]
    assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got["probabilities"], want["probabilities"], strict=True))
try:
    request("/predict", {"features": [[True] * 30]})
except urllib.error.HTTPError as error:
    assert error.code == 422
else:
    raise AssertionError("Invalid input was accepted")
print("Container readiness, metadata, prediction parity, and invalid-input checks passed")
