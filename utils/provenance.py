"""Fingerprint source files before experiments, including uncommitted changes."""
import hashlib
import json
from pathlib import Path

SOURCE_FOLDERS = ("data", "ga", "models", "training", "utils")
SOURCE_SCRIPTS = ("main.py", "run_comparison.py", "evaluate_best.py", "calibrate_proxy.py",
                  "run_ablations.py", "run_hyper_sweep.py")


def source_fingerprint(root=None):
    """Stable relative-file hashes plus a digest; ignores outputs, docs, and serving UI."""
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    paths = [path for folder in SOURCE_FOLDERS for path in (root / folder).rglob("*.py")]
    paths.extend(root / name for name in SOURCE_SCRIPTS if (root / name).is_file())
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(paths)}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"version": 1, "sha256": hashlib.sha256(encoded).hexdigest(), "files": files}
