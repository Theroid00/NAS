"""Readiness probe: a configured model must have loaded successfully."""
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
    info = json.load(response)
    if info.get("status") != "ok" or not info.get("model_loaded"):
        raise SystemExit("Model is not ready")
