"""Validate saved architecture provenance before launching expensive training."""
from pathlib import Path
from ga.chromosome import decode, SCHEMA_VERSION, SEARCH_SPACE
from utils.records import read_record


def load_winner(path):
    record = read_record(path)
    if record.get("smoke") or record.get("status") == "failed" or record.get("hyperparams", {}).get("injected_evaluator"):
        raise ValueError("Smoke/failed results cannot be used for full training")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported architecture schema; migrate explicitly before training")
    if "search_space" in record and record["search_space"] != SEARCH_SPACE:
        raise ValueError("Saved search space differs from this implementation")
    from data.specs import DATASETS
    if record.get("dataset_name") not in DATASETS:
        raise ValueError("Saved winner belongs to a different dataset")
    arch = decode(record["best_chromosome"])
    if "best_arch" in record and record["best_arch"] != arch:
        raise ValueError("Saved chromosome and architecture do not agree")
    record["best_arch"] = arch
    return record


def latest_winner(directory):
    for path in sorted(Path(directory).glob("best_*.json"), reverse=True):
        try:
            load_winner(path)
            return str(path)
        except (ValueError, KeyError, TypeError):
            continue
    raise FileNotFoundError(f"No compatible real search winner in {directory}")
