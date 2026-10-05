"""Record-relative references on disk, resolved paths at Python API boundaries."""
import json
import os
from pathlib import Path, PureWindowsPath

from utils.persistence import atomic_json

PATH_KEYS = {"trial_path", "metadata_path", "save_path", "log_path", "checkpoint_path",
             "results_path", "winner_path", "save_dir"}


def _map_paths(value, transform):
    if isinstance(value, dict):
        return {key: transform(item) if key in PATH_KEYS and isinstance(item, str) else
                _map_paths(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_paths(item, transform) for item in value]
    return value


def read_record(path, legacy_root=None, relocated_root=None):
    """Read new relative records or legacy records; relocation requires explicit roots."""
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    version = record.get("path_format")
    if version not in (None, 1):
        raise ValueError(f"Unsupported record path format: {version}")
    if (legacy_root is None) != (relocated_root is None):
        raise ValueError("Provide both legacy and relocated roots")

    def resolve(reference):
        if not reference:
            return reference
        portable = reference.replace("\\", "/")
        windows_absolute = PureWindowsPath(reference).is_absolute()
        target = Path(portable)
        if legacy_root is not None and (version is None or target.is_absolute() or windows_absolute):
            # PureWindowsPath handles Windows-origin records on Linux too.
            kind = PureWindowsPath if windows_absolute else Path
            try:
                original = kind(reference) if windows_absolute else target.resolve()
                relative = original.relative_to(kind(str(legacy_root)))
            except ValueError as error:
                raise ValueError(f"Reference is outside the explicit legacy root: {reference}") from error
            return str((Path(relocated_root) / Path(*relative.parts)).resolve())
        if windows_absolute and os.name != "nt":
            raise ValueError("Legacy Windows paths require explicit relocation roots on this platform")
        if version == 1 and not target.is_absolute():
            return str((path.parent / target).resolve())
        return str(target.resolve())

    result = _map_paths(record, resolve)
    result.pop("path_format", None)
    return result


def write_record(path, value):
    """Persist known references relative to this JSON file without mutating callers."""
    path = Path(path).resolve()

    def encode(reference):
        if not reference:
            return reference
        try:
            return Path(os.path.relpath(Path(reference).resolve(), path.parent)).as_posix()
        except ValueError as error:
            raise ValueError("Record references must share a filesystem drive; keep experiment outputs together") from error

    record = _map_paths(value, encode)
    record["path_format"] = 1
    atomic_json(path, record)
