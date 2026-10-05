"""Versioned, checksum-verified distribution of a saved inference artifact."""
import hashlib
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from serving.artifact import Predictor
from utils.persistence import atomic_json

FILES = {"manifest.json", "model.pt", "example.json"}
MAX_BYTES = 50 * 1024 * 1024


def package(artifact, output):
    artifact, output = Path(artifact), Path(output).resolve()
    predictor = Predictor(artifact)
    if not predictor.manifest.get("example_available"):
        raise ValueError("Distribution requires a bundled example")
    if output.exists() or output.with_suffix(".release.json").exists():
        raise FileExistsError("Release output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        archive = Path(temporary) / "model.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in sorted(FILES):
                info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(info, (artifact / name).read_bytes())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.rename(output)
    release = {"bundle_version": 1, "model_id": predictor.manifest["model_id"],
               "archive": output.name, "sha256": digest, "bytes": output.stat().st_size,
               "dataset": predictor.manifest["dataset_name"],
               "validation_metrics": predictor.manifest.get("validation_metrics"),
               "test_metrics": predictor.manifest.get("test_metrics"),
               "source_fingerprint": predictor.manifest.get("source_fingerprint")}
    atomic_json(output.with_suffix(".release.json"), release)
    return release


def install(archive, destination, sha256):
    archive, destination = Path(archive), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Artifact destination already exists: {destination}")
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256.lower()):
        raise ValueError("Provide a complete SHA256 digest")
    if archive.stat().st_size > MAX_BYTES:
        raise ValueError("Bundle exceeds the supported size")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != sha256.lower():
        raise ValueError("Bundle checksum differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) != len(FILES) or {m.filename for m in members} != FILES:
                raise ValueError("Bundle must contain exactly manifest.json, model.pt, and example.json")
            if sum(m.file_size for m in members) > MAX_BYTES:
                raise ValueError("Expanded bundle exceeds the supported size")
            for member in members:
                # Fixed filenames only; never use extractall on downloaded archives.
                (staging / member.filename).write_bytes(bundle.read(member))
        Predictor(staging)
        staging.rename(destination)
    return destination


def download(url, destination, sha256):
    if not url.startswith("https://"):
        raise ValueError("Download URL must use HTTPS")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "model.zip"
        with urllib.request.urlopen(url, timeout=30) as response, archive.open("wb") as stream:
            if not response.geturl().startswith("https://"):
                raise ValueError("Download redirected outside HTTPS")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("Download exceeds the supported size")
                stream.write(chunk)
        return install(archive, destination, sha256)
