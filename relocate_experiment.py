"""Copy an experiment and rewrite its references using an explicit original root."""
import argparse
from pathlib import Path
import shutil
import tempfile

from utils.records import read_record, write_record


def relocate(source, destination, legacy_root=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError("Source must be an experiment directory")
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be outside the source experiment")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Experiment relocation does not follow symbolic links")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".relocate-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "experiment"
        shutil.copytree(source, staging)
        for path in staging.rglob("*.json"):
            if path.name == "comparison.json" or path.name.startswith(("metadata_", "best_", "full_train_")):
                record = read_record(path, legacy_root=legacy_root or source, relocated_root=staging)
                write_record(path, record)
        staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--legacy-root", help="Explicit original root for legacy absolute paths, including Windows-origin records")
    args = parser.parse_args()
    print(relocate(args.source, args.destination, args.legacy_root))


if __name__ == "__main__":
    main()
