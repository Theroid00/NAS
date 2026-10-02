"""Atomic JSON snapshots and process-scoped run locks."""
import json
import os
from pathlib import Path
from uuid import uuid4


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunLock:
    """OS releases this lock even when the process is killed; retain the lock file."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def acquire(self):
        self.file = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    self.file.write(b"\0")
                    self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            self.file = None
            raise RuntimeError(f"Run is already active: {self.path}") from error

    def release(self):
        if self.file is not None:
            self.file.close()
            self.file = None
