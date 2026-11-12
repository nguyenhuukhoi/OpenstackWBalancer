"""Local coordinator lock and versioned maintenance checkpoints.

The lock is deliberately independent of state/profile/aggregate selection. This
release supports ONE deployment coordinator; it is not a distributed lease.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

SCHEMA_VERSION = 1


def package_digest():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)).replace("\\", "/"): file_digest(p)
                   for p in sorted(root.rglob("*.py"))})


from compute_maintenance.domain.errors import StateError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    try:
        with open(path, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest()
    except OSError as exc:
        raise StateError(f"cannot fingerprint required file {path}: {exc}") from exc




class HostState:
    def __init__(self, state_dir, host):
        if not host or host in (".", "..") or any(c in host for c in ("/", "\\", "\x00")):
            raise StateError("invalid compute host name for checkpoint")
        self.path = os.path.join(state_dir, host + ".json")
        self.data = {}

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as stream:
                data = json.load(stream)
        except FileNotFoundError:
            data = {}
        except (OSError, ValueError) as exc:
            raise StateError(f"cannot read checkpoint {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise StateError(f"checkpoint {self.path} must be an object")
        self.data = data
        return data

    def save(self, **updates):
        data = dict(self.data, **updates)
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            if os.name == "posix":
                directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            self.data = data
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def validate_identity(self, identity):
        if self.data.get("schema_version") != SCHEMA_VERSION:
            raise StateError("legacy/unknown checkpoint schema; reconcile the host before starting v1")
        if self.data.get("identity") != identity:
            old = self.data.get("identity") or {}
            changed = sorted(k for k in set(old) | set(identity) if old.get(k) != identity.get(k))
            raise StateError("checkpoint identity changed: " + ", ".join(changed))
        if not self.data.get("run_id"):
            raise StateError("checkpoint has no run_id")
