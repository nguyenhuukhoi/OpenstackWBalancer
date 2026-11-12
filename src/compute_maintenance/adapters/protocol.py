"""Versioned, invocation-scoped control result; JSONL is audit only."""
import json
import os
import tempfile
from contextvars import ContextVar
from pathlib import Path

from compute_maintenance.adapters.persistence import StateError

SCHEMA_VERSION = 1
current_result = ContextVar("drain_result", default=None)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".result-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class DrainResultWriter:
    def __init__(self, path, identity):
        self.path, self.identity = path, identity
        self.summary = None
        self.moves = []

    def record(self, event):
        if event.get("drain_host") == self.identity["host"]:
            self.summary = dict(event)
        elif event.get("result") == "migration_success":
            self.moves.append(event.get("vm_id"))

    def finish(self, exit_code):
        atomic_json(self.path, {"schema_version": SCHEMA_VERSION, **self.identity,
            "exit_code": exit_code, "summary": self.summary, "moved_vm_ids": self.moves})


def record_event(event):
    writer = current_result.get()
    if writer is not None:
        writer.record(event)


def read_result(path, expected, exit_code):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StateError("child result missing or invalid; outcome UNKNOWN") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise StateError("unsupported child result schema")
    if any(data.get(k) != v for k, v in expected.items()) or data.get("exit_code") != exit_code:
        raise StateError("child result identity/mode/exit mismatch")
    if exit_code in (0, 2):
        summary = data.get("summary")
        if not isinstance(summary, dict) or summary.get("drain_host") != expected["host"]:
            raise StateError("child result lacks matching drain summary")
        allowed = ({"drain_dry_run", "drain_noop"} if expected["mode"] == "dry-run"
                   else {"drain_completed", "drain_noop"})
        if exit_code == 2:
            allowed.add("drain_infeasible")
        if summary.get("result") not in allowed:
            raise StateError("invalid child outcome")
        if type(summary.get("drain_feasible")) is not bool:
            raise StateError("invalid child feasibility field")
        for key in ("drain_blocked_vm_ids", "drain_unmovable_vm_ids", "drain_infeasible_reasons"):
            values = summary.get(key)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise StateError("invalid child summary field: " + key)
        if exit_code == 0 and (not summary["drain_feasible"] or summary["drain_blocked_vm_ids"]
                or summary["drain_unmovable_vm_ids"] or summary["drain_infeasible_reasons"]):
            raise StateError("successful child result contradicts drain summary")
    return data
