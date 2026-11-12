"""Durable migration intent and conservative metric-settle recovery.

Unknown outcomes stop new submissions. They never become permission to retry.

Evidence used to close an intent (``reconcile``):

* GET /servers/{id}/migrations lists only *in-progress* live migrations, so a
  failed migration is invisible there (lab, VM 2c282511 / migration 382).
* GET /os-migrations?instance_uuid= keeps every row, terminal ones included.
  Migration ids are database auto-increment ids, so "rows with an id above the
  highest id seen just before submit" are exactly the rows of this operation,
  independent of controller/coordinator clocks.
"""
import logging
import math
import os
import re
import time
import uuid
from datetime import UTC, datetime

from compute_maintenance.adapters.persistence import HostState
from compute_maintenance.domain.errors import ConfigError, MigrationExecutionError
from compute_maintenance.domain.models import get_resource_field
from compute_maintenance.domain.server import get_server_current_host

ACTIVE_MIGRATION_STATUSES = frozenset((
    "running", "accepted", "preparing", "pre-migrating", "post-migrating",
    "migrating", "queued", "running (post-copy)", "waiting-for-vif-plug",
    "confirming", "reverting",
))
FAILED_MIGRATION_STATUSES = frozenset(("failed", "error", "cancelled", "canceled"))
DONE_MIGRATION_STATUSES = frozenset(("completed", "confirmed", "finished", "done"))
TERMINAL_MIGRATION_STATUSES = FAILED_MIGRATION_STATUSES | DONE_MIGRATION_STATUSES | {"reverted"}

# Outcomes ``reconcile`` can prove.
OUTCOME_SUCCESS = "success"
OUTCOME_FAILED = "failed"
OUTCOME_NOT_STARTED = "not_started"
OUTCOME_VM_DELETED = "vm_deleted"
OUTCOME_IN_PROGRESS = "in_progress"


class MigrationStillInProgress(MigrationExecutionError):
    """The journalled migration is still running; wait, never resubmit."""


def migration_timestamp(migration):
    """Epoch seconds of a migration row (Nova reports naive UTC)."""
    stamp = get_resource_field(migration, "updated_at", "updated", "created_at")
    if not stamp:
        raise MigrationExecutionError("migration history lacks timestamps; cannot establish metric freshness")
    try:
        dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))  # noqa: FURB162 - rc11 parsing kept as is
    except ValueError as exc:
        raise MigrationExecutionError("invalid migration history timestamp") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def migration_status(migration):
    return str(get_resource_field(migration, "status") or "").lower()


def migration_server_id(migration):
    return get_resource_field(migration, "server_id", "instance_uuid")


def scan_host_migrations(conn, hosts=None, with_signature=False):
    """Return (active, latest_terminal_timestamp, terminal_rows) for ``hosts``.

    Nova's host filter includes both source and destination. No changes-since
    filter: an old, still-running operation must still block. With
    ``with_signature`` a fourth value summarises the running rows (id, status,
    updated_at) so callers can tell progress from a stall.
    """
    active = False
    latest = None
    terminal = []
    running = set()
    queries = ({"host": host} for host in sorted(set(hosts))) if hosts is not None else ({},)
    for query in queries:
        for migration in conn.compute.migrations(**query):
            status = migration_status(migration)
            if status in ACTIVE_MIGRATION_STATUSES:
                active = True
                running.add((str(get_resource_field(migration, "id")), status,
                             str(get_resource_field(migration, "updated_at"))))
                continue
            if status not in TERMINAL_MIGRATION_STATUSES:
                raise MigrationExecutionError("UNKNOWN Nova migration status: " + status)
            stamp = migration_timestamp(migration)
            latest = stamp if latest is None else max(latest, stamp)
            terminal.append((migration, stamp))
    if with_signature:
        return active, latest, terminal, tuple(sorted(running))
    return active, latest, terminal


def _created_at(migration):
    stamp = get_resource_field(migration, "created_at")
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))  # noqa: FURB162 - rc11 parsing kept as is
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def recent_arrivals(terminal_rows, now, window):
    """Successful migrations that completed within ``window`` seconds.

    Returned as dicts {vm_id, src_host, dst_host, age}; the caller decides how
    much of each VM's load the destination's metrics still miss.
    """
    arrivals = {}
    for migration, stamp in terminal_rows:
        if migration_status(migration) not in DONE_MIGRATION_STATUSES:
            continue
        age = now - stamp
        if age < 0 or age >= window:
            continue
        vm_id = migration_server_id(migration)
        dst = get_resource_field(migration, "dest_compute")
        if not vm_id or not dst:
            continue
        previous = arrivals.get(vm_id)
        if previous is None or age < previous["age"]:
            arrivals[vm_id] = {
                "vm_id": vm_id,
                "src_host": get_resource_field(migration, "source_compute"),
                "dst_host": dst,
                "age": age,
                # Absolute completion time: "age" goes stale as soon as the
                # caller spends time on further queries.
                "done_at": stamp,
                "begin_at": _created_at(migration),
            }
    return sorted(arrivals.values(), key=lambda item: (item["age"], item["vm_id"]))


def _is_not_found(exc):
    name = type(exc).__name__
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    return name in ("NotFoundException", "ResourceNotFound") or str(status) == "404"


class MigrationJournal:
    def __init__(self, directory, settle_seconds=360, clock=time.time, sleeper=time.sleep,
                 not_started_grace=None, active_timeout=None, stall_timeout=None, max_wait=None):
        self.state = HostState(directory, "migration-journal")
        self.clock, self.sleep = clock, sleeper
        self.settle_seconds = settle_seconds
        from compute_maintenance import config as _config
        self.active_timeout = _config.runtime().migration_timeout if active_timeout is None else active_timeout
        # One no-progress window (MIGRATION_TIMEOUT); stall_timeout only for tests.
        self.stall_timeout = self.active_timeout if stall_timeout is None else stall_timeout
        self.max_wait = max(self.active_timeout,
                            _config.runtime().migration_max_wait if max_wait is None else max_wait)
        if not_started_grace is None:
            not_started_grace = _config.MIGRATION_NOT_STARTED_GRACE_SECONDS
        self.not_started_grace = not_started_grace
        saved = self.state.load()
        if saved or os.path.exists(self.state.path):
            stamp = saved.get("last_completion")
            pending = saved.get("pending")
            if (saved.get("schema_version") != 1 or "pending" not in saved
                    or type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0):
                raise MigrationExecutionError("invalid migration journal schema/completion timestamp")
            if pending is not None and (
                not isinstance(pending, dict)
                or any(not isinstance(pending.get(key), str) or not pending[key]
                       for key in ("operation_id", "vm_id", "src_host", "dst_host"))
                or not isinstance(pending.get("preexisting_ids"), list)
                or any(not isinstance(value, str) for value in pending["preexisting_ids"])
                or type(pending.get("submitted_at")) not in (int, float)
                or not math.isfinite(pending["submitted_at"]) or pending["submitted_at"] < 0
                or ("global_max_id" in pending
                    and (type(pending["global_max_id"]) is not int or pending["global_max_id"] < 0))
            ):
                raise MigrationExecutionError("invalid pending migration journal intent")
        if not saved:
            # A new installation cannot know about migrations before it started.
            self.state.save(schema_version=1, pending=None, last_completion=clock())

    # ------------------------------------------------------------ evidence
    def _operation_rows(self, conn, pending):
        """Global /os-migrations rows that belong to the pending operation."""
        from compute_maintenance.adapters.openstack import server_migration_rows
        rows = []
        prior = set(pending.get("preexisting_ids") or [])
        floor = pending.get("global_max_id")
        for row in server_migration_rows(conn, pending["vm_id"]):
            kind = str(get_resource_field(row, "migration_type") or "live-migration").lower()
            if kind != "live-migration":
                continue
            row_id = get_resource_field(row, "id")
            if str(row_id) in prior:
                continue
            if floor is not None:
                try:
                    if int(row_id) <= floor:
                        continue
                except (TypeError, ValueError):
                    continue
            else:
                # Journals written before global_max_id existed: fall back to
                # the creation time, with a small allowance for clock skew.
                created = get_resource_field(row, "created_at")
                try:
                    created_ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))  # noqa: FURB162 - as above
                    if created_ts.tzinfo is None:
                        created_ts = created_ts.replace(tzinfo=UTC)
                    if created_ts.timestamp() < pending["submitted_at"] - 5:
                        continue
                except ValueError:
                    continue
            rows.append(row)
        return rows

    def classify_pending(self, conn):
        """Return (outcome, detail) for the pending intent without changing it."""
        pending = self.state.data.get("pending")
        if not pending:
            return None, "no pending operation"
        try:
            vm = conn.compute.get_server(pending["vm_id"])
        except Exception as exc:  # only a 404 is evidence; anything else is re-raised
            if _is_not_found(exc):
                return OUTCOME_VM_DELETED, "server no longer exists"
            raise
        host = get_server_current_host(vm)
        status = str(get_resource_field(vm, "status") or "").upper()
        task = get_resource_field(vm, "task_state", "OS-EXT-STS:task_state")
        task = None if task in ("", "None") else task
        if status == "ACTIVE" and host == pending["dst_host"] and not task:
            return OUTCOME_SUCCESS, "server ACTIVE on destination"

        rows = self._operation_rows(conn, pending)
        statuses = sorted({migration_status(row) for row in rows})
        ids = sorted(str(get_resource_field(row, "id")) for row in rows)
        if task or any(s in ACTIVE_MIGRATION_STATUSES for s in statuses):
            return OUTCOME_IN_PROGRESS, f"task_state={task} migrations={ids} {statuses}"
        if rows and all(s in FAILED_MIGRATION_STATUSES for s in statuses):
            if host == pending["src_host"]:
                return OUTCOME_FAILED, f"migration(s) {ids} {statuses}; server {status} on source"
            return None, f"migration(s) {ids} failed but server is on {host} (neither source nor destination)"
        if rows and any(s in DONE_MIGRATION_STATUSES for s in statuses):
            if host == pending["dst_host"]:
                return OUTCOME_SUCCESS, f"migration(s) {ids} {statuses}; server {status} on destination"
            return None, f"migration(s) {ids} completed but server is on {host}"
        if not rows:
            age = self.clock() - pending["submitted_at"]
            if host == pending["src_host"] and status == "ACTIVE":
                if age >= self.not_started_grace:
                    return OUTCOME_NOT_STARTED, f"no Nova migration record {age:.0f}s after submit"
                return OUTCOME_IN_PROGRESS, f"no Nova migration record yet ({age:.0f}s after submit)"
        return None, f"migration(s) {ids} {statuses}; server {status} on {host} task={task}"

    def reconcile(self, conn):
        """Close the pending intent when Nova proves its outcome.

        Returns the outcome ("success", "failed", "not_started", "vm_deleted"),
        or None when nothing was pending. Raises MigrationStillInProgress while
        the operation runs, and MigrationExecutionError when Nova cannot prove
        the outcome (the journal is retained; no new submit is allowed).
        """
        pending = self.state.data.get("pending")
        if not pending:
            return None
        outcome, detail = self.classify_pending(conn)
        if outcome in (OUTCOME_SUCCESS, OUTCOME_FAILED, OUTCOME_NOT_STARTED, OUTCOME_VM_DELETED):
            logging.info("Migration journal %s for VM %s reconciled: %s (%s)",
                         pending["operation_id"], pending["vm_id"], outcome, detail)
            self.finish(outcome, detail=detail)
            return outcome
        if outcome == OUTCOME_IN_PROGRESS:
            raise MigrationStillInProgress(
                f"migration {pending['operation_id']} for {pending['vm_id']} still in progress: {detail}"
            )
        raise MigrationExecutionError(
            f"UNKNOWN migration {pending['operation_id']} for {pending['vm_id']}: {detail}; "
            "reconcile Nova before any new submit (journal retained)"
        )

    def begin(self, step, preexisting_ids, global_max_id=None):
        if self.state.data.get("pending"):
            raise MigrationExecutionError("pending migration must be reconciled first")
        intent = {"operation_id": uuid.uuid4().hex, "vm_id": step.vm_id,
                  "src_host": step.src_host, "dst_host": step.dst_host, "submitted_at": self.clock(),
                  "preexisting_ids": sorted(str(x) for x in preexisting_ids)}
        if global_max_id is not None:
            intent["global_max_id"] = int(global_max_id)
        self.state.save(pending=intent)

    def finish(self, outcome, detail=None):
        self.state.save(pending=None, last_outcome=outcome, last_outcome_detail=detail,
                        last_completion=self.clock())

    # --------------------------------------------------------------- settle
    def _timer(self):
        from compute_maintenance.domain.errors import ProgressTimer
        return ProgressTimer(self.active_timeout, self.stall_timeout, self.max_wait, self.clock)

    def _pending_progress(self, conn):
        """Progress signature of the journalled migration (per-server counters)."""
        pending = self.state.data.get("pending") or {}
        try:
            rows = list(conn.compute.server_migrations(pending.get("vm_id")))
        except Exception:  # noqa: BLE001 - progress evidence is best effort
            return None
        return tuple(sorted(tuple(str(get_resource_field(r, name)) for name in (
            "id", "status", "memory_processed_bytes", "memory_remaining_bytes",
            "disk_processed_bytes", "updated_at")) for r in rows))

    def _reconcile_waiting(self, conn, timer):
        """Reconcile, waiting while our own operation still runs and progresses."""
        while True:
            try:
                return self.reconcile(conn)
            except MigrationStillInProgress as exc:
                timer.observe((str(exc), self._pending_progress(conn)))
                reason = timer.expired()
                if reason:
                    raise MigrationExecutionError(
                        f"journalled migration still in progress after {timer.elapsed():.0f}s ({reason}); no new submit "
                        f"permitted: {exc}") from None
                logging.info("Waiting for the journalled migration to finish: %s", exc)
                self.sleep(10)

    def _wait_no_active(self, conn, hosts, timer):
        while True:
            active, latest, terminal, signature = scan_host_migrations(conn, hosts, with_signature=True)
            if not active:
                return latest, terminal
            # Nova may expose ACTIVE on the destination before post-migration
            # cleanup commits the terminal migration row (observed on the lab).
            timer.observe(signature)
            reason = timer.expired()
            if reason:
                raise MigrationExecutionError(
                    f"Nova migration remained in progress for {timer.elapsed():.0f}s ({reason}); no new submit permitted")
            logging.info("Waiting for Nova migration records to become terminal before planning")
            self.sleep(10)

    def wait_settled(self, conn, active_deadline=None, hosts=None):
        """Barrier: wait a full query window after local OR external migrations.

        The caller must collect a fresh snapshot after this barrier. No estimated
        load is added, so there is no double counting as metrics converge.
        """
        timer = self._timer()
        while True:
            self._reconcile_waiting(conn, timer)
            latest_nova, _terminal = self._wait_no_active(conn, hosts, timer)
            latest = float(self.state.data.get("last_completion", self.clock()))
            if latest_nova is not None:
                latest = max(latest, latest_nova)
            delay = latest + self.settle_seconds - self.clock()
            if delay > self.settle_seconds + 60:
                raise MigrationExecutionError("migration timestamp is in the future; check coordinator/API clocks")
            if delay <= 0:
                return
            logging.info("Waiting %.1fs for migration metric window to settle", delay)
            self.sleep(delay)
            # An external migration may have started while this process waited.

    def settle_for_compensation(self, conn, hosts, window):
        """Drain mode: reconcile and wait for running migrations, but do not
        wait out the metric window. Returns recent arrivals (see
        ``recent_arrivals``) whose load the caller adds to the destinations.
        """
        timer = self._timer()
        self._reconcile_waiting(conn, timer)
        latest, terminal = self._wait_no_active(conn, hosts, timer)
        now = self.clock()
        if latest is not None and latest - now > 60:
            raise MigrationExecutionError("migration timestamp is in the future; check coordinator/API clocks")
        return recent_arrivals(terminal, now, window)


_DURATION_UNITS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}
_DURATION_PART = re.compile(r"(\d+)(ms|[smhdwy])")
_DURATION = r"(?:\d+(?:ms|[smhdwy]))+"


def parse_promql_duration(text):
    """Seconds of a PromQL duration such as 5m, 1h5m or 10m30s."""
    total = 0.0
    for amount, unit in _DURATION_PART.findall(text):
        total += int(amount) * _DURATION_UNITS[unit]
    return total


def promql_lookback_seconds(query):
    """Largest range plus largest offset used anywhere in ``query``.

    Compound durations ([10m30s]), subqueries ([5m:30s]) and ``offset`` all
    widen the window of samples a value depends on.
    """
    ranges = [parse_promql_duration(m) for m in re.findall(r"\[(" + _DURATION + r")(?::" + _DURATION + r")?\]", query)]
    ranges += [parse_promql_duration(m) for m in re.findall(r"\[(" + _DURATION + r"):\]", query)]
    offsets = [parse_promql_duration(m) for m in re.findall(r"\boffset\s+(" + _DURATION + r")", query)]
    return (max(ranges) if ranges else 0.0) + (max(offsets) if offsets else 0.0)


def validate_settle_window(cfg, seconds):
    windows = [300.0]
    for name, value in vars(cfg).items():
        if name.startswith("prometheus_query") and isinstance(value, str):
            windows.append(promql_lookback_seconds(value))
    if seconds < max(windows) + 60:
        raise ConfigError("runtime settle must cover the largest PromQL range plus 60s scrape margin")
