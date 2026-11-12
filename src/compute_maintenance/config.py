"""Configuration: decision thresholds, config keys, parsing and validation.

Which layer is canonical (rc12)
-------------------------------
The live decision values of this process (thresholds, drain limits,
soft-anti-affinity heal, migration waits) are ONE frozen object,
:class:`RuntimeSettings`, read at call time through :func:`runtime`. The CLI
replaces it once per process after loading the config, through
:func:`apply_thresholds`, :func:`apply_soft_aa_heal` and
:func:`apply_migration_timeouts` (all built on :func:`set_runtime`).

The upper-case module constants of the same values (``RAM_HOTSPOT_THRESHOLD``
... ``MIGRATION_MAX_WAIT``, see ``RUNTIME_CONSTANTS``) are the built-in
defaults and a thin compatibility layer: every replacement of the runtime
settings rewrites them, and assigning one of them (older code, tests via
monkeypatch) is forwarded into :func:`runtime`, so the two views never
disagree. New code reads ``runtime().<field>``. Never
``from compute_maintenance.config import <CONSTANT>``: that copies the value at
import time and silently ignores the operator's config.
"""
import dataclasses
import json
import logging
import math
import os
import sys
import types

import compute_maintenance.domain.errors as _errors
import compute_maintenance.paths as _paths
from compute_maintenance.adapters.settings import (
    configuration_scope,
    getenv,
    read_settings,
)

MIN_IMPROVEMENT_THRESHOLD = 0.15


DEFAULT_CONFIG_FILE = _paths.CONFIG_FILE   # $WLB_CONFIG_PATH replaces it as the CLI default


PROMETHEUS_NODE_JOB = None


PROMETHEUS_LIBVIRT_JOB = None


# Keys accepted by older configs but no longer read. Scheduler capacity and
# allocation ratios come from Placement, so no openstack-exporter job is needed.
OBSOLETE_CONFIG_KEYS = (
    "PROMETHEUS_OPENSTACK_EXPORTER_JOB",
    "PROMETHEUS_CPU_RATIO",
    "PROMETHEUS_MEM_RATIO",
)


def require_job(value, name):
    value = (value or '').strip()
    if not value:
        raise _errors.ConfigError(f"{name} is required; set it in {DEFAULT_CONFIG_FILE} (or the selected --config file).")
    if any(c in value for c in '\"\\\n\r{}'):
        raise _errors.ConfigError(f"{name} must be a plain Prometheus job name, got {value!r}.")
    return value


def build_default_queries(node_job, libvirt_job):
    """Default PROMETHEUS_* queries for the given job names."""
    node_job = require_job(node_job, 'PROMETHEUS_NODE_JOB')
    libvirt_job = require_job(libvirt_job, 'PROMETHEUS_LIBVIRT_JOB')
    return {
        # Busy % = 1 - idle time / time of ALL modes, both from the same two
        # scrapes. "100 - avg(irate(idle))" divides by the timestamp gap
        # instead, and Prometheus stamps a sample when the scrape STARTS, not
        # when node_exporter reads /proc/stat: with scrape latency jitter an
        # idle host showed idle > 100% and a negative CPU (-6.9%), which made
        # the whole aggregate evaluation fail. The ratio cancels that jitter
        # and is always within 0..100.
        "cpu_used": (
            f'sort_desc(100 * (1 - sum by (alias) (irate(node_cpu_seconds_total{{job="{node_job}",mode="idle"}}[5m]))'
            f' / sum by (alias) (irate(node_cpu_seconds_total{{job="{node_job}"}}[5m]))))'
        ),
        "mem_used": (
            f'sort_desc(100 - ((avg_over_time(node_memory_MemAvailable_bytes{{job="{node_job}"}}[5m]) * 100) / avg_over_time(node_memory_MemTotal_bytes{{job="{node_job}"}}[5m])))'
        ),
        # Host-side VM RSS bytes; "__HOST_RE__" is the aggregate host regex.
        "vm_ram_rss": (
            f'avg_over_time(libvirt_domain_stat_memory_rss_bytes{{job="{libvirt_job}", host=~"__HOST_RE__"}}[5m])'
        ),
        # Drain compensation measures the RAM a host really holds now: the
        # latest raw sample (no averaging) and that sample's timestamp.
        "mem_used_instant": (
            f'100 - (node_memory_MemAvailable_bytes{{job="{node_job}"}} * 100'
            f' / node_memory_MemTotal_bytes{{job="{node_job}"}})'
        ),
        "mem_sample_timestamp": f'timestamp(node_memory_MemAvailable_bytes{{job="{node_job}"}})',
        # Runtime capacity is deliberately separate from scheduler/Placement
        # capacity: the *real* host denominator used for projecting the impact
        # of observed VM CPU/RAM usage after a migration. These values are
        # effectively static, so retain the last good sample for five minutes
        # to tolerate a brief scrape gap while still failing closed afterward.
        "logical_cpus": (
            f'count by (alias) (last_over_time(node_cpu_seconds_total{{job="{node_job}",mode="idle"}}[5m]))'
        ),
        "total_ram_bytes": (
            f'avg by (alias) (last_over_time(node_memory_MemTotal_bytes{{job="{node_job}"}}[5m]))'
        ),
    }


# Compatibility names remain importable; queries are built only after loading jobs.
_DEFAULT_QUERIES = dict.fromkeys((
    'cpu_used', 'mem_used', 'vm_ram_rss', 'logical_cpus', 'total_ram_bytes',
    'mem_used_instant', 'mem_sample_timestamp',
))


DEFAULT_PROMETHEUS_CPU_USED = _DEFAULT_QUERIES["cpu_used"]


DEFAULT_PROMETHEUS_MEM_USED = _DEFAULT_QUERIES["mem_used"]


DEFAULT_PROMETHEUS_VM_RAM_RSS = ""


DEFAULT_PROMETHEUS_VM_RAM_RSS_ENABLED_QUERY = _DEFAULT_QUERIES["vm_ram_rss"]


TARGET_AVG_MAD = 3


MIN_CPU_WEIGHT = 0.2


MIN_RAM_WEIGHT = 0.2


PRESSURE_MAD_WEIGHT = 0.7


PRESSURE_MEAN_WEIGHT = 0.3


AUTO_PRESSURE_THRESHOLD = 80.0


RAM_HOTSPOT_THRESHOLD = 80


RAM_CRITICAL_THRESHOLD = 95.0


CPU_HOTSPOT_THRESHOLD = 70.0


DRAIN_MAX_HOST_RAM_PCT = RAM_HOTSPOT_THRESHOLD


DRAIN_MAX_HOST_CPU_PCT = CPU_HOTSPOT_THRESHOLD


DRAIN_ABSOLUTE_MAX_HOST_RAM_PCT = RAM_CRITICAL_THRESHOLD


# Config keys for the decision thresholds and their built-in defaults. The
# module-level values above are the process defaults; apply_thresholds() makes
# the values loaded from the config file live (runtime(), see RuntimeSettings).
THRESHOLD_KEYS = (
    # (config key, attribute of Thresholds, built-in default)
    ("RAM_HOTSPOT_THRESHOLD", "ram_hotspot", 80.0),
    ("RAM_CRITICAL_THRESHOLD", "ram_critical", 95.0),
    ("CPU_HOTSPOT_THRESHOLD", "cpu_hotspot", 70.0),
    ("PRESSURE_THRESHOLD", "pressure", 80.0),
    ("TARGET_AVG_MAD", "target_mad", 3.0),
    ("MIN_IMPROVEMENT_THRESHOLD", "min_improvement", 0.15),
    ("PLACEMENT_PRESSURE_THRESHOLD", "placement_pressure", 90.0),
)


# rc11: soft-anti-affinity heal (DRS-style constraint correction). The module
# values are the process defaults; apply_soft_aa_heal() makes the configured
# ones live (runtime(), see RuntimeSettings).
#   HEAL_SOFT_ANTI_AFFINITY     - re-spread broken soft-anti-affinity groups
#                                 (ladder rung after the hotspots, before MAD)
#                                 and use the soft-anti-affinity tie-breaks.
#   SOFT_AA_HEAL_MAD_TOLERANCE  - heal gate, per resource: after a heal move
#                                 the CPU MAD and the RAM MAD must each stay
#                                 <= TARGET_AVG_MAD, or grow by at most this
#                                 tolerance when it was already above it.
#   SOFT_AA_TIEBREAK_MARGIN     - relative improvement band inside which a
#                                 direct move that heals a group is preferred.
HEAL_SOFT_ANTI_AFFINITY = True
SOFT_AA_HEAL_MAD_TOLERANCE = 0.05
SOFT_AA_TIEBREAK_MARGIN = 0.15

SOFT_AA_HEAL_KEYS = (
    # (config key, attribute of SoftAntiAffinityHeal, built-in default)
    ("HEAL_SOFT_ANTI_AFFINITY", "enabled", True),
    ("SOFT_AA_HEAL_MAD_TOLERANCE", "mad_tolerance", 0.05),
    ("SOFT_AA_TIEBREAK_MARGIN", "tiebreak_margin", 0.15),
)


# Cloud name in clouds.yaml (or the OS_CLOUD_NAME of an openrc environment,
# "envvars" by default in openstacksdk). Override with OS_CLOUD or --os-cloud.
DEFAULT_OS_CLOUD = "openstack"


def resolve_os_cloud(cli_value=None, config_file=None):
    """The cloud this process talks to: the one rule shared by the OpenStack
    connection, the rolling-maint parent and the drain-child result identity.

    Precedence: ``--os-cloud`` > environment ``OS_CLOUD`` > ``OS_CLOUD`` in
    ``config_file`` > :data:`DEFAULT_OS_CLOUD`. An empty ``OS_CLOUD`` in the
    environment or the file counts as unset; a value that is set but blank
    (an explicit ``--os-cloud ''`` or whitespace only) is an error.
    """
    sources = (
        ("--os-cloud", lambda: cli_value, False),
        ("OS_CLOUD in the environment", lambda: os.environ.get("OS_CLOUD"), True),
        (f"OS_CLOUD in {config_file}",
         lambda: read_settings(config_file).get("OS_CLOUD") if config_file else None, True),
    )
    for label, read, empty_is_unset in sources:
        value = read()
        if value is None or (empty_is_unset and value == ""):
            continue
        value = value.strip()
        if not value:
            raise _errors.ConfigError(f"{label} must not be empty.")
        return value
    return DEFAULT_OS_CLOUD


DRAIN_MAX_ROUND_FACTOR = 2


DRAIN_MAX_ROUND_SLACK = 3


DRAIN_TRANSIENT_UNMOVABLE_REASONS = (
    "incomplete_metrics",
    "incomplete_metadata",
    "missing_vm_details",
    # A Nova task (rebuild, snapshot, resize, ...) normally finishes on its own.
    "task_in_progress",
)


# Reasons a VM can never be planned automatically. Balance skips such VMs;
# drain reports them as unmovable so the operator moves them by hand.
#   locked               - Nova refuses admin actions while a VM is locked.
#   unsupported_resources - pinned CPU (PCPU), PCI/vGPU or custom resources that
#                          this planner does not model in Placement.
VM_UNSUPPORTED_EXTRA_SPEC_PREFIXES = ("pci_passthrough:", "resources:")
VM_UNSUPPORTED_CPU_POLICIES = ("dedicated", "mixed")


# Settle handling after a migration. Balance waits a full metric window
# (barrier). Drain does not wait: it adds the projected load of VMs that
# arrived on a host within the metric window to that host's observed load.
SETTLE_MODE_BARRIER = "barrier"
SETTLE_MODE_COMPENSATE = "compensate"


# Host CPU uses irate() (lags at most two scrapes; the window is derived from
# the node job's real scrape interval, this is only the fallback for an
# unknown interval, i.e. scrapes up to 30 s); host RAM uses a 5 minute
# avg_over_time(), so a VM that arrived t seconds ago is still missing
# (1 - t/300) of its RAM from the destination's reading.
COMPENSATION_CPU_WINDOW_SECONDS = 60.0
COMPENSATION_DEFAULT_RAM_WINDOW_SECONDS = 300.0


# Balance-mode revalidation rejected an approved move (VM locked/busy/moved
# meanwhile): keep the planner away from that VM for a short while instead of
# re-planning the very same move on every cron run.
VM_REJECTED_MOVE_COOLDOWN_SECONDS = 1800


# libvirt cpu_time covers the whole VM cgroup (vCPU, emulator and vhost
# threads), so a busy guest can legitimately read above 100% of its vCPUs.
# Values up to this sanity limit are clamped to 100%, not discarded.
VM_CPU_PCT_SANITY_MAX = 1000.0


# Grace period before "no Nova migration record at all" is accepted as proof
# that a submitted live migration never started. Nova's API sets
# task_state=migrating synchronously before casting to the conductor, so an
# accepted request is visible as a task (in progress) long before this expires;
# no row + no task + ACTIVE on source after the grace means it never ran.
MIGRATION_NOT_STARTED_GRACE_SECONDS = 120.0


# Bounded retries for idempotent OpenStack reads (list servers/migrations/...).
OPENSTACK_READ_ATTEMPTS = 3
OPENSTACK_READ_RETRY_DELAY = 5.0


DRAIN_METRIC_SETTLE_TIMEOUT = 360.0


DRAIN_METRIC_SETTLE_INTERVAL = 15.0


# A drain step refused at revalidation (nothing submitted) is re-planned from
# fresh state this many times in a row before the drain is aborted.
DRAIN_MAX_CONSECUTIVE_REFUSALS = 3
DRAIN_REFUSAL_REPLAN_DELAY = 30.0


MODE_MAD = "mad"


MODE_PRESSURE = "pressure"


MODE_RAM_HOTSPOT = "ram_hotspot"


MODE_RAM_CRITICAL_HOTSPOT = "ram_critical_hotspot"


MODE_CPU_HOTSPOT = "cpu_hotspot"


MODE_NORMAL = "normal"


# rc11 ladder rung (not a load mode): label of a soft-anti-affinity heal move,
# used as its plan type, step reason and event mode.
MODE_SOFT_AA_HEAL = "soft_aa_heal"


SOFT_AA_HEAL_PLAN_TYPE = MODE_SOFT_AA_HEAL


RAM_HOTSPOT_MODES = (MODE_RAM_HOTSPOT, MODE_RAM_CRITICAL_HOTSPOT)


HOTSPOT_MODES = (MODE_RAM_HOTSPOT, MODE_RAM_CRITICAL_HOTSPOT, MODE_CPU_HOTSPOT)


DEFAULT_EXCLUDE_VMS_FILE = ""


EXCLUDED_VM_DISPLAY_LIMIT = 5


DEFAULT_MAX_INSTANCES_PER_HOST = 0


DEFAULT_PROMETHEUS_LOGICAL_CPUS = _DEFAULT_QUERIES["logical_cpus"]


DEFAULT_PROMETHEUS_TOTAL_RAM_BYTES = _DEFAULT_QUERIES["total_ram_bytes"]


DEFAULT_PLACEMENT_VCPU_HEADROOM_PERCENT = 5.0


DEFAULT_PLACEMENT_RAM_HEADROOM_PERCENT = 5.0


PLACEMENT_PRESSURE_THRESHOLD = 90.0


DEFAULT_ENABLE_MAKE_ROOM = True


DEFAULT_MAKE_ROOM_MAX_DEPTH = 2


DEFAULT_MAKE_ROOM_MIN_IMPROVEMENT = MIN_IMPROVEMENT_THRESHOLD * DEFAULT_MAKE_ROOM_MAX_DEPTH


DEFAULT_MAKE_ROOM_MAX_VM_CPU_PERCENT = 10.0


DEFAULT_MAKE_ROOM_MAX_VM_RAM_PERCENT = 50.0


DEFAULT_PLACEMENT_FINAL_ALLOCATION_CANDIDATE_CHECK = False


DEFAULT_PLACEMENT_SYNC_TIMEOUT = 30.0


DEFAULT_PLACEMENT_SYNC_INTERVAL = 2.0


MIGRATION_RECORD_DISCOVERY_TIMEOUT = 120.0


# No built-in default: a missing URL must fail closed instead of sending
# PromQL to an unrelated (public) host name.
PROMETHEUS_QUERY_URL = None


DEFAULT_ALERT_EMAIL_TO = ""


DEFAULT_ALERT_EMAIL_FROM = "loadleveller@localhost"


DEFAULT_SMTP_SERVER = ""


DEFAULT_SMTP_PORT = 25


DEFAULT_SMTP_USER = ""


DEFAULT_SMTP_PASSWORD = ""


DEFAULT_SMTP_STARTTLS = False


POLL_INTERVAL = 2


# Waiting for a live migration (monitor) or for running migrations (journal /
# settle) is progress-aware, with one window:
#   * stop waiting once Nova reported no progress for MIGRATION_TIMEOUT seconds,
#     counted from the start or from the last progress (memory/disk counters,
#     status or row updates of this migration, or of the migration running ahead
#     of it on the same host -- max_concurrent_live_migrations queues the next one);
#   * never longer than MIGRATION_MAX_WAIT seconds in total (30 min by default).
# Stopping to wait never aborts the migration: the run records
# migration_timeout, keeps the journal pending and the next run reconciles it.
# Both are config keys (see MigrationTimeouts / apply_migration_timeouts).
# Nova's own guard (libvirt live_migration_completion_timeout, default 800 s
# per GiB, action abort) ends a hopeless migration as 'cancelled'/'error'.
MIGRATION_TIMEOUT = 600
MIGRATION_MAX_WAIT = 1800
MIGRATION_HOST_ACTIVITY_INTERVAL = 15.0

TIMEOUT_KEYS = (
    # (config key, attribute of MigrationTimeouts, built-in default, seconds)
    ("MIGRATION_TIMEOUT", "no_progress", 600.0),
    ("MIGRATION_MAX_WAIT", "maximum", 1800.0),
)
# rc8 had a separate stall window; rc9 merged it into MIGRATION_TIMEOUT.
OBSOLETE_TIMEOUT_KEYS = ("MIGRATION_STALL_TIMEOUT",)


@dataclasses.dataclass(frozen=True)
class RuntimeSettings:
    """The live decision values of this process (canonical, see the module
    docstring). Each field is the lower-case name of the module constant it
    mirrors; the defaults are those constants' built-in values."""

    ram_hotspot_threshold: float = RAM_HOTSPOT_THRESHOLD
    ram_critical_threshold: float = RAM_CRITICAL_THRESHOLD
    cpu_hotspot_threshold: float = CPU_HOTSPOT_THRESHOLD
    auto_pressure_threshold: float = AUTO_PRESSURE_THRESHOLD      # config key PRESSURE_THRESHOLD
    target_avg_mad: float = TARGET_AVG_MAD
    min_improvement_threshold: float = MIN_IMPROVEMENT_THRESHOLD
    placement_pressure_threshold: float = PLACEMENT_PRESSURE_THRESHOLD
    # Drain limits follow the hotspot thresholds (set by apply_thresholds).
    drain_max_host_ram_pct: float = DRAIN_MAX_HOST_RAM_PCT
    drain_max_host_cpu_pct: float = DRAIN_MAX_HOST_CPU_PCT
    drain_absolute_max_host_ram_pct: float = DRAIN_ABSOLUTE_MAX_HOST_RAM_PCT
    heal_soft_anti_affinity: bool = HEAL_SOFT_ANTI_AFFINITY
    soft_aa_heal_mad_tolerance: float = SOFT_AA_HEAL_MAD_TOLERANCE
    soft_aa_tiebreak_margin: float = SOFT_AA_TIEBREAK_MARGIN
    migration_timeout: float = MIGRATION_TIMEOUT
    migration_max_wait: float = MIGRATION_MAX_WAIT


# {module constant (compatibility mirror): RuntimeSettings field}
RUNTIME_CONSTANTS = {field.name.upper(): field.name for field in dataclasses.fields(RuntimeSettings)}

_runtime = RuntimeSettings()


def runtime():
    """The live :class:`RuntimeSettings` of this process; read it at call time."""
    return _runtime


def set_runtime(settings):
    """Make ``settings`` the live :class:`RuntimeSettings` and rewrite the
    module constants that mirror it (the compatibility layer)."""
    global _runtime
    if not isinstance(settings, RuntimeSettings):
        raise TypeError(f"expected RuntimeSettings, got {type(settings).__name__}")
    _runtime = settings
    globals().update({name: getattr(settings, field) for name, field in RUNTIME_CONSTANTS.items()})


SUCCESS_STATUSES = {'success', 'completed', 'done', 'finished', 'succeeded'}


FAILED_STATUSES = {'error', 'failed', 'cancelled', 'canceled'}


PROGRESS_STATUSES = {
    'accepted', 'preparing', 'running', 'migrating', 'pre-migrating', 'queued'
}


ACTIVE_VM_STATUS = 'ACTIVE'


PERCENT_METRIC_TOLERANCE = 1.0


MIGRATION_RESULT_SUCCESS = 'success'


MIGRATION_RESULT_FAILED = 'failed'


MIGRATION_RESULT_STUCK_ACTIVE = 'stuck_active'


MIGRATION_RESULT_TIMEOUT = 'timeout'


VM_COOLDOWN_SECONDS = 14400


VM_FAILED_MOVE_COOLDOWN_SECONDS = 14400


COOLDOWN_STATE_FILE = _paths.COOLDOWN_FILE


MIGRATION_EVENTS_FILE = _paths.EVENTS_FILE


LOCK_FILE = _paths.BALANCER_LOCK


ICINGA_OK = 0


ICINGA_WARNING = 1


ICINGA_CRITICAL = 2


ICINGA_UNKNOWN = 3


def env_bool(name, default=False):
    value = getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise _errors.ConfigError(
        f"{name} must be a boolean (true/false, yes/no, on/off, 1/0), got {value!r}."
    )


def parse_max_instances_map(raw):
    """Parse the per-host max-instances JSON map from config into {host: int}."""
    if not raw or not raw.strip():
        return {}

    try:
        data = json.loads(raw)
    except ValueError as e:
        raise _errors.ConfigError(f"MAX_INSTANCES_PER_HOST must be a JSON object, got {raw!r}: {e}")

    if not isinstance(data, dict):
        raise _errors.ConfigError(
            "MAX_INSTANCES_PER_HOST must be a JSON object mapping host to limit, "
            f"got {type(data).__name__}."
        )

    result = {}
    for host, value in data.items():
        if not isinstance(host, str) or not host.strip():
            raise _errors.ConfigError(
                f"MAX_INSTANCES_PER_HOST contains an invalid host key: {host!r}."
            )
        try:
            converted = int(value)
        except (TypeError, ValueError):
            raise _errors.ConfigError(
                f"MAX_INSTANCES_PER_HOST limit for host {host!r} must be an integer, got {value!r}."
            )
        if isinstance(value, bool) or (isinstance(value, float) and value != converted):
            raise _errors.ConfigError(
                f"MAX_INSTANCES_PER_HOST limit for host {host!r} must be an integer, got {value!r}."
            )
        if converted < 0:
            raise _errors.ConfigError(
                f"MAX_INSTANCES_PER_HOST limit for host {host!r} must be >= 0, got {value!r}."
            )
        result[host] = converted
    return result


def env_float(name, default):
    raw = getenv(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise _errors.ConfigError(f"{name} must be a number, got {raw!r}.")
    # NaN compares False with every bound and inf defeats every timeout, so
    # both would silently disable the range checks below.
    if not math.isfinite(value):
        raise _errors.ConfigError(f"{name} must be a finite number, got {raw!r}.")
    return value


def env_int(name, default):
    raw = getenv(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise _errors.ConfigError(f"{name} must be an integer, got {raw!r}.")


class Thresholds:
    """Decision thresholds (percent, except target_mad and min_improvement)."""

    def __init__(self, **values):
        defaults = {attr: default for _key, attr, default in THRESHOLD_KEYS}
        unknown = set(values) - set(defaults)
        if unknown:
            raise TypeError("unknown threshold(s): " + ", ".join(sorted(unknown)))
        for attr, default in defaults.items():
            setattr(self, attr, float(values.get(attr, default)))
        self.validate()

    def validate(self):
        def key(attr):
            return next(k for k, a, _d in THRESHOLD_KEYS if a == attr)
        for attr in ("ram_hotspot", "ram_critical", "cpu_hotspot", "pressure",
                     "target_mad", "placement_pressure"):
            value = getattr(self, attr)
            if not (math.isfinite(value) and 0 < value <= 100):
                raise _errors.ConfigError(f"{key(attr)} must be > 0 and <= 100, got {value}.")
        if not (math.isfinite(self.min_improvement) and 0 < self.min_improvement <= 100):
            raise _errors.ConfigError(
                f"MIN_IMPROVEMENT_THRESHOLD must be > 0 and <= 100, got {self.min_improvement}.")
        if self.ram_critical <= self.ram_hotspot:
            raise _errors.ConfigError(
                f"RAM_CRITICAL_THRESHOLD ({self.ram_critical}) must be above "
                f"RAM_HOTSPOT_THRESHOLD ({self.ram_hotspot}).")

    def pressure_reachable(self):
        """Pressure mode needs a mean above PRESSURE_THRESHOLD while no host is a
        hotspot; that is only possible if it is below one hotspot threshold."""
        return self.pressure < self.ram_hotspot or self.pressure < self.cpu_hotspot

    def describe(self):
        return (f"RAM hotspot >{self.ram_hotspot:g}%, RAM critical >{self.ram_critical:g}%, CPU hotspot >{self.cpu_hotspot:g}%, pressure mean >{self.pressure:g}%, "
                f"target MAD {self.target_mad:g}, min improvement {self.min_improvement:g}, Placement pressure >={self.placement_pressure:g}%")

    def as_dict(self):
        return {key: getattr(self, attr) for key, attr, _d in THRESHOLD_KEYS}


def load_thresholds():
    """Thresholds from the active configuration scope (config file + env)."""
    return Thresholds(**{attr: env_float(key, default) for key, attr, default in THRESHOLD_KEYS})


def apply_thresholds(thresholds):
    """Make ``thresholds`` the live values for this process (runtime()).

    Planner, drain and report code read runtime() at call time, so this is
    applied once by the CLI right after the config is loaded.
    """
    set_runtime(dataclasses.replace(
        runtime(),
        ram_hotspot_threshold=thresholds.ram_hotspot,
        ram_critical_threshold=thresholds.ram_critical,
        cpu_hotspot_threshold=thresholds.cpu_hotspot,
        auto_pressure_threshold=thresholds.pressure,
        target_avg_mad=thresholds.target_mad,
        min_improvement_threshold=thresholds.min_improvement,
        placement_pressure_threshold=thresholds.placement_pressure,
        # Drain limits follow the hotspot thresholds (as with the defaults).
        drain_max_host_ram_pct=thresholds.ram_hotspot,
        drain_max_host_cpu_pct=thresholds.cpu_hotspot,
        drain_absolute_max_host_ram_pct=thresholds.ram_critical,
    ))


class SoftAntiAffinityHeal:
    """rc11 soft-anti-affinity heal settings (see SOFT_AA_HEAL_KEYS)."""

    def __init__(self, **values):
        defaults = {attr: default for _key, attr, default in SOFT_AA_HEAL_KEYS}
        unknown = set(values) - set(defaults)
        if unknown:
            raise TypeError("unknown soft-anti-affinity setting(s): " + ", ".join(sorted(unknown)))
        self.enabled = bool(values.get("enabled", defaults["enabled"]))
        self.mad_tolerance = float(values.get("mad_tolerance", defaults["mad_tolerance"]))
        self.tiebreak_margin = float(values.get("tiebreak_margin", defaults["tiebreak_margin"]))
        if not (math.isfinite(self.mad_tolerance) and self.mad_tolerance >= 0):
            raise _errors.ConfigError(
                f"SOFT_AA_HEAL_MAD_TOLERANCE must be a finite number >= 0, got {self.mad_tolerance}.")
        if not (math.isfinite(self.tiebreak_margin) and 0 <= self.tiebreak_margin <= 1):
            raise _errors.ConfigError(
                f"SOFT_AA_TIEBREAK_MARGIN must be between 0 and 1, got {self.tiebreak_margin}.")

    def describe(self):
        if not self.enabled:
            return "soft-anti-affinity heal off"
        return (f"soft-anti-affinity heal on (MAD tolerance {self.mad_tolerance:g}, "
                f"tie-break margin {self.tiebreak_margin:g})")

    def as_dict(self):
        return {key: getattr(self, attr) for key, attr, _d in SOFT_AA_HEAL_KEYS}


def load_soft_aa_heal():
    """Soft-anti-affinity heal settings from the active configuration scope."""
    return SoftAntiAffinityHeal(
        enabled=env_bool("HEAL_SOFT_ANTI_AFFINITY", True),
        mad_tolerance=env_float("SOFT_AA_HEAL_MAD_TOLERANCE", 0.05),
        tiebreak_margin=env_float("SOFT_AA_TIEBREAK_MARGIN", 0.15),
    )


def apply_soft_aa_heal(settings):
    """Make ``settings`` the live values for this process (runtime(); the
    planner and the drain packer read it at call time)."""
    set_runtime(dataclasses.replace(
        runtime(),
        heal_soft_anti_affinity=settings.enabled,
        soft_aa_heal_mad_tolerance=settings.mad_tolerance,
        soft_aa_tiebreak_margin=settings.tiebreak_margin,
    ))


class MigrationTimeouts:
    """How long to wait for a live migration (seconds): stop after
    ``no_progress`` seconds without progress, never wait past ``maximum``."""

    def __init__(self, **values):
        defaults = {attr: default for _key, attr, default in TIMEOUT_KEYS}
        unknown = set(values) - set(defaults)
        if unknown:
            raise TypeError("unknown timeout(s): " + ", ".join(sorted(unknown)))
        for attr, default in defaults.items():
            setattr(self, attr, float(values.get(attr, default)))
        for key, attr, _default in TIMEOUT_KEYS:
            value = getattr(self, attr)
            if not (math.isfinite(value) and value > 0):
                raise _errors.ConfigError(f"{key} must be a positive number of seconds, got {value}.")
        if self.maximum < self.no_progress:
            raise _errors.ConfigError(
                f"MIGRATION_MAX_WAIT ({self.maximum:g}) must be >= MIGRATION_TIMEOUT ({self.no_progress:g}).")

    def describe(self):
        return f"stop after {self.no_progress:g} s without progress, at most {self.maximum:g} s"

    def as_dict(self):
        return {key: getattr(self, attr) for key, attr, _d in TIMEOUT_KEYS}


def load_migration_timeouts():
    """Migration timeouts from the active configuration scope (config file + env)."""
    obsolete = [key for key in OBSOLETE_TIMEOUT_KEYS if getenv(key) is not None]
    if obsolete:
        logging.warning(
            "Ignoring obsolete key(s) %s: MIGRATION_TIMEOUT is now the only no-progress window "
            "(default 600 s); remove the key from the config.", ", ".join(obsolete))
    return MigrationTimeouts(**{attr: env_float(key, default) for key, attr, default in TIMEOUT_KEYS})


def apply_migration_timeouts(timeouts):
    """Make ``timeouts`` the live values for this process (runtime(); monitor,
    journal and settle waits read it when they start waiting)."""
    set_runtime(dataclasses.replace(
        runtime(),
        migration_timeout=timeouts.no_progress,
        migration_max_wait=timeouts.maximum,
    ))


class LoadbalancerConfig:
    def __init__(
        self,
        prometheus_query_url=PROMETHEUS_QUERY_URL,
        prometheus_query_mem_used=DEFAULT_PROMETHEUS_MEM_USED,
        prometheus_query_cpu_used=DEFAULT_PROMETHEUS_CPU_USED,
        prometheus_query_vm_ram_rss=DEFAULT_PROMETHEUS_VM_RAM_RSS,
        prometheus_query_logical_cpus=DEFAULT_PROMETHEUS_LOGICAL_CPUS,
        prometheus_query_total_ram_bytes=DEFAULT_PROMETHEUS_TOTAL_RAM_BYTES,
        placement_vcpu_headroom_percent=DEFAULT_PLACEMENT_VCPU_HEADROOM_PERCENT,
        placement_ram_headroom_percent=DEFAULT_PLACEMENT_RAM_HEADROOM_PERCENT,
        enable_make_room=DEFAULT_ENABLE_MAKE_ROOM,
        make_room_max_depth=DEFAULT_MAKE_ROOM_MAX_DEPTH,
        make_room_max_vm_cpu_percent=DEFAULT_MAKE_ROOM_MAX_VM_CPU_PERCENT,
        make_room_max_vm_ram_percent=DEFAULT_MAKE_ROOM_MAX_VM_RAM_PERCENT,
        placement_final_allocation_candidate_check=DEFAULT_PLACEMENT_FINAL_ALLOCATION_CANDIDATE_CHECK,
        placement_sync_timeout=DEFAULT_PLACEMENT_SYNC_TIMEOUT,
        placement_sync_interval=DEFAULT_PLACEMENT_SYNC_INTERVAL,
        alert_email_to=None,
        alert_email_from=None,
        smtp_server=None,
        smtp_port=25,
        smtp_user=None,
        smtp_password=None,
        smtp_starttls=False,
        max_instances_per_host=None,
        max_instances_per_host_default=DEFAULT_MAX_INSTANCES_PER_HOST,
        exclude_vms_file=DEFAULT_EXCLUDE_VMS_FILE,
        make_room_min_improvement=DEFAULT_MAKE_ROOM_MIN_IMPROVEMENT,
        prometheus_node_job=PROMETHEUS_NODE_JOB,
        prometheus_libvirt_job=PROMETHEUS_LIBVIRT_JOB,
        planner_candidate_budget=100000,
        compensation_cpu_window_seconds=None,
        prometheus_query_mem_used_instant=None,
        prometheus_query_mem_sample_timestamp=None,
        thresholds=None,
        os_cloud=DEFAULT_OS_CLOUD,
        migration_timeouts=None,
        soft_aa_heal=None,
    ):
        self.thresholds = thresholds if thresholds is not None else Thresholds()
        self.migration_timeouts = migration_timeouts if migration_timeouts is not None else MigrationTimeouts()
        self.soft_aa_heal = soft_aa_heal if soft_aa_heal is not None else SoftAntiAffinityHeal()
        self.os_cloud = os_cloud
        queries = build_default_queries(prometheus_node_job, prometheus_libvirt_job)
        self.prometheus_query_mem_used_instant = (
            prometheus_query_mem_used_instant or queries["mem_used_instant"])
        self.prometheus_query_mem_sample_timestamp = (
            prometheus_query_mem_sample_timestamp or queries["mem_sample_timestamp"])
        if prometheus_query_mem_used is None:
            prometheus_query_mem_used = queries['mem_used']
        if prometheus_query_cpu_used is None:
            prometheus_query_cpu_used = queries['cpu_used']
        if prometheus_query_logical_cpus is None:
            prometheus_query_logical_cpus = queries['logical_cpus']
        if prometheus_query_total_ram_bytes is None:
            prometheus_query_total_ram_bytes = queries['total_ram_bytes']
        self.prometheus_node_job = prometheus_node_job
        if type(planner_candidate_budget) is not int or planner_candidate_budget < 1:
            raise _errors.ConfigError("PLANNER_CANDIDATE_BUDGET must be a positive integer")
        self.planner_candidate_budget = planner_candidate_budget
        self.compensation_cpu_window_seconds = compensation_cpu_window_seconds
        self.prometheus_libvirt_job = prometheus_libvirt_job
        self.prometheus_query_url = prometheus_query_url
        self.prometheus_query_mem_used = prometheus_query_mem_used
        self.prometheus_query_cpu_used = prometheus_query_cpu_used
        self.prometheus_query_vm_ram_rss = prometheus_query_vm_ram_rss
        self.prometheus_query_logical_cpus = prometheus_query_logical_cpus
        self.prometheus_query_total_ram_bytes = prometheus_query_total_ram_bytes
        self.placement_vcpu_headroom_percent = placement_vcpu_headroom_percent
        self.placement_ram_headroom_percent = placement_ram_headroom_percent
        self.enable_make_room = enable_make_room
        self.make_room_max_depth = make_room_max_depth
        self.make_room_min_improvement = make_room_min_improvement
        self.make_room_max_vm_cpu_percent = make_room_max_vm_cpu_percent
        self.make_room_max_vm_ram_percent = make_room_max_vm_ram_percent
        self.placement_final_allocation_candidate_check = placement_final_allocation_candidate_check
        self.placement_sync_timeout = placement_sync_timeout
        self.placement_sync_interval = placement_sync_interval
        self.alert_email_to = alert_email_to
        self.alert_email_from = alert_email_from
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.smtp_starttls = smtp_starttls
        self.max_instances_per_host = max_instances_per_host or {}
        self.max_instances_per_host_default = max_instances_per_host_default
        self.exclude_vms_file = exclude_vms_file

    @classmethod
    @configuration_scope
    def load_config(cls, filename: str):
        if filename and not os.path.exists(filename):
            logging.warning(
                "Config file %s not found; using environment variables and built-in defaults.",
                filename,
            )


        smtp_port = env_int("SMTP_PORT", DEFAULT_SMTP_PORT)
        if smtp_port < 1 or smtp_port > 65535:
            raise _errors.ConfigError(f"SMTP_PORT must be between 1 and 65535, got {smtp_port}.")
        max_instances_default = env_int(
            "MAX_INSTANCES_PER_HOST_DEFAULT", DEFAULT_MAX_INSTANCES_PER_HOST
        )
        if max_instances_default < 0:
            raise _errors.ConfigError(
                "MAX_INSTANCES_PER_HOST_DEFAULT must be >= 0, "
                f"got {max_instances_default}."
            )
        max_instances_map = parse_max_instances_map(getenv("MAX_INSTANCES_PER_HOST"))

        vcpu_headroom = env_float(
            "PLACEMENT_VCPU_HEADROOM_PERCENT", DEFAULT_PLACEMENT_VCPU_HEADROOM_PERCENT
        )
        ram_headroom = env_float(
            "PLACEMENT_RAM_HEADROOM_PERCENT", DEFAULT_PLACEMENT_RAM_HEADROOM_PERCENT
        )
        for name, value in (
            ("PLACEMENT_VCPU_HEADROOM_PERCENT", vcpu_headroom),
            ("PLACEMENT_RAM_HEADROOM_PERCENT", ram_headroom),
        ):
            if value < 0 or value >= 100:
                raise _errors.ConfigError(f"{name} must be >= 0 and < 100, got {value}.")

        make_room_depth = env_int("MAKE_ROOM_MAX_DEPTH", DEFAULT_MAKE_ROOM_MAX_DEPTH)
        if make_room_depth != 2:
            raise _errors.ConfigError(
                "MAKE_ROOM_MAX_DEPTH currently supports exactly 2. "
                f"Got {make_room_depth}; deeper chains are intentionally disabled."
            )

        thresholds = load_thresholds()
        # Default MAKE_ROOM bar: one direct-move threshold per step.
        make_room_min_improvement = env_float(
            "MAKE_ROOM_MIN_IMPROVEMENT", thresholds.min_improvement * DEFAULT_MAKE_ROOM_MAX_DEPTH
        )
        if (
            not math.isfinite(make_room_min_improvement)
            or make_room_min_improvement < thresholds.min_improvement
        ):
            raise _errors.ConfigError(
                "MAKE_ROOM_MIN_IMPROVEMENT must be finite and >= the direct-move "
                f"threshold MIN_IMPROVEMENT_THRESHOLD ({thresholds.min_improvement}), got "
                f"{make_room_min_improvement}."
            )
        # Same rule as the CLIs (they pass --os-cloud on top of it).
        os_cloud = resolve_os_cloud(None, filename)

        make_room_cpu_limit = env_float(
            "MAKE_ROOM_MAX_VM_CPU_PERCENT", DEFAULT_MAKE_ROOM_MAX_VM_CPU_PERCENT
        )
        make_room_ram_limit = env_float(
            "MAKE_ROOM_MAX_VM_RAM_PERCENT", DEFAULT_MAKE_ROOM_MAX_VM_RAM_PERCENT
        )
        for name, value in (
            ("MAKE_ROOM_MAX_VM_CPU_PERCENT", make_room_cpu_limit),
            ("MAKE_ROOM_MAX_VM_RAM_PERCENT", make_room_ram_limit),
        ):
            if value < 0 or value > 100:
                raise _errors.ConfigError(f"{name} must be between 0 and 100, got {value}.")

        placement_sync_timeout = env_float(
            "PLACEMENT_SYNC_TIMEOUT", DEFAULT_PLACEMENT_SYNC_TIMEOUT
        )
        placement_sync_interval = env_float(
            "PLACEMENT_SYNC_INTERVAL", DEFAULT_PLACEMENT_SYNC_INTERVAL
        )
        if placement_sync_timeout < 0:
            raise _errors.ConfigError(
                f"PLACEMENT_SYNC_TIMEOUT must be >= 0, got {placement_sync_timeout}."
            )
        if placement_sync_interval <= 0:
            raise _errors.ConfigError(
                f"PLACEMENT_SYNC_INTERVAL must be > 0, got {placement_sync_interval}."
            )

        # Job names are spliced into PromQL label matchers; require_job rejects
        # empty values and characters that would break or rewrite a query.
        jobs = {name: require_job(getenv(name), name)
                for name in ("PROMETHEUS_NODE_JOB", "PROMETHEUS_LIBVIRT_JOB")}
        obsolete = [name for name in OBSOLETE_CONFIG_KEYS if getenv(name) is not None]
        if obsolete:
            logging.warning(
                "Config %s: ignoring obsolete key(s) %s; allocation ratios come from Placement.",
                filename, ", ".join(obsolete),
            )
        queries = build_default_queries(jobs["PROMETHEUS_NODE_JOB"], jobs["PROMETHEUS_LIBVIRT_JOB"])

        cpu_window = getenv("COMPENSATION_CPU_WINDOW_SECONDS")
        if cpu_window is not None:
            cpu_window = env_float("COMPENSATION_CPU_WINDOW_SECONDS", 0)
            if cpu_window <= 0:
                raise _errors.ConfigError("COMPENSATION_CPU_WINDOW_SECONDS must be > 0.")
        cfg = cls(
            thresholds=thresholds,
            os_cloud=os_cloud,
            migration_timeouts=load_migration_timeouts(),
            soft_aa_heal=load_soft_aa_heal(),
            compensation_cpu_window_seconds=cpu_window,
            prometheus_query_mem_used_instant=getenv("PROMETHEUS_MEM_USED_INSTANT"),
            prometheus_query_mem_sample_timestamp=getenv("PROMETHEUS_MEM_SAMPLE_TIMESTAMP"),
            planner_candidate_budget=env_int("PLANNER_CANDIDATE_BUDGET", 100000),
            prometheus_node_job=jobs["PROMETHEUS_NODE_JOB"],
            prometheus_libvirt_job=jobs["PROMETHEUS_LIBVIRT_JOB"],
            prometheus_query_url=getenv("PROMETHEUS_QUERY_URL", PROMETHEUS_QUERY_URL),
            prometheus_query_mem_used=getenv("PROMETHEUS_MEM_USED", queries["mem_used"]),
            prometheus_query_cpu_used=getenv("PROMETHEUS_CPU_USED", queries["cpu_used"]),
            prometheus_query_vm_ram_rss=getenv("PROMETHEUS_VM_RAM_RSS", DEFAULT_PROMETHEUS_VM_RAM_RSS),
            prometheus_query_logical_cpus=getenv(
                "PROMETHEUS_LOGICAL_CPUS", queries["logical_cpus"]
            ),
            prometheus_query_total_ram_bytes=getenv(
                "PROMETHEUS_TOTAL_RAM_BYTES", queries["total_ram_bytes"]
            ),
            placement_vcpu_headroom_percent=vcpu_headroom,
            placement_ram_headroom_percent=ram_headroom,
            enable_make_room=env_bool("ENABLE_MAKE_ROOM", DEFAULT_ENABLE_MAKE_ROOM),
            make_room_max_depth=make_room_depth,
            make_room_min_improvement=make_room_min_improvement,
            make_room_max_vm_cpu_percent=make_room_cpu_limit,
            make_room_max_vm_ram_percent=make_room_ram_limit,
            placement_final_allocation_candidate_check=env_bool(
                "PLACEMENT_FINAL_ALLOCATION_CANDIDATE_CHECK",
                DEFAULT_PLACEMENT_FINAL_ALLOCATION_CANDIDATE_CHECK,
            ),
            placement_sync_timeout=placement_sync_timeout,
            placement_sync_interval=placement_sync_interval,
            alert_email_to=getenv("ALERT_EMAIL_TO") or getenv("EMAIL_TO") or DEFAULT_ALERT_EMAIL_TO,
            alert_email_from=getenv("ALERT_EMAIL_FROM") or getenv("EMAIL_FROM") or DEFAULT_ALERT_EMAIL_FROM,
            smtp_server=getenv("SMTP_SERVER") or DEFAULT_SMTP_SERVER,
            smtp_port=smtp_port,
            smtp_user=getenv("SMTP_USER") or DEFAULT_SMTP_USER,
            smtp_password=getenv("SMTP_PASSWORD") or DEFAULT_SMTP_PASSWORD,
            smtp_starttls=env_bool("SMTP_STARTTLS", DEFAULT_SMTP_STARTTLS),
            max_instances_per_host=max_instances_map,
            max_instances_per_host_default=max_instances_default,
            exclude_vms_file=getenv("EXCLUDE_VMS_FILE") or DEFAULT_EXCLUDE_VMS_FILE,
        )

        if not (cfg.prometheus_query_url or "").strip():
            raise _errors.ConfigError(
                "PROMETHEUS_QUERY_URL is required (e.g. http://prometheus:9090/api/v1/query); "
                f"set it in {filename or DEFAULT_CONFIG_FILE}."
            )
        if not (cfg.prometheus_query_logical_cpus or "").strip():
            raise _errors.ConfigError("PROMETHEUS_LOGICAL_CPUS is empty; runtime CPU capacity is required.")
        if not (cfg.prometheus_query_total_ram_bytes or "").strip():
            raise _errors.ConfigError("PROMETHEUS_TOTAL_RAM_BYTES is empty; runtime RAM capacity is required.")
        logging.info(
            "Config %s: Prometheus jobs node=%s libvirt=%s",
            filename,
            cfg.prometheus_node_job,
            cfg.prometheus_libvirt_job,
        )

        return cfg

    def vm_ram_rss_query(self):
        """Built-in VM RSS query template for this config's libvirt job."""
        return build_default_queries(self.prometheus_node_job, self.prometheus_libvirt_job)["vm_ram_rss"]


def enable_default_vm_ram_rss_query(cfg):
    """Enable the built-in VM RSS query unless a custom query is already set."""
    if not cfg.prometheus_query_vm_ram_rss:
        cfg.prometheus_query_vm_ram_rss = cfg_vm_ram_rss_query(cfg)


def cfg_vm_ram_rss_query(cfg):
    getter = getattr(cfg, "vm_ram_rss_query", None)
    return getter() if getter else (
        f'avg_over_time(libvirt_domain_stat_memory_rss_bytes{{job="{cfg_libvirt_job(cfg)}", host=~"__HOST_RE__"}}[5m])'
    )


def cfg_libvirt_job(cfg):
    return require_job(getattr(cfg, "prometheus_libvirt_job", None), 'PROMETHEUS_LIBVIRT_JOB')


DRAIN_PLAN_TYPE = "drain"


DRAIN_STEP_REASON = "drain_evacuate"


DRAIN_EXIT_OK = 0


DRAIN_EXIT_FAILED = 1


DRAIN_EXIT_INFEASIBLE = 2


class _ConfigModule(types.ModuleType):
    """Compatibility layer: assigning a mirrored module constant
    (``config.RAM_HOTSPOT_THRESHOLD = 70``, ``monkeypatch.setattr(config, ...)``)
    updates runtime() instead of creating a second, diverging truth."""

    def __setattr__(self, name, value):
        field = RUNTIME_CONSTANTS.get(name)
        if field is None:
            super().__setattr__(name, value)
        else:
            set_runtime(dataclasses.replace(runtime(), **{field: value}))


sys.modules[__name__].__class__ = _ConfigModule
