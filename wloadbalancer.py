#!/usr/bin/env python3
"""
OpenStack Workload Balancer — Automated VM migration to minimize MAD/spread.

Refactored version:
- monitor/runtime use the same evaluation path
- synchronized classification and messaging
- clear distinction between:
  - technical candidate
  - valid candidate above improvement threshold

Detailed statuses:
- BALANCED
- BALANCED_OPTIMIZABLE
- UNBALANCED
- UNBALANCED_NO_VALID_MOVE
- RAM_PRESSURE
- CPU_PRESSURE
- PRESSURE_NO_VALID_MOVE
- RAM_HOTSPOT_OPTIMIZABLE
- RAM_HOTSPOT_NO_VALID_MOVE
- CPU_HOTSPOT_OPTIMIZABLE
- CPU_HOTSPOT_NO_VALID_MOVE
"""

import sys
import time
import json
from collections import defaultdict, namedtuple
from datetime import datetime
import os
import logging
import smtplib
from email.mime.text import MIMEText
import requests
import dotenv
import click
import openstack
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# TUNABLES
# ============================================================================

# Minimum floor for dynamic improvement threshold
MIN_IMPROVEMENT_THRESHOLD = 0.05

PROMETHEUS_NODE_JOB = 'Prod-Openstack-Node-Exporter-DC11'
PROMETHEUS_LIBVIRT_JOB = 'Prod-Openstack-LibVirt-Exporter-DC11'
PROMETHEUS_OPENSTACK_EXPORTER_JOB = 'Prod-Openstack-Exporter-DC11'

DEFAULT_PROMETHEUS_CPU_USED = (
    f'sort_desc(100 - avg(irate(node_cpu_seconds_total{{job="{PROMETHEUS_NODE_JOB}",mode="idle"}}[5m]) * 100) by (alias))'
)
DEFAULT_PROMETHEUS_MEM_USED = (
    f'sort_desc(100 - ((avg_over_time(node_memory_MemAvailable_bytes{{job="{PROMETHEUS_NODE_JOB}"}}[5m]) * 100) / avg_over_time(node_memory_MemTotal_bytes{{job="{PROMETHEUS_NODE_JOB}"}}[5m])))'
)
DEFAULT_PROMETHEUS_CPU_RATIO = (
    f'openstack_placement_resource_allocation_ratio{{job="{PROMETHEUS_OPENSTACK_EXPORTER_JOB}",resourcetype="VCPU"}}'
)
DEFAULT_PROMETHEUS_MEM_RATIO = (
    f'openstack_placement_resource_allocation_ratio{{job="{PROMETHEUS_OPENSTACK_EXPORTER_JOB}",resourcetype="MEMORY_MB"}}'
)

# Acceptable per-host deviation
TARGET_AVG_MAD = 3
MIN_CPU_WEIGHT = 0.2
MIN_RAM_WEIGHT = 0.2
PRESSURE_MAD_WEIGHT = 0.7
PRESSURE_MEAN_WEIGHT = 0.3
AUTO_PRESSURE_THRESHOLD = 80.0

# Hotspot thresholds
RAM_HOTSPOT_THRESHOLD = 75.0
RAM_CRITICAL_THRESHOLD = 85.0
CPU_HOTSPOT_THRESHOLD = 85.0

PROMETHEUS_QUERY_URL = "http://kprometheus.com:9090/api/v1/query"

# Temporary email config. Environment/config values still override these defaults.
DEFAULT_ALERT_EMAIL_TO = ""
DEFAULT_ALERT_EMAIL_FROM = "loadleveller@localhost"
DEFAULT_SMTP_SERVER = ""
DEFAULT_SMTP_PORT = 25
DEFAULT_SMTP_USER = ""
DEFAULT_SMTP_PASSWORD = ""
DEFAULT_SMTP_STARTTLS = False

# Migration monitor
POLL_INTERVAL = 2
MIGRATION_TIMEOUT = 600  # seconds
SUCCESS_STATUSES = {'success', 'completed', 'done', 'finished', 'succeeded'}
FAILED_STATUSES = {'error', 'failed'}
PROGRESS_STATUSES = {'running', 'migrating', 'pre-migrating', 'queued'}

MIGRATION_RESULT_SUCCESS = 'success'
MIGRATION_RESULT_FAILED = 'failed'
MIGRATION_RESULT_STUCK_ACTIVE = 'stuck_active'
MIGRATION_RESULT_TIMEOUT = 'timeout'

# Cooldown to prevent VM ping-pong migrations
VM_COOLDOWN_SECONDS = 1800  # 30 minutes
VM_FAILED_MOVE_COOLDOWN_SECONDS = 3600  # 60 minutes
COOLDOWN_STATE_FILE = "/var/log/loadleveller_vm_cooldown.json"
MIGRATION_EVENTS_FILE = "/var/log/loadleveller_migration_events.jsonl"
VM_LAST_MOVED_AT = {}
VM_LAST_FAILED_MOVE_AT = {}

# Icinga/Nagios exit codes
ICINGA_OK = 0
ICINGA_WARNING = 1
ICINGA_CRITICAL = 2
ICINGA_UNKNOWN = 3


# ============================================================================
# EXCEPTIONS
# ============================================================================

class PrometheusError(Exception):
    pass


class ConfigError(Exception):
    pass


BestMove = namedtuple(
    "BestMove",
    [
        "vm_id",
        "src_host",
        "dst_host",
        "cpu_impact",
        "ram_impact",
        "new_host_cpu",
        "new_host_ram",
        "real_vm_cores",
        "vcpus",
        "real_vm_ram",
        "vm_ram_gb",
        "host_cpu_avail",
        "host_ram_avail",
        "new_weighted_mad",
        "mad_improvement",
        "mad_cpu",
        "mad_ram",
    ],
)


# ============================================================================
# CONFIG / LOGGING
# ============================================================================

def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


class LoadbalancerConfig:
    def __init__(
        self,
        prometheus_query_url=PROMETHEUS_QUERY_URL,
        prometheus_query_mem_used=DEFAULT_PROMETHEUS_MEM_USED,
        prometheus_query_cpu_used=DEFAULT_PROMETHEUS_CPU_USED,
        prometheus_query_cpu_ratio=DEFAULT_PROMETHEUS_CPU_RATIO,
        prometheus_query_mem_ratio=DEFAULT_PROMETHEUS_MEM_RATIO,
        alert_email_to=None,
        alert_email_from=None,
        smtp_server=None,
        smtp_port=25,
        smtp_user=None,
        smtp_password=None,
        smtp_starttls=False,
    ):
        self.prometheus_query_url = prometheus_query_url
        self.prometheus_query_mem_used = prometheus_query_mem_used
        self.prometheus_query_cpu_used = prometheus_query_cpu_used
        self.prometheus_query_cpu_ratio = prometheus_query_cpu_ratio
        self.prometheus_query_mem_ratio = prometheus_query_mem_ratio
        self.alert_email_to = alert_email_to
        self.alert_email_from = alert_email_from
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.smtp_starttls = smtp_starttls

    @classmethod
    def load_config(cls, filename: str):
        dotenv.load_dotenv(dotenv_path=filename)
        return cls(
            prometheus_query_url=os.getenv("PROMETHEUS_QUERY_URL", PROMETHEUS_QUERY_URL),
            prometheus_query_mem_used=os.getenv("PROMETHEUS_MEM_USED", DEFAULT_PROMETHEUS_MEM_USED),
            prometheus_query_cpu_used=os.getenv("PROMETHEUS_CPU_USED", DEFAULT_PROMETHEUS_CPU_USED),
            prometheus_query_cpu_ratio=os.getenv("PROMETHEUS_CPU_RATIO", DEFAULT_PROMETHEUS_CPU_RATIO),
            prometheus_query_mem_ratio=os.getenv("PROMETHEUS_MEM_RATIO", DEFAULT_PROMETHEUS_MEM_RATIO),
            alert_email_to=os.getenv("ALERT_EMAIL_TO") or os.getenv("EMAIL_TO") or DEFAULT_ALERT_EMAIL_TO,
            alert_email_from=os.getenv("ALERT_EMAIL_FROM") or os.getenv("EMAIL_FROM") or DEFAULT_ALERT_EMAIL_FROM,
            smtp_server=os.getenv("SMTP_SERVER") or DEFAULT_SMTP_SERVER,
            smtp_port=int(os.getenv("SMTP_PORT", str(DEFAULT_SMTP_PORT))),
            smtp_user=os.getenv("SMTP_USER") or DEFAULT_SMTP_USER,
            smtp_password=os.getenv("SMTP_PASSWORD") or DEFAULT_SMTP_PASSWORD,
            smtp_starttls=env_bool("SMTP_STARTTLS", DEFAULT_SMTP_STARTTLS),
        )


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s')


def format_migration_duration(duration_seconds):
    if duration_seconds is None:
        return "n/a"

    try:
        seconds = max(float(duration_seconds), 0.0)
    except (TypeError, ValueError):
        return "n/a"

    return f"{seconds:.1f} seconds ({seconds / 60.0:.2f} minutes)"


def format_migration_timestamp(value):
    if value is None:
        return "n/a"

    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d %H:%M:%S')

    try:
        return datetime.fromtimestamp(float(value)).strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError, OSError):
        return "n/a"


def send_migration_alert_email(
    cfg,
    aggregate,
    vm_id,
    vm_name,
    src_host,
    dst_host,
    result,
    detail=None,
    duration_seconds=None,
    started_at=None,
    ended_at=None,
    project_name=None,
    project_id=None,
):
    if not cfg.alert_email_to or not cfg.smtp_server:
        logging.debug("Migration alert email not configured; skipping email notification.")
        return

    recipients = [x.strip() for x in cfg.alert_email_to.split(",") if x.strip()]
    if not recipients:
        logging.debug("Migration alert email recipient list is empty; skipping email notification.")
        return

    sender = cfg.alert_email_from or "loadleveller@localhost"
    subject = f"[OpenStack][{result}] Migration alert for {vm_name}"
    body = "\n".join(
        [
            "OpenStack workload balancer migration alert",
            "",
            f"Result    : {result}",
            f"Aggregate : {aggregate}",
            f"VM        : {vm_name}",
            f"VM ID     : {vm_id}",
            f"Project   : {project_name or project_id or 'n/a'}",
            f"Project ID: {project_id or 'n/a'}",
            f"Source    : {src_host}",
            f"Target    : {dst_host}",
            f"Route     : {src_host} -> {dst_host}",
            f"Started   : {format_migration_timestamp(started_at)}",
            f"Ended     : {format_migration_timestamp(ended_at)}",
            f"Duration  : {format_migration_duration(duration_seconds)}",
            f"Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Detail    : {detail or 'n/a'}",
        ]
    )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    try:
        smtp = smtplib.SMTP(cfg.smtp_server, cfg.smtp_port, timeout=30)
        try:
            smtp.ehlo()
            if cfg.smtp_starttls:
                smtp.starttls()
                smtp.ehlo()

            if cfg.smtp_user:
                smtp.login(cfg.smtp_user, cfg.smtp_password or "")

            smtp.sendmail(sender, recipients, msg.as_string())
        finally:
            try:
                smtp.quit()
            except Exception:
                pass

        logging.info("Sent migration alert email for VM %s (%s) to %s.", vm_name, vm_id, cfg.alert_email_to)
    except Exception as e:
        logging.warning("Failed to send migration alert email for VM %s (%s): %s", vm_name, vm_id, e)


# ============================================================================
# PROMETHEUS
# ============================================================================

def do_query(query_url: str, query: str):
    """Execute a Prometheus instant query and return the decoded payload."""
    try:
        params = {'query': query}
        response = requests.get(query_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('status') != 'success':
            raise PrometheusError(f"Prometheus query error: {data.get('error', 'Unknown error')}")
        return data
    except ValueError as e:
        raise PrometheusError(f"Prometheus returned invalid JSON: {e}")
    except requests.RequestException as e:
        raise PrometheusError(f"Prometheus HTTP error: {e}")


def get_prometheus_result_rows(data, error_message):
    """Extract the standard Prometheus result list or raise a consistent error."""
    try:
        return data["data"]["result"]
    except (KeyError, TypeError) as e:
        raise PrometheusError(f"{error_message}: {e}")


def get_metrics(query_url: str, query: str):
    """Return host metrics keyed by host alias from a Prometheus query."""
    data = do_query(query_url, query)
    results = {}
    try:
        for r in get_prometheus_result_rows(data, "Prometheus returned malformed results"):
            host = r["metric"]["alias"]
            value = float(r["value"][1])
            results[host] = {"host": host, "value": value}
        return results
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise PrometheusError(f"Prometheus returned malformed results: {e}")


def get_allocation_ratios(cfg):
    """Return host CPU and RAM allocation ratios keyed by hypervisor name."""
    cpu_data = do_query(cfg.prometheus_query_url, cfg.prometheus_query_cpu_ratio)
    mem_data = do_query(cfg.prometheus_query_url, cfg.prometheus_query_mem_ratio)
    cpu_ratio = {}
    mem_ratio = {}

    try:
        for r in get_prometheus_result_rows(
            cpu_data,
            "Prometheus returned malformed allocation ratio results",
        ):
            host = r["metric"].get("hostname")
            if host:
                cpu_ratio[host] = float(r["value"][1])

        for r in get_prometheus_result_rows(
            mem_data,
            "Prometheus returned malformed allocation ratio results",
        ):
            host = r["metric"].get("hostname")
            if host:
                mem_ratio[host] = float(r["value"][1])

        return cpu_ratio, mem_ratio
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise PrometheusError(f"Prometheus returned malformed allocation ratio results: {e}")


# ============================================================================
# COOLDOWN STATE
# ============================================================================


def load_active_cooldown_entries(raw, max_age, label, now=None):
    if not isinstance(raw, dict):
        logging.warning("Cooldown %s data is not a JSON object. Ignoring it.", label)
        return {}

    if now is None:
        now = time.time()

    loaded = {}
    for vm_id, recorded_at in raw.items():
        try:
            recorded_at = float(recorded_at)
        except (TypeError, ValueError):
            logging.debug("Ignore invalid %s cooldown timestamp for VM %s: %r", label, vm_id, recorded_at)
            continue

        if recorded_at > now:
            logging.debug("Ignore future %s cooldown timestamp for VM %s: %r", label, vm_id, recorded_at)
            continue

        if now - recorded_at < max_age:
            loaded[vm_id] = recorded_at

    return loaded


def load_cooldown_state(filename=COOLDOWN_STATE_FILE):
    global VM_LAST_MOVED_AT, VM_LAST_FAILED_MOVE_AT

    try:
        if not os.path.exists(filename):
            VM_LAST_MOVED_AT = {}
            VM_LAST_FAILED_MOVE_AT = {}
            return

        with open(filename, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            logging.warning("Cooldown state file %s is not a JSON object. Ignoring it.", filename)
            VM_LAST_MOVED_AT = {}
            VM_LAST_FAILED_MOVE_AT = {}
            return

        now = time.time()
        if "moved" in raw or "failed" in raw:
            moved_raw = raw.get("moved", {})
            failed_raw = raw.get("failed", {})
        else:
            # Backward compatibility with the old flat {vm_id: moved_at} format.
            moved_raw = raw
            failed_raw = {}

        VM_LAST_MOVED_AT = load_active_cooldown_entries(
            moved_raw, VM_COOLDOWN_SECONDS, "moved", now=now
        )
        VM_LAST_FAILED_MOVE_AT = load_active_cooldown_entries(
            failed_raw, VM_FAILED_MOVE_COOLDOWN_SECONDS, "failed-move", now=now
        )
        logging.info(
            "Loaded %d moved and %d failed-move VM cooldown entries from %s.",
            len(VM_LAST_MOVED_AT),
            len(VM_LAST_FAILED_MOVE_AT),
            filename,
        )
    except FileNotFoundError:
        VM_LAST_MOVED_AT = {}
        VM_LAST_FAILED_MOVE_AT = {}
    except Exception as e:
        logging.warning("Failed to load cooldown state from %s: %s", filename, e)
        VM_LAST_MOVED_AT = {}
        VM_LAST_FAILED_MOVE_AT = {}


def save_cooldown_state(filename=COOLDOWN_STATE_FILE):
    try:
        prune_expired_cooldowns()
        directory = os.path.dirname(filename) or "."
        os.makedirs(directory, exist_ok=True)

        tmp_file = f"{filename}.tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "moved": VM_LAST_MOVED_AT,
                    "failed": VM_LAST_FAILED_MOVE_AT,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        os.replace(tmp_file, filename)

        logging.debug(
            "Saved %d moved and %d failed-move VM cooldown entries to %s.",
            len(VM_LAST_MOVED_AT),
            len(VM_LAST_FAILED_MOVE_AT),
            filename,
        )
    except Exception as e:
        logging.warning("Failed to save cooldown state to %s: %s", filename, e)


def prune_expired_cooldowns(now=None):
    if now is None:
        now = time.time()

    expired = [
        vm_id for vm_id, moved_at in VM_LAST_MOVED_AT.items()
        if now - moved_at >= VM_COOLDOWN_SECONDS
    ]
    failed_expired = [
        vm_id for vm_id, failed_at in VM_LAST_FAILED_MOVE_AT.items()
        if now - failed_at >= VM_FAILED_MOVE_COOLDOWN_SECONDS
    ]

    for vm_id in expired:
        VM_LAST_MOVED_AT.pop(vm_id, None)
    for vm_id in failed_expired:
        VM_LAST_FAILED_MOVE_AT.pop(vm_id, None)

    if expired:
        logging.debug("Pruned %d expired moved VM cooldown entries.", len(expired))
    if failed_expired:
        logging.debug("Pruned %d expired failed-move VM cooldown entries.", len(failed_expired))


# ============================================================================
# OPENSTACK
# ============================================================================

def get_openstack_connection():
    conn = openstack.connect(cloud='openstack')
    conn.compute.default_microversion = '2.87'
    return conn


def get_resource_field(resource, *names):
    for name in names:
        if isinstance(resource, dict):
            for key in (name, name.lower(), name.upper(), name.title()):
                value = resource.get(key)
                if value is not None:
                    return value
        else:
            value = getattr(resource, name, None)
            if value is not None:
                return value
    return None


def find_nova_compute_service(conn, host):
    try:
        services = list(conn.compute.services(host=host, binary="nova-compute"))
    except TypeError:
        services = list(conn.compute.services())
    except Exception as e:
        logging.warning("Failed to query nova-compute service for host %s: %s", host, e)
        return None

    for service in services:
        service_host = get_resource_field(service, "host", "Host")
        binary = get_resource_field(service, "binary", "Binary")
        if service_host == host and (binary in (None, "", "nova-compute")):
            return service

    return None


def is_nova_compute_service_ready(conn, host):
    service = find_nova_compute_service(conn, host)
    if not service:
        return False, f"nova-compute service not found for host {host}"

    status = str(get_resource_field(service, "status", "Status") or "UNKNOWN").lower()
    state = str(get_resource_field(service, "state", "State") or "UNKNOWN").lower()
    forced_down = bool(get_resource_field(service, "is_forced_down", "forced_down", "Forced Down") or False)
    disabled_reason = get_resource_field(service, "disabled_reason", "Disabled Reason")

    if status != "enabled":
        reason = f"nova-compute on {host} is not enabled (status={status})"
        if disabled_reason:
            reason = f"{reason}, disabled_reason={disabled_reason}"
        return False, reason

    if state != "up":
        return False, f"nova-compute on {host} is not up (state={state})"

    if forced_down:
        return False, f"nova-compute on {host} is forced down"

    return True, f"nova-compute on {host} is enabled and up"


# ============================================================================
# MATH / MODE
# ============================================================================

def mean_absolute_deviation(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum(abs(x - mean) for x in values) / len(values)


def calculate_adaptive_weights(cpu_signal, ram_signal, min_cpu_weight=MIN_CPU_WEIGHT, min_ram_weight=MIN_RAM_WEIGHT):
    total = cpu_signal + ram_signal
    if total <= 0:
        return 0.5, 0.5

    cpu_share = cpu_signal / total
    ram_share = ram_signal / total

    cpu_weight = max(cpu_share, min_cpu_weight)
    ram_weight = max(ram_share, min_ram_weight)

    weight_sum = cpu_weight + ram_weight
    cpu_weight /= weight_sum
    ram_weight /= weight_sum

    return cpu_weight, ram_weight


def calculate_pressure_signals(cpu_mad, ram_mad, cpu_mean, ram_mean):
    cpu_signal = (cpu_mad * PRESSURE_MAD_WEIGHT) + (cpu_mean * PRESSURE_MEAN_WEIGHT)
    ram_signal = (ram_mad * PRESSURE_MAD_WEIGHT) + (ram_mean * PRESSURE_MEAN_WEIGHT)
    return cpu_signal, ram_signal


def determine_adaptive_mode(cpu_mean, ram_mean, threshold=AUTO_PRESSURE_THRESHOLD):
    if cpu_mean > threshold or ram_mean > threshold:
        return "pressure"
    return "mad"


def determine_hotspot_mode(host_cpu_pct, host_ram_pct):
    max_cpu = max(host_cpu_pct.values()) if host_cpu_pct else 0.0
    max_ram = max(host_ram_pct.values()) if host_ram_pct else 0.0

    if max_ram > RAM_CRITICAL_THRESHOLD:
        return "ram_critical_hotspot", max_cpu, max_ram
    if max_ram > RAM_HOTSPOT_THRESHOLD:
        return "ram_hotspot", max_cpu, max_ram
    if max_cpu > CPU_HOTSPOT_THRESHOLD:
        return "cpu_hotspot", max_cpu, max_ram
    return "normal", max_cpu, max_ram


def choose_effective_mode(cpu_mean, ram_mean, host_cpu_pct, host_ram_pct):
    hotspot_mode, max_cpu, max_ram = determine_hotspot_mode(host_cpu_pct, host_ram_pct)
    if hotspot_mode != "normal":
        return hotspot_mode, max_cpu, max_ram
    return determine_adaptive_mode(cpu_mean, ram_mean), max_cpu, max_ram


def get_hotspot_source_host(mode, host_cpu_pct, host_ram_pct):
    if mode in ("ram_hotspot", "ram_critical_hotspot") and host_ram_pct:
        return max(host_ram_pct, key=host_ram_pct.get)
    if mode == "cpu_hotspot" and host_cpu_pct:
        return max(host_cpu_pct, key=host_cpu_pct.get)
    return None


def calculate_mode_weights(cpu_mad, ram_mad, cpu_mean, ram_mean, mode):
    if mode == "pressure":
        cpu_signal, ram_signal = calculate_pressure_signals(cpu_mad, ram_mad, cpu_mean, ram_mean)
        cpu_weight, ram_weight = calculate_adaptive_weights(cpu_signal, ram_signal)
        return cpu_weight, ram_weight, cpu_signal, ram_signal

    if mode == "ram_hotspot":
        cpu_signal, ram_signal = cpu_mad, ram_mad
        return 0.2, 0.8, cpu_signal, ram_signal

    if mode == "ram_critical_hotspot":
        cpu_signal, ram_signal = cpu_mad, ram_mad
        return 0.1, 0.9, cpu_signal, ram_signal

    if mode == "cpu_hotspot":
        cpu_signal, ram_signal = cpu_mad, ram_mad
        return 0.8, 0.2, cpu_signal, ram_signal

    cpu_signal, ram_signal = cpu_mad, ram_mad
    cpu_weight, ram_weight = calculate_adaptive_weights(cpu_signal, ram_signal)
    return cpu_weight, ram_weight, cpu_signal, ram_signal


def is_hotspot_guardrail_satisfied(mode, new_host_cpu, new_host_ram, dst_host, current_max_cpu, current_max_ram):
    """
    For hotspot modes:
    - hotspot metric must go down
    - destination must not become hotspot
    """
    if mode in ("ram_hotspot", "ram_critical_hotspot"):
        new_max_ram = max(new_host_ram.values())
        dst_ram = new_host_ram[dst_host]
        if new_max_ram >= current_max_ram:
            return False
        if dst_ram >= RAM_HOTSPOT_THRESHOLD:
            return False

    elif mode == "cpu_hotspot":
        new_max_cpu = max(new_host_cpu.values())
        dst_cpu = new_host_cpu[dst_host]
        if new_max_cpu >= current_max_cpu:
            return False
        if dst_cpu >= CPU_HOTSPOT_THRESHOLD:
            return False

    return True

def format_mode_label(mode):
    if mode == "ram_hotspot":
        return f"RAM_HOTSPOT (host RAM above {RAM_HOTSPOT_THRESHOLD:.1f}%)"

    if mode == "ram_critical_hotspot":
        return f"RAM_CRITICAL_HOTSPOT (host RAM above {RAM_CRITICAL_THRESHOLD:.1f}%)"

    if mode == "cpu_hotspot":
        return f"CPU_HOTSPOT (host CPU above {CPU_HOTSPOT_THRESHOLD:.1f}%)"

    if mode == "pressure":
        return f"PRESSURE (cluster average above {AUTO_PRESSURE_THRESHOLD:.1f}%)"

    return "BALANCING (no pressure or hotspot)"

def format_cluster_status(status):
    labels = {
        "RAM_HOTSPOT_OPTIMIZABLE": "RAM_HOTSPOT_OPTIMIZABLE",
        "RAM_HOTSPOT_NO_VALID_MOVE": "RAM_HOTSPOT_NO_VALID_MOVE",
        "CPU_HOTSPOT_OPTIMIZABLE": "CPU_HOTSPOT_OPTIMIZABLE",
        "CPU_HOTSPOT_NO_VALID_MOVE": "CPU_HOTSPOT_NO_VALID_MOVE",

        "RAM_PRESSURE": "RAM_PRESSURE",
        "CPU_PRESSURE": "CPU_PRESSURE",
        "PRESSURE_NO_VALID_MOVE": "PRESSURE_NO_VALID_MOVE",

        "BALANCED": "BALANCED",
        "BALANCED_OPTIMIZABLE": "BALANCED_OPTIMIZABLE",

        "UNBALANCED": "UNBALANCED",
        "UNBALANCED_NO_VALID_MOVE": "UNBALANCED_NO_VALID_MOVE",

        "UNKNOWN": "UNKNOWN",
    }
    return labels.get(status, status)


# ============================================================================
# SERVER GROUPS / CACHES
# ============================================================================

def build_server_group_index(conn, vm_ids):
    vm_ids = set(vm_ids)
    vm_to_groups = {vm_id: [] for vm_id in vm_ids}

    try:
        for sg in conn.compute.server_groups(all_projects=True):
            members = list(getattr(sg, "member_ids", []) or [])
            if not members:
                continue

            if not any(m in vm_ids for m in members):
                continue

            policies = getattr(sg, "policies", None)
            if policies and isinstance(policies, (list, tuple)) and policies:
                policy = policies[0]
            else:
                policy = getattr(sg, "policy", None)

            info = {
                "id": getattr(sg, "id", None),
                "name": getattr(sg, "name", None),
                "policy": policy,
                "members": members,
            }

            for m in members:
                if m in vm_ids:
                    vm_to_groups[m].append(info)

        logging.info(
            "Built server group index for %d VMs (some may have 0 groups).",
            len(vm_to_groups),
        )
    except Exception as e:
        logging.warning(
            "Failed to build server group index, continuing WITHOUT server-group constraints: %s",
            e,
        )
        return {vm_id: [] for vm_id in vm_ids}

    return vm_to_groups


def is_server_group_move_compliant(vm_id, src_host, dst_host, vm_host_map, vm_to_groups):
    groups = vm_to_groups.get(vm_id) or []
    if not groups:
        return True

    for g in groups:
        policy = g.get("policy")
        members = g.get("members") or []
        if not policy or not members:
            continue

        if policy == "affinity":
            ref_host = None
            for member in members:
                host = dst_host if member == vm_id else vm_host_map.get(member)
                if host is None:
                    continue
                if ref_host is None:
                    ref_host = host
                elif host != ref_host:
                    logging.debug(
                        "Reject move of %s %s→%s: affinity group %s/%s would span hosts (%s vs %s).",
                        vm_id, src_host, dst_host, g.get("name"), g.get("id"), ref_host, host,
                    )
                    return False

        elif policy == "anti-affinity":
            seen_hosts = set()
            for member in members:
                host = dst_host if member == vm_id else vm_host_map.get(member)
                if host is None:
                    continue
                if host in seen_hosts:
                    logging.debug(
                        "Reject move of %s %s→%s: anti-affinity group %s/%s would have two members on host %s.",
                        vm_id, src_host, dst_host, g.get("name"), g.get("id"), host,
                    )
                    return False
                seen_hosts.add(host)

    return True


def build_server_cache(conn, hosts_in_agg):
    server_cache = {}
    hosts_set = set(hosts_in_agg)

    try:
        for server in conn.compute.servers(all_projects=True, details=True):
            host = getattr(server, "hypervisor_hostname", None)
            if host in hosts_set:
                server_cache[server.id] = server
        logging.info(
            "Built server cache for %d servers across %d hosts.",
            len(server_cache),
            len(hosts_in_agg),
        )
    except Exception as e:
        logging.warning("Failed to build server cache: %s", e)

    return server_cache


def get_server_project_id(server):
    if not server:
        return None

    project_id = get_resource_field(server, "project_id", "tenant_id")
    if isinstance(project_id, dict):
        return project_id.get("id") or project_id.get("project_id") or project_id.get("tenant_id")

    return project_id


def resolve_project_name(conn, project_id, project_cache=None):
    if not project_id:
        return None

    if project_cache is not None and project_id in project_cache:
        return project_cache[project_id]

    project_name = None
    try:
        project = conn.identity.get_project(project_id)
        project_name = get_resource_field(project, "name", "Name")
    except Exception as e:
        logging.debug("Failed to resolve project name for project %s: %s", project_id, e)

    if not project_name:
        project_name = project_id

    if project_cache is not None:
        project_cache[project_id] = project_name

    return project_name


def build_vm_metric_queries(hosts_in_agg):
    """Build the aggregate-scoped Prometheus queries for VM CPU and RAM usage."""
    host_re = "|".join(hosts_in_agg)
    mem_query = (
        f'(100 - (avg_over_time(libvirt_domain_stat_memory_usable_bytes{{job="{PROMETHEUS_LIBVIRT_JOB}", host=~"{host_re}"}}[5m]) '
        f'/ avg_over_time(libvirt_domain_stat_memory_available_bytes{{job="{PROMETHEUS_LIBVIRT_JOB}", host=~"{host_re}"}}[5m])) * 100)'
    )
    cpu_query = (
        f'(irate(libvirt_domain_info_cpu_time_seconds_total{{job="{PROMETHEUS_LIBVIRT_JOB}", host=~"{host_re}"}}[5m]) '
        f'/ libvirt_domain_info_virtual_cpus{{job="{PROMETHEUS_LIBVIRT_JOB}", host=~"{host_re}"}}) * 100'
    )
    return mem_query, cpu_query


def get_default_vm_metadata(vm_id):
    """Return the conservative VM metadata defaults already used by the scorer."""
    return {
        "name": vm_id,
        "vcpus": 1,
        "ram_gb": 2.0,
        "project_id": None,
        "project_name": None,
    }


def get_vm_metadata(conn, vm_id, server_cache=None, project_cache=None):
    """Resolve VM metadata while preserving the existing fallback behavior."""
    try:
        server = server_cache.get(vm_id) if server_cache else conn.compute.get_server(vm_id)
        if not server:
            return get_default_vm_metadata(vm_id)

        flavor = getattr(server, "flavor", {}) or {}
        ram_mb = flavor.get("ram")
        name = getattr(server, "name", vm_id)
        vcpus = flavor.get("vcpus")
        project_id = get_server_project_id(server)
        project_name = resolve_project_name(
            conn,
            project_id,
            project_cache=project_cache,
        )

        if vcpus is None:
            vcpus = 1
        if ram_mb is None:
            ram_mb = 2048
        return {
            "name": name,
            "vcpus": vcpus,
            "ram_gb": float(ram_mb) / 1024.0,
            "project_id": project_id,
            "project_name": project_name,
        }
    except Exception as e:
        logging.debug(
            "Failed to resolve metadata for VM %s; using default values: %s",
            vm_id,
            e,
        )

    return get_default_vm_metadata(vm_id)


def group_vm_metrics_by_host(vm_metrics):
    """Group VM metrics by source host to avoid repeated full scans."""
    metrics_by_host = defaultdict(dict)
    for vm_id, metric in vm_metrics.items():
        host = metric.get("host")
        if host is not None:
            metrics_by_host[host][vm_id] = metric
    return metrics_by_host


# ============================================================================
# VM METRICS / RESOURCES
# ============================================================================

def get_all_vm_metrics(cfg, hosts_in_agg):
    """Fetch per-VM CPU and RAM usage for the selected aggregate hosts."""
    mem_query, cpu_query = build_vm_metric_queries(hosts_in_agg)

    mem_data = do_query(cfg.prometheus_query_url, mem_query)
    cpu_data = do_query(cfg.prometheus_query_url, cpu_query)

    vm_metrics = {}

    try:
        for entry in get_prometheus_result_rows(
            mem_data,
            "Prometheus returned malformed VM memory results",
        ):
            vm_id = entry["metric"].get("instanceId")
            host = entry["metric"].get("host")
            if vm_id and host:
                vm_metrics.setdefault(vm_id, {"host": host, "cpu_pct": 0.0, "ram_pct": 0.0})
                vm_metrics[vm_id]["ram_pct"] = float(entry["value"][1])

        for entry in get_prometheus_result_rows(
            cpu_data,
            "Prometheus returned malformed VM CPU results",
        ):
            vm_id = entry["metric"].get("instanceId")
            host = entry["metric"].get("host")
            if vm_id and host:
                vm_metrics.setdefault(vm_id, {"host": host, "cpu_pct": 0.0, "ram_pct": 0.0})
                vm_metrics[vm_id]["cpu_pct"] = float(entry["value"][1])
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise PrometheusError(f"Prometheus returned malformed VM metric results: {e}")

    logging.info(
        "Fetched aggregate VM metrics for %d VMs across %d hosts.",
        len(vm_metrics),
        len(hosts_in_agg),
    )

    return vm_metrics


def calculate_vm_scores(conn, host, cfg, server_cache=None, vm_metrics=None, project_cache=None):
    """Build VM score inputs for a single host using already-fetched metrics."""
    if vm_metrics is None:
        raise ValueError("vm_metrics is required")

    vm_names = {}
    vm_details = {}

    for vm_id, metric in vm_metrics.items():
        if metric["host"] != host:
            continue

        metadata = get_vm_metadata(
            conn,
            vm_id,
            server_cache=server_cache,
            project_cache=project_cache,
        )

        cpu_pct = metric.get("cpu_pct", 0.0)
        ram_pct = metric.get("ram_pct", 0.0)

        vm_names[vm_id] = metadata["name"]
        vm_details[vm_id] = {
            "cpu_pct": cpu_pct,
            "ram_pct": ram_pct,
            "vcpus": metadata["vcpus"],
            "ram_gb": metadata["ram_gb"],
            "host": host,
            "project_id": metadata["project_id"],
            "project_name": metadata["project_name"],
        }

    return vm_names, vm_details


def get_host_resources_map(conn, cfg):
    host_resources = {}
    cpu_ratio_map, mem_ratio_map = get_allocation_ratios(cfg)

    for hypervisor in conn.compute.hypervisors():
        full = conn.compute.get_hypervisor(hypervisor.id)
        name = getattr(full, "name", None)
        total_cores = getattr(full, "vcpus", 0) or 0
        used_cores = getattr(full, "vcpus_used", 0) or 0

        mem_mb = getattr(full, "memory_size", 0) or 0
        used_mem_mb = getattr(full, "memory_used", 0) or 0

        if not name:
            continue

        cpu_ratio = cpu_ratio_map.get(name, 7.0)
        mem_ratio = mem_ratio_map.get(name, 1.2)

        free_cores = (total_cores * cpu_ratio) - used_cores
        free_cores = max(free_cores, 0)

        total_ram_gb = ((float(mem_mb) * mem_ratio) - 24576) / 1024.0
        free_ram_gb = ((float(mem_mb) * mem_ratio) - 24576 - float(used_mem_mb)) / 1024.0

        total_ram_gb = max(total_ram_gb, 0.0)
        free_ram_gb = max(free_ram_gb, 0.0)

        host_resources[name] = {
            "total_cores": int(total_cores),
            "free_cores": float(free_cores),
            "total_ram_gb": float(total_ram_gb),
            "free_ram_gb": float(free_ram_gb),
        }

    return host_resources


# ============================================================================
# MOVE SEARCH / CLASSIFICATION
# ============================================================================

def find_best_move(
    mode,
    source_hosts,
    hosts_in_agg,
    host_cpu_pct,
    host_ram_pct,
    host_cores,
    host_free_cores,
    host_ram,
    host_free_ram,
    vm_host_map,
    all_vm_details,
    vm_to_groups,
    enforce_server_groups,
    baseline_weighted_mad,
    cpu_weight,
    ram_weight,
    dest_ready_hosts=None,
):
    """Search the best VM move candidate while preserving current scoring behavior."""
    best_move = None
    best_improvement = 0.0

    current_max_cpu = max(host_cpu_pct.values()) if host_cpu_pct else 0.0
    current_max_ram = max(host_ram_pct.values()) if host_ram_pct else 0.0

    for vm_id, src_host in list(vm_host_map.items()):
        if src_host not in source_hosts:
            continue

        last_moved_at = VM_LAST_MOVED_AT.get(vm_id)
        if last_moved_at is not None:
            cooldown_left = VM_COOLDOWN_SECONDS - (time.time() - last_moved_at)
            if cooldown_left > 0:
                logging.debug(
                    "Skip VM %s due to cooldown: %.0f seconds remaining.",
                    vm_id,
                    cooldown_left,
                )
                continue

        last_failed_at = VM_LAST_FAILED_MOVE_AT.get(vm_id)
        if last_failed_at is not None:
            cooldown_left = VM_FAILED_MOVE_COOLDOWN_SECONDS - (time.time() - last_failed_at)
            if cooldown_left > 0:
                logging.debug(
                    "Skip VM %s due to failed-move cooldown: %.0f seconds remaining.",
                    vm_id,
                    cooldown_left,
                )
                continue

        details = all_vm_details[vm_id]
        vcpus = details.get("vcpus", 1)
        vm_cpu_pct = details.get("cpu_pct", 0.0)
        vm_ram_gb = details.get("ram_gb", 2.0)
        vm_ram_pct = details.get("ram_pct", 0.0)

        real_vm_cores = vcpus * (vm_cpu_pct / 100.0)
        real_vm_ram = vm_ram_gb * (vm_ram_pct / 100.0)

        for dst_host in hosts_in_agg:
            if dst_host == src_host:
                continue

            if dest_ready_hosts is not None and dst_host not in dest_ready_hosts:
                logging.debug(
                    "Skip move of VM %s from %s to %s: destination nova-compute is not ready.",
                    vm_id,
                    src_host,
                    dst_host,
                )
                continue

            host_cpu_avail = host_free_cores[dst_host]
            host_ram_avail = host_free_ram[dst_host]

            # Keep feasibility aligned with scheduler placement logic:
            # destination must have enough allocatable capacity for the VM flavor,
            # even though balancing impact is scored using real observed usage.
            if host_cpu_avail < vcpus or host_ram_avail < vm_ram_gb:
                continue

            if enforce_server_groups:
                if not is_server_group_move_compliant(vm_id, src_host, dst_host, vm_host_map, vm_to_groups):
                    continue

            cpu_impact = (real_vm_cores / host_cores[src_host]) * 100
            ram_impact = (real_vm_ram / host_ram[src_host]) * 100

            new_host_cpu = host_cpu_pct.copy()
            new_host_ram = host_ram_pct.copy()
            new_host_cpu[src_host] -= cpu_impact
            new_host_cpu[dst_host] += cpu_impact
            new_host_ram[src_host] -= ram_impact
            new_host_ram[dst_host] += ram_impact

            if mode in ("ram_hotspot", "ram_critical_hotspot", "cpu_hotspot"):
                if not is_hotspot_guardrail_satisfied(
                    mode, new_host_cpu, new_host_ram, dst_host, current_max_cpu, current_max_ram
                ):
                    continue

            mad_cpu = mean_absolute_deviation(list(new_host_cpu.values()))
            mad_ram = mean_absolute_deviation(list(new_host_ram.values()))
            new_weighted_mad = (mad_cpu * cpu_weight) + (mad_ram * ram_weight)
            mad_improvement = baseline_weighted_mad - new_weighted_mad

            if mad_improvement > best_improvement:
                best_improvement = mad_improvement
                best_move = BestMove(
                    vm_id=vm_id,
                    src_host=src_host,
                    dst_host=dst_host,
                    cpu_impact=cpu_impact,
                    ram_impact=ram_impact,
                    new_host_cpu=new_host_cpu,
                    new_host_ram=new_host_ram,
                    real_vm_cores=real_vm_cores,
                    vcpus=vcpus,
                    real_vm_ram=real_vm_ram,
                    vm_ram_gb=vm_ram_gb,
                    host_cpu_avail=host_cpu_avail,
                    host_ram_avail=host_ram_avail,
                    new_weighted_mad=new_weighted_mad,
                    mad_improvement=mad_improvement,
                    mad_cpu=mad_cpu,
                    mad_ram=mad_ram,
                )

    return best_move, best_improvement


def classify_cluster_detail_status(mode, max_cpu, max_ram, baseline_cpu_mad, baseline_ram_mad, has_valid_move):
    is_unbalanced_by_mad = (
        baseline_cpu_mad > TARGET_AVG_MAD or
        baseline_ram_mad > TARGET_AVG_MAD
    )

    if mode == "ram_hotspot" or mode == "ram_critical_hotspot":
        if has_valid_move:
            return "RAM_HOTSPOT_OPTIMIZABLE"
        return "RAM_HOTSPOT_NO_VALID_MOVE"

    if mode == "cpu_hotspot":
        if has_valid_move:
            return "CPU_HOTSPOT_OPTIMIZABLE"
        return "CPU_HOTSPOT_NO_VALID_MOVE"

    if mode == "pressure":
        if has_valid_move:
            return "RAM_PRESSURE" if max_ram >= max_cpu else "CPU_PRESSURE"
        return "PRESSURE_NO_VALID_MOVE"

    if is_unbalanced_by_mad:
        if has_valid_move:
            return "UNBALANCED"
        return "UNBALANCED_NO_VALID_MOVE"

    if has_valid_move:
        return "BALANCED_OPTIMIZABLE"

    return "BALANCED"


def map_detail_status_to_severity(detail_status, reason=None):
    # Healthy or near-optimal state
    if detail_status in ("BALANCED", "BALANCED_OPTIMIZABLE"):
        return "OK"

    # Imbalance or pressure exists but can still be improved (valid move available)
    # → do not escalate to CRITICAL to avoid alert noise
    if detail_status in (
        "UNBALANCED",
        "RAM_PRESSURE",
        "CPU_PRESSURE",
        "RAM_HOTSPOT_OPTIMIZABLE",
        "CPU_HOTSPOT_OPTIMIZABLE",
    ):
        return "WARNING"

    # High pressure or hotspot with no valid move → real issue
    # → requires immediate attention
    if detail_status in (
        "PRESSURE_NO_VALID_MOVE",
        "RAM_HOTSPOT_NO_VALID_MOVE",
        "CPU_HOTSPOT_NO_VALID_MOVE",
    ):
        return "CRITICAL"

    # Special case: unbalanced but no valid move
    if detail_status == "UNBALANCED_NO_VALID_MOVE":
        # Hard block: no VM can be moved at all (resource/scheduler constraints)
        if reason == "no_technical_candidate":
            return "CRITICAL"

        # Soft block: moves exist but do not meet improvement threshold
        # → keep as WARNING to reduce alert noise
        return "WARNING"

    # Fallback
    return "UNKNOWN"


def map_severity_to_exit_code(severity):
    if severity == "OK":
        return ICINGA_OK
    if severity == "WARNING":
        return ICINGA_WARNING
    if severity == "CRITICAL":
        return ICINGA_CRITICAL
    return ICINGA_UNKNOWN


def derive_reason(has_candidate, has_valid_move, mode=None):
    if has_valid_move:
        return None

    if not has_candidate:
        return "no_technical_candidate"

    if mode in ("ram_hotspot", "ram_critical_hotspot", "cpu_hotspot"):
        return "no_hotspot_candidate_above_improvement_threshold"

    if mode == "pressure":
        return "no_pressure_candidate_above_improvement_threshold"

    return "no_candidate_above_improvement_threshold"


# ============================================================================
# EVALUATION (SINGLE SOURCE OF TRUTH)
# ============================================================================

def build_host_resource_views(host_resources, hosts_in_agg):
    """Build per-host capacity maps using the existing fallback defaults."""
    return (
        {h: host_resources.get(h, {}).get("total_cores", 64) for h in hosts_in_agg},
        {h: host_resources.get(h, {}).get("free_cores", 64) for h in hosts_in_agg},
        {h: host_resources.get(h, {}).get("total_ram_gb", 256.0) for h in hosts_in_agg},
        {h: host_resources.get(h, {}).get("free_ram_gb", 256.0) for h in hosts_in_agg},
    )


def build_effective_host_metrics(hosts_in_agg, cpu_metrics, mem_metrics, agg_name):
    """Return host CPU/RAM maps for hosts with both metrics available."""
    host_cpu_pct = {}
    host_ram_pct = {}

    for host in hosts_in_agg:
        cpu_entry = cpu_metrics.get(host)
        mem_entry = mem_metrics.get(host)
        if not cpu_entry or not mem_entry:
            logging.warning("Host %s in aggregate %s missing CPU or RAM metrics, skipping.", host, agg_name)
            continue
        host_cpu_pct[host] = cpu_entry["value"]
        host_ram_pct[host] = mem_entry["value"]

    return host_cpu_pct, host_ram_pct


def get_destination_readiness(conn, effective_hosts, agg_name):
    """Return destination-ready hosts and exclusion reasons."""
    dest_ready_hosts = set()
    dest_not_ready_reasons = {}

    for host in effective_hosts:
        ready, reason = is_nova_compute_service_ready(conn, host)
        if ready:
            dest_ready_hosts.add(host)
        else:
            dest_not_ready_reasons[host] = reason
            logging.warning(
                "Host %s excluded as migration destination for aggregate %s: %s",
                host,
                agg_name,
                reason,
            )

    return dest_ready_hosts, dest_not_ready_reasons


def build_aggregate_vm_inventory(conn, effective_hosts, cfg, server_cache, vm_metrics, project_cache):
    """Build aggregate-wide VM naming, detail, and host placement maps."""
    all_vm_names = {}
    all_vm_details = {}
    vm_host_map = {}
    vm_metrics_by_host = group_vm_metrics_by_host(vm_metrics)

    for src_host in effective_hosts:
        host_vm_metrics = vm_metrics_by_host.get(src_host, {})
        vm_names, vm_details = calculate_vm_scores(
            conn,
            src_host,
            cfg,
            server_cache=server_cache,
            vm_metrics=host_vm_metrics,
            project_cache=project_cache,
        )
        for vm_id, details in vm_details.items():
            all_vm_names[vm_id] = vm_names[vm_id]
            all_vm_details[vm_id] = details
            vm_host_map[vm_id] = src_host

    return all_vm_names, all_vm_details, vm_host_map


def compute_baseline_metrics(host_cpu_pct, host_ram_pct):
    """Compute cluster baseline means, MADs, and adaptive weights."""
    cpu_mean = sum(host_cpu_pct.values()) / len(host_cpu_pct)
    ram_mean = sum(host_ram_pct.values()) / len(host_ram_pct)
    baseline_cpu_mad = mean_absolute_deviation(list(host_cpu_pct.values()))
    baseline_ram_mad = mean_absolute_deviation(list(host_ram_pct.values()))
    mode, max_cpu, max_ram = choose_effective_mode(cpu_mean, ram_mean, host_cpu_pct, host_ram_pct)
    cpu_weight, ram_weight, cpu_signal, ram_signal = calculate_mode_weights(
        baseline_cpu_mad, baseline_ram_mad, cpu_mean, ram_mean, mode
    )
    baseline_weighted_mad = (baseline_cpu_mad * cpu_weight) + (baseline_ram_mad * ram_weight)

    return {
        "cpu_mean": cpu_mean,
        "ram_mean": ram_mean,
        "baseline_cpu_mad": baseline_cpu_mad,
        "baseline_ram_mad": baseline_ram_mad,
        "mode": mode,
        "max_cpu": max_cpu,
        "max_ram": max_ram,
        "cpu_weight": cpu_weight,
        "ram_weight": ram_weight,
        "cpu_signal": cpu_signal,
        "ram_signal": ram_signal,
        "baseline_weighted_mad": baseline_weighted_mad,
    }


def determine_improvement_threshold(mode, baseline_weighted_mad):
    """Return the current dynamic threshold used to accept a move."""
    if mode in ("ram_hotspot", "ram_critical_hotspot", "cpu_hotspot"):
        return MIN_IMPROVEMENT_THRESHOLD
    return max(MIN_IMPROVEMENT_THRESHOLD, baseline_weighted_mad * 0.05)


def search_best_move_for_state(
    mode,
    effective_hosts,
    host_cpu_pct,
    host_ram_pct,
    host_cores,
    host_free_cores,
    host_ram,
    host_free_ram,
    vm_host_map,
    all_vm_details,
    vm_to_groups,
    enforce_server_groups,
    baseline_weighted_mad,
    cpu_weight,
    ram_weight,
    dest_ready_hosts,
    improvement_threshold,
):
    """Run the current hotspot-first/full-search strategy unchanged."""
    hotspot_source_host = get_hotspot_source_host(mode, host_cpu_pct, host_ram_pct)

    def search_move(source_hosts, scope_label):
        move, improvement = find_best_move(
            mode=mode,
            source_hosts=source_hosts,
            hosts_in_agg=effective_hosts,
            host_cpu_pct=host_cpu_pct,
            host_ram_pct=host_ram_pct,
            host_cores=host_cores,
            host_free_cores=host_free_cores,
            host_ram=host_ram,
            host_free_ram=host_free_ram,
            vm_host_map=vm_host_map,
            all_vm_details=all_vm_details,
            vm_to_groups=vm_to_groups,
            enforce_server_groups=enforce_server_groups,
            baseline_weighted_mad=baseline_weighted_mad,
            cpu_weight=cpu_weight,
            ram_weight=ram_weight,
            dest_ready_hosts=dest_ready_hosts,
        )
        return move, improvement, scope_label

    best_move = None
    best_improvement = 0.0
    search_scope = "FULL"

    if hotspot_source_host:
        move1, improvement1, scope1 = search_move(
            [hotspot_source_host],
            f"HOTSPOT-FIRST ({hotspot_source_host})",
        )
        has_valid_move1 = bool(move1 and improvement1 > improvement_threshold)

        if has_valid_move1:
            best_move, best_improvement, search_scope = move1, improvement1, scope1
        else:
            move2, improvement2, scope2 = search_move(
                effective_hosts,
                f"FALLBACK-FULL (after {hotspot_source_host})",
            )
            if improvement2 >= improvement1:
                best_move, best_improvement, search_scope = move2, improvement2, scope2
            else:
                best_move, best_improvement, search_scope = move1, improvement1, scope1
    else:
        best_move, best_improvement, search_scope = search_move(effective_hosts, "FULL")

    return best_move, best_improvement, search_scope, hotspot_source_host


def evaluate_aggregate_state(cfg, conn, agg_name, hosts_in_agg, enforce_server_groups=True):
    """Evaluate one aggregate and return the full decision state used by runtime and monitor modes."""
    prune_expired_cooldowns()
    host_resources = get_host_resources_map(conn, cfg)
    host_cores, host_free_cores, host_ram, host_free_ram = build_host_resource_views(
        host_resources,
        hosts_in_agg,
    )

    cpu_metrics = get_metrics(cfg.prometheus_query_url, cfg.prometheus_query_cpu_used)
    mem_metrics = get_metrics(cfg.prometheus_query_url, cfg.prometheus_query_mem_used)
    host_cpu_pct, host_ram_pct = build_effective_host_metrics(
        hosts_in_agg,
        cpu_metrics,
        mem_metrics,
        agg_name,
    )

    effective_hosts = list(host_cpu_pct.keys())
    if len(effective_hosts) < 2:
        return {
            "aggregate": agg_name,
            "error": "less_than_2_hosts_with_metrics",
            "hosts_in_agg": effective_hosts,
        }

    dest_ready_hosts, dest_not_ready_reasons = get_destination_readiness(
        conn,
        effective_hosts,
        agg_name,
    )

    server_cache = build_server_cache(conn, effective_hosts)
    project_cache = {}
    vm_metrics = get_all_vm_metrics(cfg, effective_hosts)
    all_vm_names, all_vm_details, vm_host_map = build_aggregate_vm_inventory(
        conn,
        effective_hosts,
        cfg,
        server_cache,
        vm_metrics,
        project_cache,
    )

    vm_ids = set(vm_host_map.keys())
    if enforce_server_groups:
        vm_to_groups = build_server_group_index(conn, vm_ids)
    else:
        vm_to_groups = {vm_id: [] for vm_id in vm_ids}

    baseline = compute_baseline_metrics(host_cpu_pct, host_ram_pct)
    cpu_mean = baseline["cpu_mean"]
    ram_mean = baseline["ram_mean"]
    baseline_cpu_mad = baseline["baseline_cpu_mad"]
    baseline_ram_mad = baseline["baseline_ram_mad"]
    mode = baseline["mode"]
    max_cpu = baseline["max_cpu"]
    max_ram = baseline["max_ram"]
    cpu_weight = baseline["cpu_weight"]
    ram_weight = baseline["ram_weight"]
    cpu_signal = baseline["cpu_signal"]
    ram_signal = baseline["ram_signal"]
    baseline_weighted_mad = baseline["baseline_weighted_mad"]
    improvement_threshold = determine_improvement_threshold(mode, baseline_weighted_mad)
    best_move, best_improvement, search_scope, hotspot_source_host = search_best_move_for_state(
        mode,
        effective_hosts,
        host_cpu_pct,
        host_ram_pct,
        host_cores,
        host_free_cores,
        host_ram,
        host_free_ram,
        vm_host_map,
        all_vm_details,
        vm_to_groups,
        enforce_server_groups,
        baseline_weighted_mad,
        cpu_weight,
        ram_weight,
        dest_ready_hosts,
        improvement_threshold,
    )

    has_candidate = bool(best_move)

    has_valid_move = bool(
        best_move and best_improvement >= improvement_threshold
    )

    detail_status = classify_cluster_detail_status(
        mode=mode,
        max_cpu=max_cpu,
        max_ram=max_ram,
        baseline_cpu_mad=baseline_cpu_mad,
        baseline_ram_mad=baseline_ram_mad,
        has_valid_move=has_valid_move,
    )

    reason = derive_reason(has_candidate, has_valid_move, mode)
    severity = map_detail_status_to_severity(detail_status, reason)
    exit_code = map_severity_to_exit_code(severity)

    state = {
        "aggregate": agg_name,
        "hosts_in_agg": effective_hosts,
        "host_cpu_pct": host_cpu_pct,
        "host_ram_pct": host_ram_pct,
        "host_cores": host_cores,
        "host_free_cores": host_free_cores,
        "host_ram": host_ram,
        "host_free_ram": host_free_ram,
        "dest_ready_hosts": sorted(dest_ready_hosts),
        "dest_not_ready_reasons": dest_not_ready_reasons,
        "all_vm_names": all_vm_names,
        "all_vm_details": all_vm_details,
        "vm_host_map": vm_host_map,
        "vm_to_groups": vm_to_groups,
        "cpu_mean": cpu_mean,
        "ram_mean": ram_mean,
        "baseline_cpu_mad": baseline_cpu_mad,
        "baseline_ram_mad": baseline_ram_mad,
        "baseline_weighted_mad": baseline_weighted_mad,
        "improvement_threshold": improvement_threshold,
        "mode": mode,
        "max_cpu": max_cpu,
        "max_ram": max_ram,
        "cpu_weight": cpu_weight,
        "ram_weight": ram_weight,
        "cpu_signal": cpu_signal,
        "ram_signal": ram_signal,
        "best_move": best_move,
        "best_improvement": best_improvement,
        "has_candidate": has_candidate,
        "has_valid_move": has_valid_move,
        "search_scope": search_scope,
        "hotspot_source_host": hotspot_source_host,
        "detail_status": detail_status,
        "reason": reason,
        "severity": severity,
        "exit_code": exit_code,
    }
    state["message"] = build_monitor_message(state)
    return state


# ============================================================================
# MIGRATION MONITOR
# ============================================================================

def reset_error_server_to_active(conn, server_id, vm_name):
    try:
        server = conn.compute.get_server(server_id)
    except Exception as e:
        logging.warning(
            "Failed to check VM %s (%s) state after migration failure: %s",
            vm_name,
            server_id,
            e,
        )
        return False

    status = str(getattr(server, "status", "") or "").upper()
    if status != "ERROR":
        logging.debug(
            "VM %s (%s) state after migration failure is %s; no reset needed.",
            vm_name,
            server_id,
            status or "UNKNOWN",
        )
        return False

    try:
        conn.compute.reset_server_state(server, state="active")
        logging.warning(
            "VM %s (%s) was in ERROR after migration failure; reset state to ACTIVE.",
            vm_name,
            server_id,
        )
        return True
    except Exception as e:
        logging.warning(
            "Failed to reset VM %s (%s) state from ERROR to ACTIVE: %s",
            vm_name,
            server_id,
            e,
        )
        return False


def get_server_current_host(server):
    if isinstance(server, dict):
        return (
            server.get("OS-EXT-SRV-ATTR:host") or
            server.get("hypervisor_hostname") or
            server.get("compute_host")
        )

    return (
        getattr(server, "OS-EXT-SRV-ATTR:host", None) or
        getattr(server, "hypervisor_hostname", None) or
        getattr(server, "compute_host", None)
    )


def get_server_status(server):
    if isinstance(server, dict):
        return str(server.get("status") or server.get("Status") or "UNKNOWN").upper()

    return str(getattr(server, "status", None) or "UNKNOWN").upper()


def get_migration_identifier(migration):
    """Return the migration identifier regardless of resource shape."""
    return getattr(migration, "id", None) or getattr(migration, "uuid", None)


def get_migration_status(migration):
    """Return the migration status regardless of resource shape."""
    if migration is None:
        return None
    return getattr(migration, "status", None) or (
        migration['status'] if isinstance(migration, dict) and 'status' in migration else 'n/a'
    )


def find_tracked_migration(migrations, migration_id):
    """Return the tracked migration, or the newest one as fallback."""
    if migration_id:
        for migration in migrations:
            if get_migration_identifier(migration) == migration_id:
                return migration
    if migrations:
        return migrations[-1]
    return None


def get_migration_progress(server, migration):
    """Return the best available migration progress percentage."""
    if hasattr(server, 'progress') and server.progress is not None:
        return server.progress
    if hasattr(migration, 'progress') and migration.progress is not None:
        return migration.progress
    if isinstance(migration, dict) and 'progress' in migration and migration['progress'] is not None:
        return migration['progress']
    return None


def map_migration_state(status):
    """Map raw migration status to the existing display state labels."""
    normalized_status = status.lower() if status else ""
    if normalized_status == "running (post-copy)":
        return "IN PROGRESS"
    if normalized_status in PROGRESS_STATUSES:
        return "IN PROGRESS"
    if normalized_status in SUCCESS_STATUSES:
        return "SUCCESS"
    if normalized_status in FAILED_STATUSES:
        return "FAILED"
    return f"UNKNOWN ({status})"


def get_server_migration_result(vm_name, current_host, server_status, src_host=None, dst_host=None, verbose=True):
    """Check server placement/state shortcuts that indicate migration completion or failure."""
    if dst_host and current_host == dst_host and server_status == "ACTIVE":
        if verbose:
            print(f"\nMigration completed: VM {vm_name} is now ACTIVE on {dst_host}.")
        return MIGRATION_RESULT_SUCCESS

    if src_host and current_host != src_host and current_host not in ("", None, "UNKNOWN"):
        if server_status == "ACTIVE":
            if verbose:
                print(f"\nMigration completed: VM {vm_name} moved from {src_host} to {current_host}.")
            return MIGRATION_RESULT_SUCCESS

    if server_status == "ERROR":
        if verbose:
            print(f"\nMigration failed: VM {vm_name} entered ERROR state.")
        return MIGRATION_RESULT_FAILED

    return None


def update_active_stuck_state(
    vm_name,
    current_host,
    server_status,
    src_host,
    active_stuck_start,
    active_stuck_timeout,
    verbose=True,
):
    """Preserve the current stuck-on-source detection behavior."""
    if src_host and current_host == src_host and server_status == "ACTIVE":
        if active_stuck_start is None:
            return time.time(), None
        if time.time() - active_stuck_start >= active_stuck_timeout:
            if verbose:
                print(
                    f"\nMigration did not start: VM {vm_name} stayed ACTIVE on {src_host} "
                    f"for {active_stuck_timeout} seconds."
                )
            return active_stuck_start, MIGRATION_RESULT_STUCK_ACTIVE
        return active_stuck_start, None

    return None, None


def lookup_global_migration_result(conn, migration_id, verbose=True):
    """Check the global migrations list for a tracked migration result."""
    if not migration_id:
        return False, None

    for migration in conn.compute.migrations():
        current_id = get_migration_identifier(migration)
        if str(current_id) != str(migration_id):
            continue

        status = getattr(migration, "status", None)
        normalized_status = status.lower() if status else ""
        if verbose:
            print(f"\nMigration {migration_id} found in global migrations with status: {status}")

        if normalized_status == "running (post-copy)":
            if verbose:
                print(f"Migration {migration_id} is running in post-copy (global record).")
            return True, None
        if normalized_status in SUCCESS_STATUSES:
            if verbose:
                print(f"Migration {migration_id} completed successfully (global record).")
            return True, MIGRATION_RESULT_SUCCESS
        if normalized_status in FAILED_STATUSES:
            if verbose:
                print(f"Migration {migration_id} failed (global record).")
            return True, MIGRATION_RESULT_FAILED
        if verbose:
            print(f"Migration {migration_id} is in state '{status}' (global record).")
        return True, None

    return False, None


def monitor_migration(conn, server_id, vm_name, src_host=None, dst_host=None, timeout=None, verbose=True):
    """Poll OpenStack migration state until the existing success/failure rules are satisfied."""
    if timeout is None:
        timeout = MIGRATION_TIMEOUT

    active_stuck_timeout = 120
    migration = None
    orig_host = src_host
    dest_host = dst_host

    start_time = time.time()
    last_state = None
    active_stuck_start = None
    migration_id = None

    while True:
        try:
            migrations = list(conn.compute.server_migrations(server_id))

            if not migration and migrations:
                migration = migrations[-1]
                migration_id = get_migration_identifier(migration)
                orig_host = getattr(migration, 'source_node', None) or src_host
                dest_host = getattr(migration, 'dest_node', None) or dst_host
                if verbose:
                    print(f"Migration ID: {migration_id}")
                    print(f"Status: {getattr(migration, 'status', 'n/a')}")
                    print(f"Source Server: {orig_host}")
                    print(f"Destination Server: {dest_host}\n")

            mig = find_tracked_migration(migrations, migration_id)

            server = conn.compute.get_server(server_id)
            current_host = get_server_current_host(server)
            server_status = get_server_status(server)
            server_result = get_server_migration_result(
                vm_name,
                current_host,
                server_status,
                src_host=src_host,
                dst_host=dst_host,
                verbose=verbose,
            )
            if server_result:
                return server_result

            active_stuck_start, stuck_result = update_active_stuck_state(
                vm_name,
                current_host,
                server_status,
                src_host,
                active_stuck_start,
                active_stuck_timeout,
                verbose=verbose,
            )
            if stuck_result:
                return stuck_result

            if not mig:
                global_found, global_result = lookup_global_migration_result(
                    conn,
                    migration_id,
                    verbose=verbose,
                )
                if global_result:
                    return global_result
                if not global_found:
                    if migration_id:
                        if verbose:
                            print(
                                f"\nMigration {migration_id} vanished before completion. "
                                f"VM is on {current_host} with status {server_status}. Please check logs."
                            )
                        return MIGRATION_RESULT_FAILED

                if (time.time() - start_time) > timeout:
                    if verbose:
                        print(f"\nMigration monitoring timed out after {timeout} seconds.")
                    return MIGRATION_RESULT_TIMEOUT

                time.sleep(POLL_INTERVAL)
                continue

            percent = get_migration_progress(server, mig)
            progress_str = f"{percent}%" if percent is not None else "?"
            status = get_migration_status(mig)
            state = map_migration_state(status)

            if state != last_state and verbose:
                print()

            if verbose:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Migration {migration_id} | VM: {vm_name} | "
                    f"State: {state} | Progress: {progress_str}   ",
                    end="\r",
                    flush=True
                )
            last_state = state

            if state == "SUCCESS":
                if verbose:
                    print(f"\nMigration {migration_id} completed successfully!")
                return MIGRATION_RESULT_SUCCESS
            if state == "FAILED":
                if verbose:
                    print(f"\nMigration {migration_id} failed!")
                return MIGRATION_RESULT_FAILED

            if (time.time() - start_time) > timeout:
                if verbose:
                    print(f"\nMigration {migration_id} monitoring timed out after {timeout} seconds.")
                return MIGRATION_RESULT_TIMEOUT

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            if verbose:
                print(f"\nMonitoring error or migration disappeared (will retry): {e}")
            if (time.time() - start_time) > timeout:
                if verbose:
                    print(f"\nMigration monitoring timed out after {timeout} seconds.")
                return MIGRATION_RESULT_TIMEOUT
            time.sleep(POLL_INTERVAL)
# ============================================================================
# RUNTIME PRINTING / APPLY
# ============================================================================

def print_baseline_table(state):
    hosts_in_agg = state["hosts_in_agg"]
    host_cpu_pct = state["host_cpu_pct"]
    host_ram_pct = state["host_ram_pct"]
    host_cores = state["host_cores"]
    host_ram = state["host_ram"]
    host_free_cores = state["host_free_cores"]
    host_free_ram = state["host_free_ram"]

    print("Baseline host CPU and RAM usage before migration:")
    print(f"{'Host':<18} {'CPU %':>7} {'RAM %':>8} {'Cores':>7} {'RAM (GB)':>10} {'Avail Cores':>13} {'Avail RAM (GB)':>15}")
    print("-" * 80)
    for h in hosts_in_agg:
        print(
            f"{h:<18} "
            f"{host_cpu_pct[h]:7.2f} "
            f"{host_ram_pct[h]:8.2f} "
            f"{host_cores[h]:7d} "
            f"{host_ram[h]:10.2f} "
            f"{host_free_cores[h]:13.2f} "
            f"{host_free_ram[h]:15.2f}"
        )
    print()

def format_reason(reason, state=None):
    if reason == "no_technical_candidate":
        return "no feasible VM can be moved"

    if reason == "no_hotspot_candidate_above_improvement_threshold":
        if state:
            return (
                "no move reduces hotspot enough because best improvement "
                f"({state['best_improvement']:.3f}) is below threshold "
                f"({state['improvement_threshold']:.3f})"
            )
        return "no move reduces hotspot enough"

    if reason == "no_pressure_candidate_above_improvement_threshold":
        return "no move reduces pressure enough because no candidate meets required improvement"

    if reason == "no_candidate_above_improvement_threshold":
        return "no migration provides enough improvement"

    return reason

def get_hot_compute(state):
    mode = state["mode"]
    host_cpu_pct = state.get("host_cpu_pct", {})
    host_ram_pct = state.get("host_ram_pct", {})

    if mode in ("ram_hotspot", "ram_critical_hotspot") and host_ram_pct:
        return max(host_ram_pct, key=host_ram_pct.get)

    if mode == "cpu_hotspot" and host_cpu_pct:
        return max(host_cpu_pct, key=host_cpu_pct.get)

    return None

def format_candidate_line(state):
    best_move = state.get("best_move")
    if not best_move:
        return "Candidate: none"

    vm_name = state["all_vm_names"].get(best_move.vm_id, best_move.vm_id)

    if state["has_valid_move"]:
        return f"Proposed move: {vm_name} ({best_move.src_host} → {best_move.dst_host})"

    return f"Best candidate: {vm_name} ({best_move.src_host} → {best_move.dst_host}) [below threshold]"

def build_headline(state):
    severity = state["severity"]
    status = state["detail_status"]
    best = state["best_improvement"]
    threshold = state["improvement_threshold"]
    max_cpu = state["max_cpu"]
    max_ram = state["max_ram"]
    
    hot_compute = get_hot_compute(state)
    hot_str = f" | hot_compute={hot_compute}" if hot_compute else ""

    if status == "RAM_HOTSPOT_NO_VALID_MOVE":
        return (
            f"{severity} - RAM hotspot {max_ram:.2f}% "
            f"but no effective migration (best={best:.3f} < {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "RAM_HOTSPOT_OPTIMIZABLE":
        return (
            f"{severity} - RAM hotspot {max_ram:.2f}% "
            f"and can be improved automatically (best={best:.3f} >= {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "CPU_HOTSPOT_NO_VALID_MOVE":
        return (
            f"{severity} - CPU hotspot {max_cpu:.2f}% "
            f"but no effective migration (best={best:.3f} < {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "CPU_HOTSPOT_OPTIMIZABLE":
        return (
            f"{severity} - CPU hotspot {max_cpu:.2f}% "
            f"and can be improved automatically (best={best:.3f} >= {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "PRESSURE_NO_VALID_MOVE":
        return (
            f"{severity} - resource pressure but no effective migration "
            f"(best={best:.3f} < {threshold:.3f})"
            f"{hot_str}"
        )

    if status in ("RAM_PRESSURE", "CPU_PRESSURE"):
        return (
            f"{severity} - resource pressure and can be improved automatically "
            f"(best={best:.3f} >= {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "UNBALANCED_NO_VALID_MOVE":
        return (
            f"{severity} - cluster unbalanced but no effective migration "
            f"(best={best:.3f} < {threshold:.3f})"
        )

    if status == "UNBALANCED":
        return (
            f"{severity} - cluster unbalanced and can be improved automatically "
            f"(best={best:.3f} >= {threshold:.3f})"
        )

    if status == "BALANCED_OPTIMIZABLE":
        return (
            f"{severity} - cluster balanced with small optimization available "
            f"(best={best:.3f} >= {threshold:.3f})"
        )

    return f"{severity} - cluster balanced"

def build_detail_lines(state):    

    lines = [
        f"Mode: {format_mode_label(state['mode'])}",
        f"Peak: CPU={state['max_cpu']:.2f}% | RAM={state['max_ram']:.2f}%",
        "",
        f"Cluster: CPU avg={state['cpu_mean']:.2f}% | RAM avg={state['ram_mean']:.2f}%",
        (
            f"Spread : CPU MAD={state['baseline_cpu_mad']:.2f}% | "
            f"RAM MAD={state['baseline_ram_mad']:.2f}% | "
            f"Weighted MAD={state['baseline_weighted_mad']:.2f}"
        ),
        f"Priority: CPU {state['cpu_weight']*100:.0f}% | RAM {state['ram_weight']*100:.0f}%",
        "",
        f"Status: {'can be improved automatically' if state['has_valid_move'] else 'cannot be fixed automatically'}",
        format_candidate_line(state),
    ]

    if state.get("reason"):
        lines.append(f"Reason: {format_reason(state['reason'], state)}")

    return lines

def build_monitor_message(state):
    lines = [build_headline(state), ""]
    lines.extend(build_detail_lines(state))
    return "\n".join(lines)


def print_state_summary(state):
    print(build_monitor_message(state))
    print()

def print_move_details(state, migration_round):
    best_move = state["best_move"]
    if not best_move or not state["has_valid_move"]:
        return

    vm_name = state["all_vm_names"].get(best_move.vm_id, best_move.vm_id)
    mode_label = format_mode_label(state["mode"])
    new_cpu_mean = sum(best_move.new_host_cpu.values()) / len(best_move.new_host_cpu)
    new_ram_mean = sum(best_move.new_host_ram.values()) / len(best_move.new_host_ram)

    print(
        f"[Round {migration_round}] "
        f"[{format_cluster_status(state['detail_status'])}] "
        f"Migrate {vm_name} from {best_move.src_host} to {best_move.dst_host}"
    )
    print(
        f"   {vm_name} ({best_move.src_host} → {best_move.dst_host}): "
        f"Need {best_move.real_vm_cores:.2f} real cores(flavor {best_move.vcpus}), "
        f"{best_move.real_vm_ram:.2f} real GB(flavor {best_move.vm_ram_gb:.2f} GB) — "
        f"Available {best_move.host_cpu_avail:.2f} real cores, {best_move.host_ram_avail:.2f} GB."
    )
    print(f"   CPU impact: {best_move.cpu_impact:.2f}%  | RAM impact: {best_move.ram_impact:.2f}%")
    print(f"   Search scope: {state['search_scope']}")
    print(f"   New host CPU: {[f'{v:.2f}' for v in best_move.new_host_cpu.values()]}")
    print(f"   New host RAM: {[f'{v:.2f}' for v in best_move.new_host_ram.values()]}")
    print(f"   Means before scoring: CPU={state['cpu_mean']:.2f}%  RAM={state['ram_mean']:.2f}%")
    print(
        f"   Decision mode: {mode_label} | "
        f"Max CPU={state['max_cpu']:.2f}% Max RAM={state['max_ram']:.2f}% | "
        f"CPU signal={state['cpu_signal']:.2f}  RAM signal={state['ram_signal']:.2f}  "
        f"Weights: CPU={state['cpu_weight']:.2f}  RAM={state['ram_weight']:.2f}"
    )
    print(f"   Means after migration: CPU={new_cpu_mean:.2f}%  RAM={new_ram_mean:.2f}%")
    print(
        f"   MADs after migration:  CPU={best_move.mad_cpu:.2f}  RAM={best_move.mad_ram:.2f}  "
        f"Weighted MAD={best_move.new_weighted_mad:.2f}"
    )
    print(
        f"   Max after migration: CPU={max(best_move.new_host_cpu.values()):.2f}%  "
        f"RAM={max(best_move.new_host_ram.values()):.2f}%"
    )
    print(
        f"   MAD improvement this round: baseline_weighted_mad ({state['baseline_weighted_mad']:.3f}) - "
        f"new_weighted_mad ({best_move.new_weighted_mad:.3f}) = {best_move.mad_improvement:.3f}\n"
    )


def build_migration_event(state, result, duration_seconds=None, started_at=None, ended_at=None, detail=None):
    event = {
        "event_time": format_migration_timestamp(ended_at or datetime.now()),
        "event_ts": time.time(),
        "aggregate": state.get("aggregate"),
        "result": result,
        "detail_status": state.get("detail_status"),
        "severity": state.get("severity"),
        "duration_seconds": duration_seconds,
        "started_at": format_migration_timestamp(started_at),
        "ended_at": format_migration_timestamp(ended_at),
        "best_improvement": state.get("best_improvement"),
        "improvement_threshold": state.get("improvement_threshold"),
    }

    if detail:
        event["detail"] = detail

    best_move = state.get("best_move")
    if best_move:
        vm_detail = state.get("all_vm_details", {}).get(best_move.vm_id, {})
        event.update(
            {
                "vm_id": best_move.vm_id,
                "vm_name": state.get("all_vm_names", {}).get(best_move.vm_id, best_move.vm_id),
                "project_id": vm_detail.get("project_id"),
                "project_name": vm_detail.get("project_name"),
                "src_host": best_move.src_host,
                "dst_host": best_move.dst_host,
            }
        )

    return event


def format_migration_event_line(event):
    return json.dumps(event, ensure_ascii=False, sort_keys=True)


def append_migration_event(event, filename=None):
    if filename is None:
        filename = MIGRATION_EVENTS_FILE

    try:
        directory = os.path.dirname(filename) or "."
        os.makedirs(directory, exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            f.write(format_migration_event_line(event) + "\n")
        logging.debug("Recorded migration event to %s: %s", filename, event.get("result"))
    except Exception as e:
        logging.warning("Failed to record migration event to %s: %s", filename, e)


def auto_balance_aggregate(
    cfg,
    conn,
    agg_name,
    hosts_in_agg,
    dry_run=False,
    enforce_server_groups=True,
    assume_yes=False,
    send_all=False,
    send_error=False,
):
    print(f"\n[Aggregate: {agg_name}] Auto-migrating all VMs to improve balance.\n")

    if enforce_server_groups:
        logging.debug("Server group checks ENABLED (default behavior).")
    else:
        logging.warning("Server group checks DISABLED (--no-server-groups). Affinity/anti-affinity will be ignored!")

    state = evaluate_aggregate_state(
        cfg=cfg,
        conn=conn,
        agg_name=agg_name,
        hosts_in_agg=hosts_in_agg,
        enforce_server_groups=enforce_server_groups,
    )

    if state.get("error"):
        logging.info("Aggregate %s: %s, skipping.", agg_name, state["error"])
        return

    print_baseline_table(state)
    print_state_summary(state)

    if not state["has_valid_move"]:
        return

    print_move_details(state, migration_round=1)

    if dry_run:
        append_migration_event(
            build_migration_event(
                state,
                "dry_run",
                detail="valid move proposed; dry-run mode did not execute migration",
            )
        )
        return

    best_move = state["best_move"]
    vm_id = best_move.vm_id
    src_host = best_move.src_host
    dst_host = best_move.dst_host
    vm_name = state["all_vm_names"].get(vm_id, vm_id)
    vm_detail = state["all_vm_details"].get(vm_id, {})
    project_id = vm_detail.get("project_id")
    project_name = vm_detail.get("project_name")

    dest_ready, dest_reason = is_nova_compute_service_ready(conn, dst_host)
    if not dest_ready:
        logging.warning(
            "Skipping proposed move of VM %s (%s) to %s: destination is not ready: %s",
            vm_name,
            vm_id,
            dst_host,
            dest_reason,
        )
        print(
            f"Skipping proposed move for {vm_name} ({vm_id}): "
            f"destination host {dst_host} is not ready: {dest_reason}"
        )
        append_migration_event(
            build_migration_event(
                state,
                "skipped_destination_not_ready",
                detail=dest_reason,
            )
        )
        return

    if not assume_yes:
        answer = input(
            f"Do you want to migrate {vm_name} ({vm_id}) from {src_host} to {dst_host}? ([Y]/n): "
        ).strip().lower()
        if answer not in ("", "y", "yes"):
            print("Migration cancelled.")
            append_migration_event(
                build_migration_event(
                    state,
                    "cancelled",
                    detail="operator declined migration prompt",
                )
            )
            return

    migration_started_at = time.time()
    migration_started_datetime = datetime.now()
    try:
        conn.compute.live_migrate_server(server=vm_id, host=dst_host, block_migration=False)
        migration_result = monitor_migration(conn, vm_id, vm_name, src_host=src_host, dst_host=dst_host)
    except Exception as e:
        migration_ended_datetime = datetime.now()
        migration_duration_seconds = time.time() - migration_started_at
        reset_error_server_to_active(conn, vm_id, vm_name)
        if send_error or send_all:
            send_migration_alert_email(
                cfg,
                agg_name,
                vm_id,
                vm_name,
                src_host,
                dst_host,
                "submit_or_monitor_failed",
                detail=str(e),
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                project_name=project_name,
                project_id=project_id,
            )
        append_migration_event(
            build_migration_event(
                state,
                "submit_or_monitor_failed",
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                detail=str(e),
            )
        )
        now = time.time()
        VM_LAST_FAILED_MOVE_AT[vm_id] = now
        save_cooldown_state()
        logging.warning(
            "Failed to submit or monitor migration for VM %s (%s); "
            "skipping this VM for %d seconds: %s",
            vm_name,
            vm_id,
            VM_FAILED_MOVE_COOLDOWN_SECONDS,
            e,
        )
        return

    migration_ended_datetime = datetime.now()
    migration_duration_seconds = time.time() - migration_started_at
    now = time.time()
    if migration_result == MIGRATION_RESULT_SUCCESS:
        VM_LAST_MOVED_AT[vm_id] = now
        VM_LAST_FAILED_MOVE_AT.pop(vm_id, None)
        save_cooldown_state()
        append_migration_event(
            build_migration_event(
                state,
                "migration_success",
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                detail=f"monitor_migration returned {migration_result}",
            )
        )
        if send_all:
            send_migration_alert_email(
                cfg,
                agg_name,
                vm_id,
                vm_name,
                src_host,
                dst_host,
                "migration_success",
                detail=f"monitor_migration returned {migration_result}",
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                project_name=project_name,
                project_id=project_id,
            )
        logging.info(
            "Recorded cooldown for VM %s (%s) for %d seconds (result=%s).",
            vm_name,
            vm_id,
            VM_COOLDOWN_SECONDS,
            migration_result,
        )
    else:
        reset_error_server_to_active(conn, vm_id, vm_name)
        if migration_result == MIGRATION_RESULT_STUCK_ACTIVE:
            alert_result = "stuck_active"
        elif migration_result == MIGRATION_RESULT_TIMEOUT:
            alert_result = "migration_timeout"
        else:
            alert_result = "migration_failed"
        if send_error or send_all:
            send_migration_alert_email(
                cfg,
                agg_name,
                vm_id,
                vm_name,
                src_host,
                dst_host,
                alert_result,
                detail=f"monitor_migration returned {migration_result}",
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                project_name=project_name,
                project_id=project_id,
            )
        append_migration_event(
            build_migration_event(
                state,
                alert_result,
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                detail=f"monitor_migration returned {migration_result}",
            )
        )
        VM_LAST_FAILED_MOVE_AT[vm_id] = now
        save_cooldown_state()
        logging.warning(
            "Migration failed for VM %s (%s); skipping this VM for %d seconds.",
            vm_name,
            vm_id,
            VM_FAILED_MOVE_COOLDOWN_SECONDS,
        )

# ============================================================================
# MONITOR MODE
# ============================================================================

def monitor_aggregate(cfg, conn, agg_name, hosts_in_agg, enforce_server_groups=True):
    state = evaluate_aggregate_state(
        cfg=cfg,
        conn=conn,
        agg_name=agg_name,
        hosts_in_agg=hosts_in_agg,
        enforce_server_groups=enforce_server_groups,
    )

    if state.get("error"):
        msg = f"UNKNOWN - aggregate={agg_name}, detail=UNKNOWN, reason={state['error']}"
        return {
            "aggregate": agg_name,
            "detail_status": "UNKNOWN",
            "severity": "UNKNOWN",
            "exit_code": ICINGA_UNKNOWN,
            "message": msg,
        }

    return {
        "aggregate": agg_name,
        "detail_status": state["detail_status"],
        "severity": state["severity"],
        "exit_code": state["exit_code"],
        "message": state["message"],
    }


# ============================================================================
# AGGREGATE DRIVERS
# ============================================================================

def get_aggregate_map(conn, aggregate_names=None):
    aggregate_map = {}
    for agg in conn.compute.aggregates():
        agg_name = getattr(agg, 'name', None) or getattr(agg, 'id', str(id(agg)))
        if aggregate_names and agg_name not in aggregate_names:
            continue
        aggregate_map[agg_name] = list(agg.hosts)
    return aggregate_map


def balance_by_aggregate(
    cfg,
    dry_run=False,
    aggregate_names=None,
    enforce_server_groups=True,
    assume_yes=False,
    send_all=False,
    send_error=False,
):
    conn = get_openstack_connection()
    aggregate_map = get_aggregate_map(conn, aggregate_names=aggregate_names)

    for agg_name, hosts in aggregate_map.items():
        if len(hosts) < 2:
            logging.info("Aggregate %s: less than 2 hosts, skipping.", agg_name)
            continue

        auto_balance_aggregate(
            cfg=cfg,
            conn=conn,
            agg_name=agg_name,
            hosts_in_agg=hosts,
            dry_run=dry_run,
            enforce_server_groups=enforce_server_groups,
            assume_yes=assume_yes,
            send_all=send_all,
            send_error=send_error,
        )


def monitor_by_aggregate(cfg, aggregate_names=None, enforce_server_groups=True):
    conn = get_openstack_connection()
    aggregate_map = get_aggregate_map(conn, aggregate_names=aggregate_names)

    if not aggregate_map:
        print("UNKNOWN - no aggregates matched selection")
        return ICINGA_UNKNOWN

    results = []
    highest_exit_code = ICINGA_OK

    for agg_name, hosts in aggregate_map.items():
        if len(hosts) < 2:
            msg = f"UNKNOWN - aggregate={agg_name}, detail=UNKNOWN, reason=less_than_2_hosts"
            results.append((ICINGA_UNKNOWN, msg))
            highest_exit_code = max(highest_exit_code, ICINGA_UNKNOWN)
            continue

        result = monitor_aggregate(
            cfg=cfg,
            conn=conn,
            agg_name=agg_name,
            hosts_in_agg=hosts,
            enforce_server_groups=enforce_server_groups,
        )
        results.append((result["exit_code"], result["message"]))
        highest_exit_code = max(highest_exit_code, result["exit_code"])

    for _, message in results:
        print(message)

    return highest_exit_code


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option('--config', default="/etc/loadleveller-secrets.conf", help="Path to config file")
@click.option('--verbose', is_flag=True, help="Enable verbose logging")
@click.option('--dry-run', is_flag=True, help="Only show what would be migrated, do not migrate")
@click.option('--monitor-only', is_flag=True, help="Only evaluate cluster balance/hotspot state and return Icinga-compatible exit code")
@click.option('--aggregate', '-a', multiple=True, help="Only operate on these aggregate(s)")
@click.option(
    '--no-server-groups',
    is_flag=True,
    help="Disable server group (affinity/anti-affinity) checks. WARNING: may violate policies."
)
@click.option('-y', '--yes', 'assume_yes', is_flag=True, help="Skip y/n prompt and auto-approve migrations")
@click.option('--send-all', is_flag=True, help="Send migration email alerts for success and errors")
@click.option(
    '--send-error',
    is_flag=True,
    help="Send migration email alerts for failed, stuck, timeout, or submit/monitor errors",
)
@click.option(
    '--cooldown-file',
    default=COOLDOWN_STATE_FILE,
    show_default=True,
    help="Path to JSON file used to persist VM migration cooldown state",
)
@click.option(
    '--events-file',
    default=MIGRATION_EVENTS_FILE,
    show_default=True,
    help="Path to JSONL file used to append valid migration move events",
)
def main(
    config,
    verbose,
    dry_run,
    monitor_only,
    aggregate,
    no_server_groups,
    assume_yes,
    send_all,
    send_error,
    cooldown_file,
    events_file,
):
    if monitor_only:
        # Disable INFO logs in monitor mode (keep WARNING/ERROR if any)
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s %(levelname)s: %(message)s'
        )
    else:
        setup_logging(verbose)
    enforce_server_groups = not no_server_groups

    if monitor_only and dry_run:
        logging.warning("--dry-run is ignored when --monitor-only is used.")
    if monitor_only and assume_yes:
        logging.warning("--yes is ignored when --monitor-only is used.")
    if monitor_only and send_all:
        logging.warning("--send-all is ignored when --monitor-only is used.")
    if monitor_only and send_error:
        logging.warning("--send-error is ignored when --monitor-only is used.")

    try:
        global COOLDOWN_STATE_FILE, MIGRATION_EVENTS_FILE
        COOLDOWN_STATE_FILE = cooldown_file
        MIGRATION_EVENTS_FILE = events_file
        load_cooldown_state(COOLDOWN_STATE_FILE)

        cfg = LoadbalancerConfig.load_config(config)
        aggregate_names = list(aggregate) if aggregate else None

        if monitor_only:
            exit_code = monitor_by_aggregate(
                cfg,
                aggregate_names=aggregate_names,
                enforce_server_groups=enforce_server_groups,
            )            
            sys.exit(exit_code)

        balance_by_aggregate(
            cfg,
            dry_run=dry_run,
            aggregate_names=aggregate_names,
            enforce_server_groups=enforce_server_groups,
            assume_yes=assume_yes,
            send_all=send_all,
            send_error=send_error,
        )

    except PrometheusError as e:
        logging.error("Prometheus query failed: %s", e)
        sys.exit(ICINGA_UNKNOWN if monitor_only else 1)
    except ConfigError as e:
        logging.error("Configuration error: %s", e)
        sys.exit(ICINGA_UNKNOWN if monitor_only else 1)
    except KeyError as e:
        logging.error("Missing required environment variable: %s", e)
        sys.exit(ICINGA_UNKNOWN if monitor_only else 1)
    except Exception as e:
        logging.exception("Unexpected error: %s", e)
        sys.exit(ICINGA_UNKNOWN if monitor_only else 1)


if __name__ == "__main__":
    main()
