"""sources.prometheus — extracted domain responsibilities."""
import logging
import math
import re
from collections import defaultdict

import requests

import compute_maintenance.config as _config
import compute_maintenance.domain.errors as _errors
import compute_maintenance.domain.server as _execution_migrate


def redact_url(text):
    """Remove ``user:password@`` from any URL inside ``text``."""
    return re.sub(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@", r"\1***@", str(text))


def do_query(query_url: str, query: str, at=None):
    """Execute a Prometheus instant query (optionally at time ``at``)."""
    try:
        params = {'query': query}
        if at is not None:
            params['time'] = repr(float(at))
        response = requests.get(query_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('status') != 'success':
            raise _errors.PrometheusError(f"Prometheus query error: {data.get('error', 'Unknown error')}")
        return data
    except ValueError as e:
        raise _errors.PrometheusError(f"Prometheus returned invalid JSON: {redact_url(e)}") from None
    except requests.RequestException as e:
        raise _errors.PrometheusError(f"Prometheus HTTP error: {redact_url(e)}") from None


def get_prometheus_result_rows(data, error_message):
    """Extract the standard Prometheus result list or raise a consistent error."""
    try:
        return data["data"]["result"]
    except (KeyError, TypeError) as e:
        raise _errors.PrometheusError(f"{error_message}: {e}")


def parse_finite_metric_value(raw, label):
    """Parse one Prometheus sample and reject NaN/Inf fail-closed."""
    try:
        value = float(raw)
    except (TypeError, ValueError) as e:
        raise _errors.PrometheusError(f"Prometheus metric {label} is not numeric: {raw!r}") from e
    if not math.isfinite(value):
        raise _errors.PrometheusError(f"Prometheus metric {label} is not finite: {raw!r}")
    return value


def get_metrics(query_url: str, query: str, hosts=None):
    """Return host metrics keyed by host alias from a Prometheus query.

    With ``hosts`` only those aliases are considered: a target without an
    ``alias`` label, or a duplicated alias of some unrelated node in the same
    job, must not fail the snapshot of the selected hosts.
    """
    data = do_query(query_url, query)
    wanted = set(hosts) if hosts is not None else None
    results = {}
    try:
        for r in get_prometheus_result_rows(data, "Prometheus returned malformed results"):
            host = r["metric"].get("alias") if wanted is not None else r["metric"]["alias"]
            if wanted is not None and host not in wanted:
                continue
            if host in results:
                raise _errors.PrometheusError(
                    f"Prometheus returned duplicate host series for alias {host!r}; "
                    "refusing an ambiguous host snapshot"
                )
            value = parse_finite_metric_value(r["value"][1], f"host={host}")
            results[host] = {"host": host, "value": value}
        return results
    except _errors.PrometheusError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise _errors.PrometheusError(f"Prometheus returned malformed results: {e}")


def escape_prometheus_regex_literal(value):
    """Escape RE2 metacharacters without escaping a literal hyphen."""
    return re.sub(r"([\\.^$*+?{}\[\]|()])", r"\\\1", value)


def aggregate_host_regex(hosts_in_agg):
    # Host names are data, not regular-expression fragments.  Anchoring also
    # prevents a host such as ``compute1`` from matching ``compute10``.
    return "^(" + "|".join(
        escape_prometheus_regex_literal(host) for host in hosts_in_agg
    ) + ")$"


def build_vm_metric_queries(hosts_in_agg, rss_query_template=None, libvirt_job=_config.PROMETHEUS_LIBVIRT_JOB):
    """Build the aggregate-scoped Prometheus queries for VM CPU and RAM usage."""
    libvirt_job = _config.require_job(libvirt_job, 'PROMETHEUS_LIBVIRT_JOB')
    host_re = aggregate_host_regex(hosts_in_agg)
    mem_query = (
        f'(100 - (avg_over_time(libvirt_domain_stat_memory_usable_bytes{{job="{libvirt_job}", host=~"{host_re}"}}[5m]) '
        f'/ avg_over_time(libvirt_domain_stat_memory_available_bytes{{job="{libvirt_job}", host=~"{host_re}"}}[5m])) * 100)'
    )
    cpu_query = (
        f'(irate(libvirt_domain_info_cpu_time_seconds_total{{job="{libvirt_job}", host=~"{host_re}"}}[5m]) '
        f'/ libvirt_domain_info_virtual_cpus{{job="{libvirt_job}", host=~"{host_re}"}}) * 100'
    )
    rss_query = None
    if rss_query_template:
        rss_query = rss_query_template.replace("__HOST_RE__", host_re)
    return mem_query, cpu_query, rss_query


def group_vm_metrics_by_host(vm_metrics):
    """Group VM metrics by source host to avoid repeated full scans."""
    metrics_by_host = defaultdict(dict)
    for vm_id, metric in vm_metrics.items():
        host = metric.get("host")
        if host is not None:
            metrics_by_host[host][vm_id] = metric
    return metrics_by_host


def get_all_vm_metrics(cfg, hosts_in_agg, server_cache=None, ambiguous=None):
    """Fetch per-VM CPU and RAM usage for the selected aggregate hosts.

    VMs exposed on two hosts (right after a live migration) are never
    candidates. With an ``ambiguous`` dict their largest observed sample is
    stored there, so the load of a VM that just arrived can still be
    estimated from real usage instead of its full flavor.
    """
    mem_query, cpu_query, rss_query = build_vm_metric_queries(
        hosts_in_agg,
        rss_query_template=cfg.prometheus_query_vm_ram_rss,
        libvirt_job=_config.cfg_libvirt_job(cfg),
    )

    mem_data = do_query(cfg.prometheus_query_url, mem_query)
    cpu_data = do_query(cfg.prometheus_query_url, cpu_query)
    rss_data = do_query(cfg.prometheus_query_url, rss_query) if rss_query else None

    # Build a snapshot per (VM, host) first.  During live migration exporters
    # can briefly expose one VM on both source and destination.  Merging those
    # rows by VM id would create a synthetic CPU/RAM sample that never existed
    # on either host, so ambiguous VMs are deliberately excluded as candidates.
    samples = defaultdict(dict)
    hosts_seen_by_vm = defaultdict(set)

    def add_sample(entry, metric_name, divisor=1.0):
        vm_id = entry["metric"].get("instanceId")
        host = entry["metric"].get("host")
        if not vm_id or not host:
            return
        key = (vm_id, host)
        value = parse_finite_metric_value(
            entry["value"][1], f"vm={vm_id},host={host},metric={metric_name}"
        ) / divisor
        if metric_name in samples[key]:
            # Same VM and host under two label sets (rename, resize or exporter
            # restart keep the old series inside the range window). Keep the
            # larger value: over-estimating load is the safe side.
            logging.debug(
                "Duplicate %s series for VM %s on %s; keeping the larger value.",
                metric_name, vm_id, host,
            )
            value = max(value, samples[key][metric_name])
        samples[key][metric_name] = value
        hosts_seen_by_vm[vm_id].add(host)

    try:
        for entry in get_prometheus_result_rows(
            mem_data,
            "Prometheus returned malformed VM memory results",
        ):
            add_sample(entry, "ram_pct")

        for entry in get_prometheus_result_rows(
            cpu_data,
            "Prometheus returned malformed VM CPU results",
        ):
            add_sample(entry, "cpu_pct")

        if rss_data is not None:
            for entry in get_prometheus_result_rows(
                rss_data,
                "Prometheus returned malformed VM memory RSS results",
            ):
                add_sample(entry, "ram_rss_gb", divisor=(1024.0 ** 3))
    except _errors.PrometheusError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise _errors.PrometheusError(f"Prometheus returned malformed VM metric results: {e}")

    # RAM% comes from the guest balloon driver (usable/available). Guests whose
    # driver reports no memory stats (FreeBSD: OPNsense/pfSense, Windows without
    # the virtio balloon service) have CPU metrics but no RAM%, and used to be
    # skipped -- so a drain left them RUNNING on a host about to reboot. For
    # those VMs only, use the host-side RSS instead (always exported by libvirt,
    # >= what the guest really uses, so the estimate errs on the safe side).
    if rss_data is None:
        fill_vm_rss_fallback(cfg, hosts_in_agg, samples)

    vm_metrics = {}
    for vm_id, seen_hosts in hosts_seen_by_vm.items():
        if len(seen_hosts) != 1:
            logging.warning(
                "VM %s has Prometheus series on multiple hosts (%s); skipping it "
                "as a migration candidate until the exporter snapshot converges.",
                vm_id,
                ", ".join(sorted(seen_hosts)),
            )
            if ambiguous is not None:
                merged = {}
                for host in seen_hosts:
                    for key, value in samples[(vm_id, host)].items():
                        if value is None:
                            continue
                        if key == "cpu_pct":
                            value = min(max(value, 0.0), 100.0)
                        merged[key] = max(merged.get(key, value), value)
                ambiguous[vm_id] = merged
            continue

        host = next(iter(seen_hosts))
        sample = samples[(vm_id, host)]
        server = (server_cache or {}).get(vm_id)
        nova_status = _execution_migrate.get_server_status(server) if server is not None else None
        if nova_status and nova_status != _config.ACTIVE_VM_STATUS:
            # Powered-off/transitional guests are deliberately not migration
            # candidates and commonly have no libvirt RAM/RSS series. Classify
            # them from Nova later instead of emitting a misleading metric-gap
            # warning or waiting for metrics that cannot return while stopped.
            logging.debug(
                "VM %s on %s is %s; runtime metrics are not required for a "
                "non-ACTIVE VM.",
                vm_id,
                host,
                nova_status,
            )
            continue
        ram_from_rss = (
            "cpu_pct" in sample
            and "ram_pct" not in sample
            and sample.get("ram_rss_gb") is not None
        )
        if ram_from_rss:
            logging.warning(
                "VM %s on %s has no guest memory stats (balloon); using host RSS "
                "%.2f GiB as its RAM usage.",
                vm_id,
                host,
                sample["ram_rss_gb"],
            )
            # Placeholder in range; calculate_vm_scores() derives the real
            # percentage from RSS and the flavor RAM.
            sample["ram_pct"] = 0.0
        if "cpu_pct" not in sample or "ram_pct" not in sample:
            logging.warning(
                "VM %s on %s is missing %s metric; skipping it as a migration candidate.",
                vm_id,
                host,
                "CPU" if "cpu_pct" not in sample else "RAM",
            )
            continue

        if not (
            -_config.PERCENT_METRIC_TOLERANCE
            <= sample["cpu_pct"]
            <= _config.VM_CPU_PCT_SANITY_MAX
        ) or not (
            -_config.PERCENT_METRIC_TOLERANCE
            <= sample["ram_pct"]
            <= 100.0 + _config.PERCENT_METRIC_TOLERANCE
        ):
            logging.warning(
                "VM %s on %s has out-of-range CPU/RAM metrics (%s/%s); skipping it.",
                vm_id,
                host,
                sample["cpu_pct"],
                sample["ram_pct"],
            )
            continue
        sample["cpu_pct"] = min(max(sample["cpu_pct"], 0.0), 100.0)
        sample["ram_pct"] = min(max(sample["ram_pct"], 0.0), 100.0)
        if sample.get("ram_rss_gb") is not None and sample["ram_rss_gb"] < 0:
            logging.warning(
                "VM %s on %s has negative RSS; skipping it.", vm_id, host
            )
            continue

        vm_metrics[vm_id] = {
            "host": host,
            "cpu_pct": sample["cpu_pct"],
            "ram_pct": None if ram_from_rss else sample["ram_pct"],
            "ram_rss_gb": sample.get("ram_rss_gb"),
            "metrics_complete": True,
        }

    logging.info(
        "Fetched aggregate VM metrics for %d VMs across %d hosts.",
        len(vm_metrics),
        len(hosts_in_agg),
    )

    return vm_metrics


def fill_vm_rss_fallback(cfg, hosts_in_agg, samples):
    """Add ``ram_rss_gb`` to samples that have CPU but no RAM%. Best effort:
    a failed query leaves them skipped, exactly as before."""
    need = {key for key, s in samples.items() if "cpu_pct" in s and "ram_pct" not in s}
    if not need:
        return
    query = _config.cfg_vm_ram_rss_query(cfg).replace(
        "__HOST_RE__", aggregate_host_regex(hosts_in_agg)
    )
    try:
        rows = get_prometheus_result_rows(
            do_query(cfg.prometheus_query_url, query),
            "Prometheus returned malformed VM memory RSS results",
        )
        parsed = {}
        seen = set()
        for entry in rows:
            key = (entry["metric"].get("instanceId"), entry["metric"].get("host"))
            if key not in need:
                continue
            if key in seen:
                raise _errors.PrometheusError(f"duplicate RSS series for {key}")
            seen.add(key)
            value = parse_finite_metric_value(
                entry["value"][1], f"vm={key[0]},host={key[1]},metric=ram_rss_gb"
            )
            if value > 0:
                parsed[key] = value / (1024.0 ** 3)
            else:
                raise _errors.PrometheusError(f"non-positive RSS sample for {key}")
        # No partial last-wins updates if any sample was malformed/ambiguous.
        for key, value in parsed.items():
            samples[key]["ram_rss_gb"] = value
    except (_errors.PrometheusError, KeyError, IndexError, TypeError) as e:
        logging.warning(
            "VM RSS fallback query failed (%s); %d VM(s) without guest memory "
            "stats stay skipped.", e, len(need)
        )


def get_vm_rss_gb(cfg, vm_ids):
    """Host-side RSS (GiB) of the given VMs: what host MemAvailable really loses."""
    ids = sorted({str(v) for v in vm_ids if v})
    if not ids:
        return {}
    job = _config.cfg_libvirt_job(cfg)
    selector = "|".join(escape_prometheus_regex_literal(v) for v in ids)
    query = f'max by (instanceId) (libvirt_domain_stat_memory_rss_bytes{{job="{job}", instanceId=~"^({selector})$"}})'
    result = {}
    for row in get_prometheus_result_rows(do_query(cfg.prometheus_query_url, query), "VM RSS query failed"):
        vm_id = row["metric"].get("instanceId")
        value = parse_finite_metric_value(row["value"][1], "rss of " + str(vm_id))
        if vm_id and value > 0:
            result[vm_id] = value / (1024.0 ** 3)
    return result


def get_vm_cpu_pct_at(cfg, vm_id, at):
    """A VM's CPU percentage (of its vCPUs) at time ``at``, or None.

    Used for a VM that just arrived: its new series has no irate() yet and
    the old one is inflated by QEMU's migration thread, so the sample taken
    just before the migration started is the best estimate of its load.
    """
    job = _config.cfg_libvirt_job(cfg)
    sel = f'{{job="{job}", instanceId="{escape_prometheus_regex_literal(str(vm_id))}"}}'
    query = (f"max(irate(libvirt_domain_info_cpu_time_seconds_total{sel}[5m]) / "
             f"libvirt_domain_info_virtual_cpus{sel}) * 100")
    rows = get_prometheus_result_rows(do_query(cfg.prometheus_query_url, query, at=at), "VM CPU query failed")
    if not rows:
        return None
    value = parse_finite_metric_value(rows[0]["value"][1], "cpu of " + str(vm_id))
    return min(max(value, 0.0), 100.0) if value <= _config.VM_CPU_PCT_SANITY_MAX else None


_SCRAPE_INTERVAL_CACHE = {}


def get_scrape_interval_seconds(query_url, job):
    """Largest scrape interval of the active targets of ``job`` (seconds).

    Read from /api/v1/targets next to the query endpoint. Returns None when
    it cannot be determined; callers then fall back to a configured default.
    Cached per (url, job) for the lifetime of the process.
    """
    key = (query_url, job)
    if key in _SCRAPE_INTERVAL_CACHE:
        return _SCRAPE_INTERVAL_CACHE[key]
    from compute_maintenance.application.recovery import parse_promql_duration
    base = re.sub(r"/api/v1/query/?$", "", str(query_url or "").rstrip("/"))
    interval = None
    try:
        response = requests.get(base + "/api/v1/targets", params={"state": "active"}, timeout=10)
        response.raise_for_status()
        targets = response.json()["data"]["activeTargets"]
        values = [parse_promql_duration(str(t.get("scrapeInterval") or ""))
                  for t in targets if (t.get("labels") or {}).get("job") == job]
        values = [v for v in values if v > 0]
        interval = max(values) if values else None
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        logging.warning("Cannot read the scrape interval of job %s: %s", job, redact_url(e))
    _SCRAPE_INTERVAL_CACHE[key] = interval
    return interval


def get_runtime_capacity_map(cfg, hosts, missing=None):
    """Return real host CPU/RAM denominators from node-exporter metrics.

    Without ``missing`` any host lacking capacity raises (strict, the old
    contract). With a dict, such hosts are recorded there with the reason and
    left out of the result, so only the aggregate(s) containing them fail.
    """
    cpu_counts = get_metrics(cfg.prometheus_query_url, cfg.prometheus_query_logical_cpus, hosts)
    ram_totals = get_metrics(cfg.prometheus_query_url, cfg.prometheus_query_total_ram_bytes, hosts)

    result = {}
    for host in hosts:
        logical = (cpu_counts.get(host) or {}).get("value")
        ram_bytes = (ram_totals.get(host) or {}).get("value")

        problem = None
        if logical is None or logical <= 0 or not math.isfinite(float(logical)):
            problem = f"Host {host} is missing a valid positive logical-CPU capacity metric"
        elif ram_bytes is None or ram_bytes <= 0 or not math.isfinite(float(ram_bytes)):
            problem = f"Host {host} is missing a valid positive total-RAM capacity metric"
        if problem:
            if missing is None:
                raise _errors.PrometheusError(problem)
            missing[host] = problem
            continue
        ram_gb = float(ram_bytes) / (1024.0 ** 3)

        result[host] = {
            "logical_cpus": float(logical),
            "ram_gb": float(ram_gb),
        }
    return result
