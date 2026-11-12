"""Inspect raw sample ages; range-query evaluation times are not scrape times."""
import json

from compute_maintenance.adapters.prometheus import (
    do_query,
    get_prometheus_result_rows,
    parse_finite_metric_value,
)
from compute_maintenance.domain.errors import PrometheusError


def validate_runtime_freshness(cfg, hosts, vm_hosts, max_age=120):
    observations = {}
    def ages(metric, job, labels, expected, aggregate_cpu=False, require_all=True):
        query = 'time() - timestamp(' + metric + '{job=' + json.dumps(job) + '})'
        if aggregate_cpu:
            # Reject if ANY currently exposed CPU/mode series is stale. The
            # outer instant evaluation time does not prove raw CPU freshness.
            query = 'max by (alias) (' + query + ')'
        found = {}
        for row in get_prometheus_result_rows(do_query(cfg.prometheus_query_url, query), "freshness query failed"):
            key = tuple(row["metric"].get(label) for label in labels)
            if key not in expected:
                continue
            if key in found:
                raise PrometheusError("duplicate raw freshness series: " + str(key))
            found[key] = parse_finite_metric_value(row["value"][1], "raw sample age")
        # require_all=False: the series is optional per VM (guests without a
        # balloon driver have no usable/available memory), but any series that
        # IS exposed for a planned VM must be fresh.
        invalid = [key for key in expected
                   if (key in found or require_all) and not 0 <= found.get(key, -1.0) <= max_age]
        if invalid:
            raise PrometheusError("raw samples missing/stale for " + metric + ": " + str(sorted(invalid)))
        observations[metric] = max(found.values(), default=0)

    host_keys = {(host,) for host in hosts}
    ages("node_cpu_seconds_total", cfg.prometheus_node_job, ("alias",), host_keys, aggregate_cpu=True)
    for metric in ("node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes"):
        ages(metric, cfg.prometheus_node_job, ("alias",), host_keys)
    # Count/CPU counter has one series per VM and survives guest balloon gaps.
    vm_keys = {(vm, host) for vm, host in vm_hosts.items()}
    ages("libvirt_domain_info_cpu_time_seconds_total", cfg.prometheus_libvirt_job, ("instanceId", "host"), vm_keys)
    # RSS is host-observed and available even for guests without balloon stats.
    ages("libvirt_domain_stat_memory_rss_bytes", cfg.prometheus_libvirt_job, ("instanceId", "host"), vm_keys)
    # The CPU percentage divides by virtual_cpus; RAM percentage reads the
    # balloon usable/available pair. Both feed the planner directly.
    ages("libvirt_domain_info_virtual_cpus", cfg.prometheus_libvirt_job, ("instanceId", "host"), vm_keys)
    for metric in ("libvirt_domain_stat_memory_usable_bytes", "libvirt_domain_stat_memory_available_bytes"):
        ages(metric, cfg.prometheus_libvirt_job, ("instanceId", "host"), vm_keys, require_all=False)
    return observations
