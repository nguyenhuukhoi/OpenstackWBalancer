"""execution.monitor — extracted domain responsibilities."""
import compute_maintenance.adapters.openstack as _sources_openstack
import compute_maintenance.application.snapshots as _snapshot
import compute_maintenance.config as _config
import compute_maintenance.domain.errors as _errors
import compute_maintenance.domain.plan_context as _plan_context
import compute_maintenance.domain.severity as _severity

# rc11 names; the combination rule lives in domain.severity (single source).
EXIT_CODE_SEVERITY = _severity.EXIT_CODE_SEVERITY
worst_exit_code = _severity.worst_exit_code


def unknown_line(agg_name, reason):
    """Monitor line of an aggregate that could not be evaluated."""
    return (f"{_severity.UNKNOWN} - aggregate={agg_name}, "
            f"detail={_severity.UNKNOWN_DETAIL_STATUS}, reason={reason}")


def monitor_aggregate(
    cfg,
    conn,
    agg_name,
    hosts_in_agg,
    enforce_server_groups=True,
    cluster_cache=None,
    exclude_vm_patterns=None,
):
    try:
        state = _snapshot.evaluate_aggregate_state(
            cfg=cfg,
            conn=conn,
            agg_name=agg_name,
            hosts_in_agg=hosts_in_agg,
            enforce_server_groups=enforce_server_groups,
            cluster_cache=cluster_cache,
            exclude_vm_patterns=exclude_vm_patterns,
            exclude_unready_hosts=True,
        )
    except (_errors.OpenStackDataError, _errors.PrometheusError, _errors.PlacementError) as e:
        # One aggregate's data problem is UNKNOWN for that aggregate only.
        state = _plan_context.PlanContext(aggregate=agg_name, error=str(e))

    if state.error:
        return {
            "aggregate": agg_name,
            "detail_status": _severity.UNKNOWN_DETAIL_STATUS,
            "severity": _severity.UNKNOWN,
            "exit_code": _severity.map_severity_to_exit_code(_severity.UNKNOWN),
            "message": unknown_line(agg_name, state.error),
        }

    return {
        "aggregate": agg_name,
        "detail_status": state.detail_status,
        "severity": state.severity,
        "exit_code": state.exit_code,
        "message": state.message,
    }


def monitor_by_aggregate(
    cfg,
    aggregate_names=None,
    enforce_server_groups=True,
    exclude_vm_patterns=None,
):
    conn = _sources_openstack.get_openstack_connection(getattr(cfg, "os_cloud", _config.DEFAULT_OS_CLOUD))
    aggregate_map = _sources_openstack.get_aggregate_map(
        conn,
        aggregate_names=aggregate_names,
        show_available_on_missing=False,
    )

    if not aggregate_map:
        print(f"{_severity.UNKNOWN} - no aggregates matched selection")
        return _severity.map_severity_to_exit_code(_severity.UNKNOWN)

    # Fetch cluster-wide data once and reuse it for every aggregate.
    all_hosts = _sources_openstack.build_selected_host_list(aggregate_map)
    cluster_cache = _snapshot.build_cluster_cache(
        conn, cfg, all_hosts, enforce_server_groups=enforce_server_groups
    )

    results = []

    for agg_name, hosts in aggregate_map.items():
        if len(hosts) < 2:
            results.append((_severity.map_severity_to_exit_code(_severity.UNKNOWN),
                            unknown_line(agg_name, "less_than_2_hosts")))
            continue

        result = monitor_aggregate(
            cfg=cfg,
            conn=conn,
            agg_name=agg_name,
            hosts_in_agg=hosts,
            enforce_server_groups=enforce_server_groups,
            cluster_cache=cluster_cache,
            exclude_vm_patterns=exclude_vm_patterns,
        )
        results.append((result["exit_code"], result["message"]))

    # Every aggregate keeps its own line; only the exit code is combined.
    for _, message in results:
        print(message)

    return _severity.worst_exit_code(code for code, _ in results)
