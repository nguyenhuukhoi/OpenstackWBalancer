"""Best-effort diagnostics correlated only with actions created after submission."""
from compute_maintenance.domain.models import get_resource_field


class MigrationMonitorResult(str):
    def __new__(cls, outcome, diagnostics=None):
        value = super().__new__(cls, outcome)
        value.diagnostics = diagnostics or {}
        return value


def snapshot_action_ids(conn, server_id):
    try:
        return {get_resource_field(a, "request_id") for a in conn.compute.server_actions(server_id)}
    except Exception:  # noqa: BLE001 - no action history: correlation is optional
        return None  # no historical correlation is possible


def collect_new_server_action_diagnostics(conn, server_id, preexisting_action_request_ids):
    if preexisting_action_request_ids is None:
        return {}
    try:
        actions = [a for a in conn.compute.server_actions(server_id)
                   if get_resource_field(a, "request_id") not in preexisting_action_request_ids
                   and get_resource_field(a, "action") in ("os-migrateLive", "live-migration", "live_migration")]
        # Concurrent new actions are ambiguous: never pick an arbitrary error.
        if len(actions) != 1:
            return {}
        action = actions[0]
        request_id = get_resource_field(action, "request_id")
        if not request_id:
            return {}
        detail = conn.compute.get_server_action(request_id, server_id, ignore_missing=True)
        events = get_resource_field(detail, "events") or []
        return {"action_request_id": request_id, "message": get_resource_field(detail, "message"),
                "events": [{key: get_resource_field(e, key) for key in ("event", "result", "host", "details")}
                           for e in events]}
    except Exception as exc:  # noqa: BLE001 - diagnostics are best effort; the failure type is reported
        return {"diagnostics_unavailable": type(exc).__name__}


def build_migration_monitor_result(outcome, *, conn=None, server_id=None, migration=None, server=None,
                                   src_host=None, dst_host=None, reason=None,
                                   preexisting_action_request_ids=None):
    details = {"reason": reason, "migration_id": get_resource_field(migration, "id"),
               "migration_uuid": get_resource_field(migration, "uuid"),
               "raw_migration_status": get_resource_field(migration, "status"),
               "current_host": get_resource_field(server, "hypervisor_hostname", "OS-EXT-SRV-ATTR:hypervisor_hostname"),
               "expected_source": src_host, "expected_destination": dst_host}
    if conn is not None and outcome != "success":
        details.update(collect_new_server_action_diagnostics(conn, server_id, preexisting_action_request_ids))
    return MigrationMonitorResult(outcome, {k: v for k, v in details.items() if v is not None})


def format_migration_monitor_detail(result, submit_error=None, monitor_called=True):
    import json
    prefix = f"monitor_migration returned {result}" if monitor_called else "migration submission rejected"
    detail = prefix + "; " + "; ".join(f"{k}={json.dumps(v) if isinstance(v, (dict, list)) else v}"
                                       for k, v in getattr(result, "diagnostics", {}).items())
    if submit_error is not None:
        detail += f"; submit_error={submit_error}"
    return detail
