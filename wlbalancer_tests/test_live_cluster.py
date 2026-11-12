"""Read-only integration probes against the current OpenStack cluster.

Run explicitly with WLB_LIVE_CONFIG and WLB_LIVE_AGGREGATE set.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.legacy_api import balance as wlb  # noqa: E402


@pytest.fixture(scope="module")
def live_snapshot():
    config_path = os.getenv("WLB_LIVE_CONFIG")
    aggregate_name = os.getenv("WLB_LIVE_AGGREGATE", "loadtest-agg")
    if not config_path:
        pytest.skip("set WLB_LIVE_CONFIG to enable live cluster probes")

    cfg = wlb.LoadbalancerConfig.load_config(config_path)
    conn = wlb.get_openstack_connection()
    aggregate_map = wlb.get_aggregate_map(conn, [aggregate_name], True)
    hosts = aggregate_map[aggregate_name]
    excludes = wlb.load_exclude_vm_patterns(cfg.exclude_vms_file)
    cache = wlb.build_cluster_cache(conn, cfg, hosts, enforce_server_groups=True)
    state = wlb.evaluate_aggregate_state(
        cfg,
        conn,
        aggregate_name,
        hosts,
        enforce_server_groups=True,
        cluster_cache=cache,
        exclude_vm_patterns=excludes,
    )
    return cfg, conn, aggregate_name, hosts, cache, state


def test_selected_aggregate_has_multiple_unique_hosts(live_snapshot):
    _cfg, _conn, _name, hosts, _cache, _state = live_snapshot
    assert len(hosts) >= 2
    assert len(hosts) == len(set(hosts))


def test_every_compute_service_is_ready(live_snapshot):
    _cfg, conn, _name, hosts, _cache, _state = live_snapshot
    failures = {}
    for host in hosts:
        ready, reason = wlb.is_nova_compute_service_ready(conn, host)
        if not ready:
            failures[host] = reason
    assert failures == {}


def test_runtime_and_placement_cover_every_host(live_snapshot):
    _cfg, _conn, _name, hosts, cache, state = live_snapshot
    assert not state.get("error"), state.get("error")
    assert set(state["hosts_in_agg"]) == set(hosts)
    assert set(cache["host_resources"]) == set(hosts)
    for host in hosts:
        assert 0 <= state["host_cpu_pct"][host] <= 100
        assert 0 <= state["host_ram_pct"][host] <= 100
        assert state["host_cores"][host] > 0
        assert state["host_ram"][host] > 0
        for resource_class in ("VCPU", "MEMORY_MB"):
            record = state["host_placement"][host][resource_class]
            assert record["used"] >= 0
            assert record["nominal_capacity"] > 0
            assert record["policy_capacity"] <= record["nominal_capacity"]


def test_candidate_inputs_are_complete_and_nova_aligned(live_snapshot):
    _cfg, _conn, _name, hosts, _cache, state = live_snapshot
    for vm_id, details in state["all_vm_details"].items():
        assert details["status"] == wlb.ACTIVE_VM_STATUS
        assert details["metadata_complete"] is True
        assert details["metrics_complete"] is True
        assert details["vcpus"] > 0
        assert details["ram_gb"] > 0
        assert state["vm_host_map"][vm_id] in hosts


def test_required_vm_snapshots_are_present_and_nova_aligned(live_snapshot):
    """Optionally require named VMs to survive all Prometheus ambiguity gates.

    This catches a gap that a generic ``for all returned details`` assertion
    cannot: a VM with dual-host or incomplete metrics is deliberately omitted
    from ``all_vm_details``.  Live migration scenarios can set
    ``WLB_LIVE_REQUIRED_VMS`` to a comma-separated list and wait until every
    expected workload has a complete, single-host snapshot again.
    """
    raw = os.getenv("WLB_LIVE_REQUIRED_VMS", "")
    required_names = {name.strip() for name in raw.split(",") if name.strip()}
    if not required_names:
        pytest.skip("set WLB_LIVE_REQUIRED_VMS to enforce named VM convergence")

    _cfg, _conn, _name, hosts, cache, state = live_snapshot
    servers_by_name = {
        getattr(server, "name", ""): server
        for server in cache["server_cache"].values()
    }
    missing_servers = sorted(required_names - set(servers_by_name))
    assert missing_servers == [], f"required Nova servers not found: {missing_servers}"

    failures = {}
    for name in sorted(required_names):
        server = servers_by_name[name]
        details = state["all_vm_details"].get(server.id)
        if details is None:
            failures[name] = "missing complete single-host Prometheus snapshot"
            continue
        nova_host = wlb.get_server_current_host(server)
        metric_host = state["vm_host_map"].get(server.id)
        if nova_host != metric_host or metric_host not in hosts:
            failures[name] = f"Nova host={nova_host}, metric host={metric_host}"

    assert failures == {}


def test_decision_and_plan_obey_current_safety_invariants(live_snapshot):
    _cfg, _conn, _name, hosts, _cache, state = live_snapshot
    assert state["severity"] in {"OK", "WARNING", "CRITICAL", "UNKNOWN"}
    assert state["exit_code"] in {
        wlb.ICINGA_OK,
        wlb.ICINGA_WARNING,
        wlb.ICINGA_CRITICAL,
        wlb.ICINGA_UNKNOWN,
    }
    plan = state.get("best_plan")
    if plan is None:
        assert state["has_valid_move"] is False
        return

    assert state["has_valid_move"] is True
    assert plan.improvement >= state["improvement_threshold"]
    for step in plan.steps:
        assert step.src_host in hosts
        assert step.dst_host in hosts
        assert step.src_host != step.dst_host
        assert step.vm_id in state["all_vm_details"]


def test_migration_history_can_be_snapshotted_for_test_workload(live_snapshot):
    _cfg, conn, _name, _hosts, cache, _state = live_snapshot
    test_vm = next(
        (
            server
            for server in cache["server_cache"].values()
            if (getattr(server, "name", "") or "").startswith("loadtest-")
        ),
        None,
    )
    assert test_vm is not None
    migration_ids = wlb.get_server_migration_ids(conn, test_vm.id)
    assert all(isinstance(value, str) and value for value in migration_ids)


def test_previous_low_benefit_make_room_audit_is_rejected_by_new_policy(live_snapshot):
    """Replay a real audit event without mutating the cluster again."""
    event_path = os.getenv("WLB_LIVE_REPLAY_EVENT")
    if not event_path:
        pytest.skip("set WLB_LIVE_REPLAY_EVENT to replay a real MAKE_ROOM audit")

    events = [
        json.loads(line)
        for line in Path(event_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = next(
        (item for item in reversed(events) if item.get("plan_type") == "make_room"),
        None,
    )
    assert event is not None, "audit contains no MAKE_ROOM event"

    cfg, _conn, _name, _hosts, _cache, _state = live_snapshot
    plan = SimpleNamespace(plan_type="make_room")
    new_threshold = wlb.determine_plan_improvement_threshold(plan, cfg)
    improvement = float(event["plan_improvement"])
    old_threshold = float(event["improvement_threshold"])

    assert old_threshold <= improvement < new_threshold
