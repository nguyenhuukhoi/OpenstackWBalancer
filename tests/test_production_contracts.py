"""Recovery and isolation contracts introduced by the production refactor."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from compute_maintenance.adapters.persistence import HostState, StateError
from compute_maintenance.adapters.protocol import DrainResultWriter, read_result
from compute_maintenance.application.recovery import (
    MigrationJournal,
    validate_settle_window,
)
from compute_maintenance.application.stages import StageActions
from compute_maintenance.domain.errors import ConfigError, MigrationExecutionError
from compute_maintenance.domain.health import HealthReport, HealthStatus
from compute_maintenance.domain.maintenance_errors import CommandOutcomeUnknown
from compute_maintenance.profiles.schema import validate_profile


class Clock:
    def __init__(self, now=1000):
        self.now = now
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def test_restart_preserves_unknown_migration_and_never_retries(tmp_path):
    clock = Clock()
    journal = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    step = NS(vm_id="vm", src_host="a", dst_host="b")
    journal.begin(step, {"1"}, global_max_id=1)
    recovered = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    # Real Nova contract: the per-server endpoint lists in-progress rows only;
    # /os-migrations keeps every row. Row 1 predates the submit.
    compute = NS(get_server=lambda _: {"status": "ACTIVE", "hypervisor_hostname": "a"},
                 server_migrations=lambda _: [],
                 migrations=lambda **_: [{"id": 1, "status": "completed", "migration_type": "live-migration"}])
    # Just submitted, no row yet: still in progress, never a resubmit.
    with pytest.raises(MigrationExecutionError, match="in progress"):
        recovered.reconcile(NS(compute=compute))
    with pytest.raises(MigrationExecutionError, match="pending"):
        recovered.begin(step, set())
    assert recovered.state.data["pending"]["vm_id"] == "vm"
    # Server on a third host with only old rows: unprovable, journal retained.
    compute.get_server = lambda _: {"status": "ACTIVE", "hypervisor_hostname": "c"}
    with pytest.raises(MigrationExecutionError, match="UNKNOWN"):
        recovered.reconcile(NS(compute=compute))
    assert recovered.state.data["pending"]["vm_id"] == "vm"
    compute.get_server = lambda _: {"status": "ACTIVE", "hypervisor_hostname": "b"}
    assert recovered.reconcile(NS(compute=compute)) == "success"
    assert recovered.state.data["pending"] is None


def test_settle_covers_restart_and_external_migration(tmp_path):
    clock = Clock()
    journal = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    compute = NS(migrations=lambda: [{"status": "completed", "updated_at": "1970-01-01T00:17:00"}])
    journal.wait_settled(NS(compute=compute))
    assert clock.waits == [380]
    recovered = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    recovered.wait_settled(NS(compute=compute))
    assert clock.waits == [380]


@pytest.mark.parametrize("migration", [{"status": "unexpected"}, {"status": "completed"},
    {"status": "completed", "updated_at": "invalid"}, {"status": "completed", "updated_at": "2099-01-01"}])
def test_missing_or_unsafe_migration_history_fails_closed(tmp_path, migration):
    clock = Clock()
    journal = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    with pytest.raises(MigrationExecutionError):
        journal.wait_settled(NS(compute=NS(migrations=lambda: [migration])))
    assert clock.waits == []


def test_active_destination_waits_for_nova_cleanup_then_full_metric_window(tmp_path):
    clock = Clock()
    journal = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    def migrations():
        return ([{"status": "post-migrating"}] if clock.now < 1010 else
                [{"status": "completed", "updated_at": "1970-01-01T00:16:50"}])
    journal.wait_settled(NS(compute=NS(migrations=migrations)))
    assert clock.waits == [10, 360]
    assert clock.now == 1370


def test_nonterminal_migration_wait_is_bounded(tmp_path):
    clock = Clock()
    journal = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    with pytest.raises(MigrationExecutionError, match="600s"):
        journal.wait_settled(NS(compute=NS(migrations=lambda: [{"status": "running"}])))
    assert sum(clock.waits) == 600


def test_custom_promql_window_must_fit_settle_barrier():
    with pytest.raises(ConfigError):
        validate_settle_window(NS(prometheus_query_ram="avg_over_time(ram[10m])"), 360)
    validate_settle_window(NS(prometheus_query_ram="avg_over_time(ram[10m])"), 660)


def test_configuration_loads_do_not_leak_between_runs(tmp_path, monkeypatch):
    from compute_maintenance.config import LoadbalancerConfig
    monkeypatch.delenv("PROMETHEUS_QUERY_URL", raising=False)
    for key in ('PROMETHEUS_NODE_JOB', 'PROMETHEUS_LIBVIRT_JOB'):
        monkeypatch.setenv(key, 'test-job')
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text("PROMETHEUS_QUERY_URL=http://first/api/v1/query\n")
    second.write_text("PROMETHEUS_QUERY_URL=http://second/api/v1/query\n")
    assert LoadbalancerConfig.load_config(first).prometheus_query_url == "http://first/api/v1/query"
    assert LoadbalancerConfig.load_config(second).prometheus_query_url == "http://second/api/v1/query"
    assert "PROMETHEUS_QUERY_URL" not in os.environ


@pytest.mark.parametrize("age", [121, -1, "NaN"])
def test_raw_metric_freshness_rejects_stale_future_nonfinite(monkeypatch, age):
    import compute_maintenance.adapters.freshness as module
    from compute_maintenance.domain.errors import PrometheusError
    monkeypatch.setattr(module, "do_query", lambda *a: {"data": {"result": [
        {"metric": {"alias": "a"}, "value": [1000, str(age)]}]}})
    with pytest.raises(PrometheusError):
        module.validate_runtime_freshness(NS(prometheus_query_url="url", prometheus_node_job="node"), ["a"], {})


def test_rss_duplicate_fallback_never_partially_commits(monkeypatch):
    import compute_maintenance.adapters.prometheus as module
    rows = [{"metric": {"instanceId": vm, "host": "a"}, "value": [1000, value]}
            for vm, value in [("one", "1000"), ("two", "9000"), ("two", "100")]]
    monkeypatch.setattr(module, "do_query", lambda *a: {"data": {"result": rows}})
    samples = {("one", "a"): {"cpu_pct": 1}, ("two", "a"): {"cpu_pct": 1}}
    module.fill_vm_rss_fallback(NS(prometheus_query_url="url", prometheus_libvirt_job='libvirt-test'), ["a"], samples)
    assert all("ram_rss_gb" not in sample for sample in samples.values())


def test_post_restore_failure_quarantines_and_resume_rechecks_vms():
    from compute_maintenance.domain.maintenance_errors import StepFailed
    from test_maintenance_safety import SafetyTests
    fixture = SafetyTests()
    fixture.setUp()
    try:
        state = fixture.checkpoint("post_restore")
        state.save(restore_intent=True)
        fixture.runner._active_state = state
        def process(*args):
            fixture.runner._active_state = state
            raise StepFailed("acceptance gate failed")
        def set_enabled(host, enabled, reason=None):
            fixture.service.update(Status="enabled" if enabled else "disabled", **{"Disabled Reason": reason})
        fixture.runner.process_host = process
        fixture.cloud.set_nova_enabled = set_enabled
        with pytest.raises(StepFailed):
            fixture.runner._run_all()
        assert fixture.service["Status"] == "disabled"
        assert state.data["next_stage"] == "health"
        assert state.data["quarantined"]
        fixture.servers = [{"ID": "newly-scheduled", "Status": "ACTIVE"}]
        with pytest.raises(StepFailed, match="unexpected"):
            fixture.runner.guard_stage("health", fixture.node, state)
    finally:
        fixture.doCleanups()


def test_result_protocol_rejects_stale_run_and_partial_file(tmp_path):
    path = tmp_path / "result.json"
    identity = {"run_id": "new", "cloud": "openstack", "aggregate": "agg", "host": "a", "mode": "apply"}
    writer = DrainResultWriter(path, identity)
    writer.record({"drain_host": "a", "result": "drain_completed", "drain_feasible": True,
                       "drain_blocked_vm_ids": [], "drain_unmovable_vm_ids": [], "drain_infeasible_reasons": []})
    writer.finish(0)
    assert read_result(path, identity, 0)["run_id"] == "new"
    with pytest.raises(StateError, match="mismatch"):
        read_result(path, dict(identity, run_id="old"), 0)
    path.write_text('{"schema_version":')
    with pytest.raises(StateError, match="invalid"):
        read_result(path, identity, 0)


def test_refresh_rejects_destination_already_above_drain_cap():
    from compute_maintenance.application.migrations import validate_fresh_projection
    from compute_maintenance.domain.plan_context import PlanContext
    from wlbalancer_tests.test_wlbalancer import context
    ctx = context(host_ram_pct={"h1": 40, "h2": 85})
    state = PlanContext(all_vm_details=ctx.all_vm_details, host_cpu_pct=ctx.host_cpu_pct,
                        host_ram_pct=ctx.host_ram_pct, host_cores=ctx.host_cores, host_ram=ctx.host_ram)
    step = NS(vm_id="vm1", src_host="h1", dst_host="h2")
    plan = NS(plan_type="drain")
    with pytest.raises(MigrationExecutionError, match="drain destination caps"):
        validate_fresh_projection(state, step, plan, prevent_new_hotspot=True)
    validate_fresh_projection(state, step, plan, prevent_new_hotspot=False)


def test_planner_budget_is_unknown_not_infeasible():
    from compute_maintenance.domain.balance import find_best_move
    from compute_maintenance.domain.errors import PlanningBudgetExceeded
    from wlbalancer_tests.test_wlbalancer import context
    ctx = context(search_budget=1)
    with pytest.raises(PlanningBudgetExceeded, match="budget_exhausted"):
        find_best_move(ctx, "BALANCED", ["h1"], 10, .5, .5)


def stage_runner(tmp_path, hook):
    runner = StageActions()
    runner.profile = validate_profile({"name": "test", "hooks": {"maintenance": [hook]}})
    runner._active_state = HostState(tmp_path, "host")
    runner._active_state.save(hooks={})
    runner.sleep = lambda _: None
    return runner


def test_hook_timeout_remains_pending_and_cannot_replay(tmp_path):
    runner = stage_runner(tmp_path, {"id": "update", "cmd": "update", "retries": 3, "idempotency": "idempotent"})
    calls = []
    def ssh(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired("ssh", 10)
    node = NS(name="a", address="a", ssh=ssh)
    for _ in range(2):
        with pytest.raises(CommandOutcomeUnknown):
            runner.run_hooks("maintenance", node)
    assert len(calls) == 1
    assert runner._active_state.data["hooks"]["maintenance/update"]["outcome"] == "pending"


def test_reconciled_capture_is_persisted_and_done_hook_skipped(tmp_path):
    runner = stage_runner(tmp_path, {"id": "kernel", "cmd": "select", "check_cmd": "verify", "capture": "expected_kernel", "idempotency": "reconcile"})
    runner._active_state.save(hooks={"maintenance/kernel": {"outcome": "pending"}})
    calls = []
    def ssh(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "6.8.0-test\n", "")
    node = NS(name="a", address="a", ssh=ssh)
    runner.run_hooks("maintenance", node)
    runner.run_hooks("maintenance", node)
    assert calls == ["verify"]
    assert runner._active_state.data["expected_kernel"] == "6.8.0-test"


def test_health_requires_continuous_window_after_flap(tmp_path):
    runner = StageActions()
    clock = Clock(0)
    runner.monotonic, runner.sleep = clock, clock.sleep
    runner.args = NS(settle=20, health_timeout=60)
    runner.health_once = lambda *a: (clock.now != 10, "probe")
    state = HostState(tmp_path, "host")
    state.save(baseline={"containers": ["nova_compute"]})
    runner.health(NS(name="a", boot_id=lambda: "boot"), state)
    assert clock.now == 40
    assert state.data["healthy_duration"] == 20


def test_health_exception_is_unknown_not_healthy():
    report = HealthReport()
    def unavailable():
        raise OSError("transport failed")
    report.observe("systemd", unavailable)
    assert not report.ok
    assert report.components["systemd"].status == HealthStatus.UNKNOWN


@pytest.mark.skipif(os.name != "posix", reason="POSIX flock contract")
def test_child_inherits_lock_without_releasing_parent(tmp_path):
    from compute_maintenance.adapters.locking import coordinator_lock
    path = str(tmp_path / "lock")
    child = "from compute_maintenance.adapters.locking import shared_apply_lock; import sys;\nwith shared_apply_lock(sys.argv[1], int(sys.argv[2])): print('child holds lock')"
    competitor = "from compute_maintenance.adapters.locking import coordinator_lock; import sys;\nwith coordinator_lock(sys.argv[1]): print('acquired')"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    with coordinator_lock(path) as stream:
        p = subprocess.run([sys.executable, "-c", child, path, str(stream.fileno())], pass_fds=(stream.fileno(),),
                           env=env, capture_output=True, check=False)
        assert p.returncode == 0, p.stderr
        p = subprocess.run([sys.executable, "-c", competitor, path], env=env, capture_output=True, check=False)
        assert p.returncode != 0
    p = subprocess.run([sys.executable, "-c", competitor, path], env=env, capture_output=True, check=False)
    assert p.returncode == 0, p.stderr
