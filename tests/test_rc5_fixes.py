"""Regression tests for the rc4 audit findings fixed in rc5.

Each test is named after the audit ID (audit-rc4-20261110/AUDIT-RC4-2026-11-10.md).
Nova fakes follow the real contract verified on the lab:
  * GET /servers/{id}/migrations lists in-progress live migrations only;
  * GET /os-migrations keeps terminal rows (error/failed/completed);
  * the API sets task_state=migrating before the conductor creates the row.
"""
import json
import logging
import time
from types import SimpleNamespace as NS

import pytest

import compute_maintenance.application.migrations as M
import compute_maintenance.application.recovery as R
import compute_maintenance.application.snapshots as S
import compute_maintenance.config as C
from compute_maintenance.adapters import prometheus as P
from compute_maintenance.adapters.cooldown import CooldownStore
from compute_maintenance.domain import balance, drain, models, policies
from compute_maintenance.domain.errors import (
    MigrationExecutionError,
    PlanningBudgetExceeded,
)
from compute_maintenance.domain.plan_context import PlanContext
from compute_maintenance.domain.snapshot import freeze_context_inputs


# --------------------------------------------------------------- helpers
class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Row(dict):
    __getattr__ = dict.get


class NotFoundException(Exception):
    status_code = 404


class FakeNova:
    """Minimal Nova with the real migration-listing contract."""

    def __init__(self, host="src", status="ACTIVE", task=None):
        self.server = {"status": status, "compute_host": host, "task_state": task}
        self.global_rows = []      # /os-migrations rows of the VM
        self.in_progress = []      # /servers/{id}/migrations rows
        self.submitted = False
        self.deleted = False
        self.compute = self

    def get_server(self, _vm):
        if self.deleted:
            raise NotFoundException("gone")
        return Row(self.server)

    def server_migrations(self, _vm):
        return list(self.in_progress)

    def migrations(self, **query):
        return list(self.global_rows)

    def live_migrate_server(self, **_):
        self.submitted = True


def journal_at(tmp_path, clock=None, **kw):
    clock = clock or Clock()
    return R.MigrationJournal(tmp_path, settle_seconds=360, clock=clock, sleeper=clock.sleep, **kw), clock


def pending_step():
    return NS(vm_id="vm", src_host="src", dst_host="dst")


def inv(total, used, ratio=1.0):
    return policies.build_placement_inventory_record(
        {"total": total, "reserved": 0, "allocation_ratio": ratio}, used, 5.0)


def placement(vcpu_used=0, mem_used=0):
    return {"resource_provider_uuid": "x", "VCPU": inv(32, vcpu_used, 4.0),
            "MEMORY_MB": inv(102400, mem_used)}


def vm(vcpus=2, ram_gb=4, cpu_pct=20.0, ram_pct=50.0, **extra):
    d = {"vcpus": vcpus, "ram_gb": ram_gb, "cpu_pct": cpu_pct, "ram_pct": ram_pct, "ram_rss_gb": None,
         "status": "ACTIVE", "metadata_complete": True, "metrics_complete": True}
    d.update(extra)
    return d


def ctx(hosts, cpu, ram, vms, vm_host, place=None, groups=None, drain_mode=False, host_az=None):
    groups = groups or {}
    c = models.BalanceContext(
        effective_hosts=list(hosts), host_cpu_pct=dict(cpu), host_ram_pct=dict(ram),
        host_cores={h: 32 for h in hosts}, host_ram={h: 100.0 for h in hosts},
        host_free_cores={h: 0 for h in hosts}, host_free_ram={h: 0 for h in hosts},
        host_placement=place or {h: placement() for h in hosts}, vm_host_map=dict(vm_host),
        all_vm_details=vms, vm_to_groups={v: groups.get(v, []) for v in vm_host},
        enforce_server_groups=True, dest_ready_hosts=set(hosts),
        host_running_vms={h: sum(1 for x in vm_host.values() if x == h) for h in hosts},
        host_max_instances={h: 0 for h in hosts}, excluded_vm_ids=set(),
        reject_stats={}, dest_reject_stats={}, skip_stats={}, ignore_cooldown=drain_mode,
        vm_status_map={v: "ACTIVE" for v in vm_host}, cooldown_remaining={v: 0.0 for v in vm_host},
        host_az=host_az)
    return freeze_context_inputs(c)


# ======================================================================= J1
class TestJournalReconcile:
    def test_failed_migration_is_proven_from_os_migrations(self, tmp_path):
        journal, _ = journal_at(tmp_path)
        journal.begin(pending_step(), set(), global_max_id=10)
        nova = FakeNova(host="src")
        nova.global_rows = [Row(id=9, status="completed", migration_type="live-migration"),
                            Row(id=11, status="error", migration_type="live-migration")]
        assert journal.reconcile(nova) == "failed"
        assert journal.state.data["pending"] is None
        assert journal.state.data["last_outcome"] == "failed"

    def test_failed_migration_leaving_vm_in_error_on_source_is_closed(self, tmp_path):
        # Lab incident 382: MTU mismatch -> VM ERROR, still running on source.
        journal, _ = journal_at(tmp_path)
        journal.begin(pending_step(), set(), global_max_id=10)
        nova = FakeNova(host="src", status="ERROR")
        nova.global_rows = [Row(id=11, status="error", migration_type="live-migration")]
        assert journal.reconcile(nova) == "failed"

    def test_rows_from_before_the_submit_are_ignored(self, tmp_path):
        journal, clock = journal_at(tmp_path)
        journal.begin(pending_step(), set(), global_max_id=10)
        nova = FakeNova(host="src")
        nova.global_rows = [Row(id=10, status="error", migration_type="live-migration")]
        with pytest.raises(R.MigrationStillInProgress):
            journal.reconcile(nova)  # no row of ours yet, within the grace period
        clock.sleep(C.MIGRATION_NOT_STARTED_GRACE_SECONDS)
        assert journal.reconcile(nova) == "not_started"

    def test_task_state_means_in_progress_even_without_a_row(self, tmp_path):
        journal, clock = journal_at(tmp_path)
        journal.begin(pending_step(), set(), global_max_id=0)
        clock.sleep(10_000)
        nova = FakeNova(host="src", task="migrating")
        with pytest.raises(R.MigrationStillInProgress):
            journal.reconcile(nova)
        assert journal.state.data["pending"]["vm_id"] == "vm"

    def test_running_row_means_in_progress(self, tmp_path):
        journal, _ = journal_at(tmp_path)
        journal.begin(pending_step(), set(), global_max_id=0)
        nova = FakeNova(host="src")
        nova.global_rows = [Row(id=1, status="running", migration_type="live-migration")]
        with pytest.raises(R.MigrationStillInProgress):
            journal.reconcile(nova)

    def test_deleted_server_closes_the_intent(self, tmp_path):
        journal, _ = journal_at(tmp_path)
        journal.begin(pending_step(), set(), global_max_id=0)
        nova = FakeNova()
        nova.deleted = True
        assert journal.reconcile(nova) == "vm_deleted"

    def test_unprovable_state_keeps_the_journal(self, tmp_path):
        journal, _ = journal_at(tmp_path)
        journal.begin(pending_step(), set(), global_max_id=0)
        nova = FakeNova(host="elsewhere")
        nova.global_rows = [Row(id=1, status="error", migration_type="live-migration")]
        with pytest.raises(MigrationExecutionError, match="UNKNOWN"):
            journal.reconcile(nova)
        with pytest.raises(MigrationExecutionError, match="pending"):
            journal.begin(pending_step(), set())

    def test_rc4_journal_without_global_max_id_uses_creation_time(self, tmp_path):
        clock = Clock(now=1794305352.8)  # submitted_at of the lab incident
        journal, _ = journal_at(tmp_path, clock=clock)
        journal.begin(pending_step(), set())
        nova = FakeNova(host="src", status="ERROR")
        nova.global_rows = [Row(id=300, status="error", migration_type="live-migration",
                                created_at="2026-11-09T10:00:00.000000"),
                            Row(id=382, status="error", migration_type="live-migration",
                                created_at="2026-11-10T10:09:13.000000")]
        assert journal.reconcile(nova) == "failed"

    def test_invalid_global_max_id_is_rejected_on_load(self, tmp_path):
        journal, _ = journal_at(tmp_path)
        journal.begin(pending_step(), set(), global_max_id=3)
        with open(journal.state.path) as handle:
            data = json.load(handle)
        data["pending"]["global_max_id"] = "3"
        with open(journal.state.path, "w") as handle:
            json.dump(data, handle)
        with pytest.raises(MigrationExecutionError, match="invalid pending"):
            R.MigrationJournal(tmp_path)

    def test_wait_settled_waits_for_own_migration_then_settles(self, tmp_path):
        journal, _clock = journal_at(tmp_path)
        journal.begin(pending_step(), set(), global_max_id=0)
        nova = FakeNova(host="src", task="migrating")
        polls = {"n": 0}
        real_get = nova.get_server

        def get_server(vm_id):
            polls["n"] += 1
            if polls["n"] > 3:
                nova.server.update(compute_host="dst", task_state=None)
            return real_get(vm_id)
        nova.get_server = get_server
        journal.wait_settled(nova, hosts=["src", "dst"])
        assert journal.state.data["pending"] is None
        assert journal.state.data["last_outcome"] == "success"


class TestExecutorFailurePath:
    """execute_migration_step with the journal, against the real Nova contract."""

    @pytest.fixture
    def harness(self, tmp_path, monkeypatch):
        events, mails = [], []
        monkeypatch.setattr(M._report, "append_migration_event", lambda e: events.append(e))
        monkeypatch.setattr(M._report, "build_migration_event", lambda state, result, **kw: result)
        monkeypatch.setattr(M._notify, "send_migration_alert_email", lambda *a, **k: mails.append(a[6]))
        monkeypatch.setattr(M, "snapshot_action_ids", lambda conn, vm: set())
        monkeypatch.setattr(M, "validate_fresh_projection", lambda *a, **k: None)
        monkeypatch.setattr(M, "validate_migration_step", lambda *a, **k: (True, "ok"))
        monkeypatch.setattr(C, "POLL_INTERVAL", 0)
        monkeypatch.setattr(C, "MIGRATION_RECORD_DISCOVERY_TIMEOUT", 0.05)
        store = CooldownStore(str(tmp_path / "cool.json"))
        cfg = NS(cooldown_store=store)
        journal, clock = journal_at(tmp_path / "ops")
        cfg.migration_journal = journal
        state = PlanContext(aggregate="agg", hosts_in_agg=["src", "dst"], collected_at=time.time(),
                            all_vm_details={"vm": {}}, all_vm_names={"vm": "vm"},
                            exclude_unready_hosts=False)  # drain plan context (rc10)
        monkeypatch.setattr(M._snapshot, "evaluate_aggregate_state",
                            lambda *a, **k: PlanContext(**dict(state, collected_at=time.time())))
        step = NS(vm_id="vm", src_host="src", dst_host="dst", vcpus=1, vm_ram_gb=1)
        plan = NS(plan_type="drain", steps=[step])
        return NS(events=events, mails=mails, store=store, cfg=cfg, journal=journal, clock=clock,
                  state=state, step=step, plan=plan)

    def test_failed_migration_is_recorded_and_journal_closed(self, harness):
        nova = FakeNova(host="src")
        nova.global_rows = [Row(id=5, status="completed", migration_type="live-migration")]
        polls = {"n": 0}

        def server_migrations(_vm):
            if not nova.submitted:
                return []
            polls["n"] += 1
            if polls["n"] == 1:
                return [Row(id=6, status="running", source_node="src", dest_node="dst")]
            nova.global_rows = nova.global_rows + [Row(id=6, status="error", migration_type="live-migration")]
            return []
        nova.server_migrations = server_migrations

        ok = M.execute_migration_step(harness.cfg, nova, harness.state, harness.step, harness.plan, 1,
                                      send_error=True)
        assert ok is False
        assert harness.events == ["migration_failed"]
        assert harness.mails == ["migration_failed"]
        assert "vm" in harness.store.failed
        assert harness.journal.state.data["pending"] is None
        assert harness.journal.state.data["last_outcome"] == "failed"

    def test_precheck_failure_is_detected_from_os_migrations_at_once(self, harness):
        nova = FakeNova(host="src")
        original_submit = nova.live_migrate_server

        def submit(**kw):
            original_submit(**kw)
            nova.global_rows = [Row(id=1, status="error", migration_type="live-migration")]
        nova.live_migrate_server = submit
        started = time.monotonic()
        ok = M.execute_migration_step(harness.cfg, nova, harness.state, harness.step, harness.plan, 1)
        assert ok is False
        assert time.monotonic() - started < 5
        assert harness.events == ["migration_failed"]
        assert harness.journal.state.data["pending"] is None

    def test_still_running_after_timeout_records_then_keeps_journal(self, harness, monkeypatch):
        # rc9: one no-progress window; an unchanged row stalls after it.
        monkeypatch.setattr(C, "MIGRATION_TIMEOUT", 0.05)
        nova = FakeNova(host="src", task="migrating")
        nova.server_migrations = lambda _vm: (
            [Row(id=1, status="running", source_node="src", dest_node="dst")] if nova.submitted else [])
        nova.global_rows = []
        with pytest.raises(R.MigrationStillInProgress):
            M.execute_migration_step(harness.cfg, nova, harness.state, harness.step, harness.plan, 1)
        assert harness.events == ["migration_timeout"]
        assert "vm" in harness.store.failed
        assert harness.journal.state.data["pending"]["vm_id"] == "vm"

    def test_revalidation_rejection_sets_short_cooldown(self, harness, monkeypatch):
        monkeypatch.setattr(M, "validate_migration_step", lambda *a, **k: (False, "VM is locked"))
        nova = FakeNova()
        ok = M.execute_migration_step(harness.cfg, nova, harness.state, harness.step, harness.plan, 1)
        assert ok is None and not nova.submitted  # None = refused before submit
        assert harness.events == ["step_revalidation_failed"]
        remaining = harness.store.remaining("vm")
        assert 0 < remaining <= C.VM_REJECTED_MOVE_COOLDOWN_SECONDS


# ================================================================ PL2
class TestLockedAndBusyVms:
    def test_balance_skips_locked_and_busy_vms(self):
        vms = {"locked": vm(locked=True), "busy": vm(task_state="image_snapshot"), "ok": vm()}
        c = ctx(["a", "b"], {"a": 60, "b": 20}, {"a": 60, "b": 20}, vms,
                {"locked": "a", "busy": "a", "ok": "a"})
        assert balance.vm_candidate_details(c, "locked") is None
        assert balance.vm_candidate_details(c, "busy") is None
        assert balance.vm_candidate_details(c, "ok") is not None
        assert c.skip_stats == {"locked": 1, "task_in_progress": 1}

    def test_drain_reports_locked_vm_as_unmovable_and_busy_as_transient(self):
        vms = {"locked": vm(locked=True), "busy": vm(task_state="rebuilding"), "pinned": vm(
            unsupported_resources="hw:cpu_policy=dedicated")}
        c = ctx(["a", "b"], {"a": 30, "b": 20}, {"a": 30, "b": 20}, vms,
                {"locked": "a", "busy": "a", "pinned": "a"}, drain_mode=True)
        reasons = {vm_id: drain.classify_drain_vm(c, vm_id)[1] for vm_id in vms}
        assert reasons == {"locked": "locked", "busy": "task_in_progress", "pinned": "unsupported_resources"}
        assert "task_in_progress" in C.DRAIN_TRANSIENT_UNMOVABLE_REASONS
        assert "locked" not in C.DRAIN_TRANSIENT_UNMOVABLE_REASONS


# ================================================================ PL1
def test_PL1_make_room_budget_does_not_scale_with_servers_outside_the_aggregate():
    cfg = NS(enable_make_room=True, make_room_max_vm_cpu_percent=10.0,
             make_room_max_vm_ram_percent=50.0, make_room_min_improvement=0.30)
    hosts = [f"h{i}" for i in range(6)]
    vms, vh = {}, {}
    for h in hosts:
        for j in range(30):
            v = f"{h}-v{j}"
            vms[v] = vm(4, 16, 20.0, 50.0)
            vh[v] = h
    for k in range(2000):
        vh[f"ext{k}"] = f"other-{k % 20}"
    cpu = {h: 40.0 + i * 0.5 for i, h in enumerate(hosts)}
    ram = {h: 60.0 + i * 0.5 for i, h in enumerate(hosts)}
    place = {h: placement() for h in hosts}
    place["h5"] = placement(mem_used=97280 - 8000)
    c = ctx(hosts, cpu, ram, vms, vh, place)
    base = balance.compute_baseline_metrics(cpu, ram)
    try:
        balance.search_best_make_room_for_state(
            c, cfg, base["mode"], base["baseline_weighted_mad"], base["cpu_weight"], base["ram_weight"])
    except PlanningBudgetExceeded:
        pytest.fail("MAKE_ROOM budget charged for servers outside the aggregate")
    assert c.candidates_examined < 100000


# ============================================================ PL3 / PL4
def test_PL3_make_room_never_parks_the_room_vm_on_the_primary_source():
    cfg = NS(enable_make_room=True, make_room_max_vm_cpu_percent=10.0,
             make_room_max_vm_ram_percent=50.0, make_room_min_improvement=0.30)
    vms = {"p": vm(4, 16, 10.0, 90.0), "r": vm(2, 8, 5.0, 100.0)}
    vh = {"p": "A", "r": "B"}
    cpu = {"A": 30.0, "B": 20.0, "C": 25.0}
    ram = {"A": 90.0, "B": 50.0, "C": 76.0}
    place = {"A": placement(), "B": placement(mem_used=97280 - 10000), "C": placement()}
    c = ctx(["A", "B", "C"], cpu, ram, vms, vh, place)
    base = balance.compute_baseline_metrics(cpu, ram)
    plan = balance.search_best_make_room_for_state(
        c, cfg, base["mode"], base["baseline_weighted_mad"], base["cpu_weight"], base["ram_weight"])
    # rc4 planned "r B->A" (A = hot primary source, 90% -> 98% between the
    # steps). The only other buffer, C, would cross 80% RAM. No valid plan.
    assert plan is None


def test_PL4_destination_already_cpu_hot_is_refused_in_ram_hotspot_mode():
    vms = {"v": vm(8, 8, 100.0, 50.0)}
    cpu = {"A": 30.0, "B": 85.0, "C": 30.0}
    ram = {"A": 88.0, "B": 30.0, "C": 70.0}
    c = ctx(["A", "B", "C"], cpu, ram, vms, {"v": "A"})
    base = balance.compute_baseline_metrics(cpu, ram)
    move, _imp, _scope, _src = balance.search_best_move_for_state(
        c, base["mode"], base["baseline_weighted_mad"], base["cpu_weight"], base["ram_weight"], 0.15)
    # rc4 chose A->B with B's CPU projected at 110%.
    assert move is None
    assert c.dest_reject_stats["B"]["would_create_new_hotspot"] >= 1
    assert balance.destination_hot_after({"B": 110.0}, {"B": 30.0}, "B")


# ============================================================ PL5 / PL6
def group(name, policy, members, rules=None):
    return {"id": name, "name": name, "policy": policy, "members": members, "rules": rules or {}}


def test_PL5_drain_prefers_spreading_soft_anti_affinity_members():
    g = group("etcd", "soft-anti-affinity", ["m1", "m2", "m3"])
    vms = {"m1": vm(), "m2": vm(), "m3": vm()}
    vh = {"m1": "A", "m2": "A", "m3": "B"}
    c = ctx(["A", "B", "C", "D"], {h: 20.0 for h in "ABCD"}, {h: 20.0 for h in "ABCD"}, vms, vh,
            groups={m: [g] for m in vms}, drain_mode=True)
    plan = drain.plan_host_drain(c, "agg", "A", 0.5, 0.5)
    placed = {s.vm_id: s.dst_host for s in plan.steps}
    assert set(placed.values()) == {"C", "D"} and len(set(placed.values())) == 2
    assert not plan.soft_policy_warnings


def test_PL5_unavoidable_soft_violation_is_reported():
    g = group("etcd", "soft-anti-affinity", ["m1", "m2", "m3"])
    vms = {"m1": vm(), "m2": vm(), "m3": vm()}
    vh = {"m1": "A", "m2": "B", "m3": "C"}
    c = ctx(["A", "B", "C"], {h: 20.0 for h in "ABC"}, {h: 20.0 for h in "ABC"}, vms, vh,
            groups={m: [g] for m in vms}, drain_mode=True)
    plan = drain.plan_host_drain(c, "agg", "A", 0.5, 0.5)
    assert plan.feasible and len(plan.soft_policy_warnings) == 1


def test_PL5_balance_never_worsens_soft_anti_affinity():
    g = group("etcd", "soft-anti-affinity", ["m1", "m2"])
    vms = {"m1": vm(8, 8, 90.0, 50.0), "m2": vm()}
    vh = {"m1": "A", "m2": "B"}
    c = ctx(["A", "B"], {"A": 60.0, "B": 10.0}, {"A": 40.0, "B": 40.0}, vms, vh,
            groups={m: [g] for m in vms})
    base = balance.compute_baseline_metrics(c.host_cpu_pct, c.host_ram_pct)
    move, *_ = balance.search_best_move_for_state(
        c, base["mode"], base["baseline_weighted_mad"], base["cpu_weight"], base["ram_weight"], 0.15)
    assert move is None
    assert c.reject_stats.get("soft_server_group", 0) >= 1


def test_PL6_anti_affinity_honours_max_server_per_host():
    g = group("web", "anti-affinity", ["w1", "w2", "w3"], rules={"max_server_per_host": 2})
    vh = {"w1": "B", "w2": "C", "w3": "A"}
    assert policies.is_server_group_move_compliant("w3", "A", "B", vh, {"w3": [g]})
    vh2 = {"w1": "B", "w2": "B", "w3": "A"}
    assert not policies.is_server_group_move_compliant("w3", "A", "B", vh2, {"w3": [g]})


def test_PL6_pre_split_affinity_group_can_join_any_member_host():
    g = group("db", "affinity", ["d1", "d2", "d3"])
    vh = {"d1": "B", "d2": "C", "d3": "A"}
    assert policies.is_server_group_move_compliant("d3", "A", "B", vh, {"d3": [g]})
    assert not policies.is_server_group_move_compliant("d3", "A", "D", vh, {"d3": [g]})


def test_PL6_existing_violation_elsewhere_does_not_block_unrelated_move():
    g = group("web", "anti-affinity", ["w1", "w2", "w3"])
    vh = {"w1": "B", "w2": "B", "w3": "A"}  # already violated on B
    assert policies.is_server_group_move_compliant("w3", "A", "C", vh, {"w3": [g]})
    assert not policies.is_server_group_move_compliant("w3", "A", "B", vh, {"w3": [g]})


# ================================================================ PL8 AZ
def test_PL8_moves_stay_inside_the_availability_zone():
    vms = {"v": vm(availability_zone="az1")}
    c = ctx(["A", "B", "C"], {"A": 60.0, "B": 10.0, "C": 10.0}, {h: 40.0 for h in "ABC"}, vms,
            {"v": "A"}, host_az={"A": "az1", "B": "az2", "C": "az1"})
    assert balance.availability_zone_allows(c, vms["v"], "A", "C")
    assert not balance.availability_zone_allows(c, vms["v"], "A", "B")
    plan = drain.plan_host_drain(ctx(["A", "B", "C"], {h: 20.0 for h in "ABC"}, {h: 20.0 for h in "ABC"},
                                     vms, {"v": "A"}, drain_mode=True,
                                     host_az={"A": "az1", "B": "az2", "C": "az1"}),
                                 "agg", "A", 0.5, 0.5)
    assert [s.dst_host for s in plan.steps] == ["C"]


# ===================================================== F1 / F2 / F3 / F6
def _vm_metric_rows(monkeypatch, mem_rows, cpu_rows):
    payloads = iter([{"data": {"result": mem_rows}}, {"data": {"result": cpu_rows}}])
    monkeypatch.setattr(P, "do_query", lambda url, query: next(payloads))
    monkeypatch.setattr(P, "fill_vm_rss_fallback", lambda *a: None)
    cfg = NS(prometheus_query_vm_ram_rss="", prometheus_query_url="u", prometheus_libvirt_job="lv")
    return P.get_all_vm_metrics(cfg, ["h1"])


def row(labels, value):
    return {"metric": labels, "value": [0, str(value)]}


def test_F2_relabelled_vm_keeps_the_larger_sample(monkeypatch):
    metrics = _vm_metric_rows(
        monkeypatch,
        [row({"instanceId": "v", "host": "h1", "instanceName": "old"}, 30),
         row({"instanceId": "v", "host": "h1", "instanceName": "new"}, 40)],
        [row({"instanceId": "v", "host": "h1"}, 10)])
    assert metrics["v"]["ram_pct"] == 40.0


def test_F3_unrelated_target_without_alias_does_not_break_selected_hosts(monkeypatch):
    monkeypatch.setattr(P, "do_query", lambda url, q: {"data": {"result": [
        row({"alias": "h1"}, 10), row({"instance": "ceph-new:9100"}, 50),
        row({"alias": "other"}, 1), row({"alias": "other"}, 2)]}})
    assert P.get_metrics("u", "q", ["h1"]) == {"h1": {"host": "h1", "value": 10.0}}


def test_F6_credentials_are_redacted_from_prometheus_errors(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("failed for url: http://wlb:S3cret@prom:9090/api/v1/query")
    monkeypatch.setattr(P.requests, "get", boom)
    with pytest.raises(Exception) as info:
        P.do_query("http://wlb:S3cret@prom:9090/api/v1/query", "up")
    assert "S3cret" not in str(info.value)


# ======================================================= F10 settle parse
@pytest.mark.parametrize("query,seconds", [
    ("rate(x[5m])", 300), ("avg_over_time(x[10m30s])", 630), ("max_over_time(x[1h5m])", 3900),
    ("rate(x[5m] offset 10m)", 900), ("max_over_time(rate(x[1m])[15m:30s])", 900),
])
def test_F10_promql_lookback_understands_compound_ranges_and_offsets(query, seconds):
    assert R.promql_lookback_seconds(query) == seconds


def test_F10_settle_window_rejects_long_custom_queries():
    with pytest.raises(Exception, match="settle"):
        R.validate_settle_window(NS(prometheus_query_cpu_used="rate(x[5m] offset 10m)"), 360)


# ================================================== drain settle (H2/D1)
class TestDrainCompensation:
    def test_recent_arrivals_keep_only_fresh_successful_moves(self):
        now = 10_000.0
        iso = lambda age: time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now - age))
        rows = [
            (Row(id=1, status="completed", server_id="a", source_compute="x", dest_compute="d", updated_at=iso(30)), now - 30),
            (Row(id=2, status="error", server_id="b", source_compute="x", dest_compute="d"), now - 10),
            (Row(id=3, status="completed", server_id="c", source_compute="x", dest_compute="d"), now - 900),
        ]
        arrivals = R.recent_arrivals(rows, now, 300)
        assert [a["vm_id"] for a in arrivals] == ["a"]

    def test_compensation_adds_remaining_share_of_ram_to_the_destination(self):
        cpu, ram = {"d": 20.0}, {"d": 40.0}
        details = {"a": vm(4, 32, 50.0, 50.0)}  # 2 cores, 16 GiB used
        added = S.apply_arrival_compensation(
            [{"vm_id": "a", "dst_host": "d", "age": 150.0}], cpu, ram, {"d": 32}, {"d": 100.0},
            {"a": "d"}, details, {}, 300.0)
        assert ram["d"] == pytest.approx(48.0)  # 16 GiB * (1 - 150/300) of 100 GiB
        assert cpu["d"] == pytest.approx(20.0)  # irate already caught up after 60 s
        assert added["d"]["vms"] == 1

    def test_vm_without_metrics_is_compensated_with_its_full_flavor(self):
        cpu, ram = {"d": 20.0}, {"d": 40.0}
        server = NS(flavor={"vcpus": 4, "ram": 8192})
        S.apply_arrival_compensation(
            [{"vm_id": "a", "dst_host": "d", "age": 0.0}], cpu, ram, {"d": 32}, {"d": 100.0},
            {"a": "d"}, {}, {"a": server}, 300.0)
        assert ram["d"] == pytest.approx(48.0) and cpu["d"] == pytest.approx(32.5)

    def test_vm_that_moved_on_is_not_counted(self):
        cpu, ram = {"d": 20.0}, {"d": 40.0}
        S.apply_arrival_compensation(
            [{"vm_id": "a", "dst_host": "d", "age": 0.0}], cpu, ram, {"d": 32}, {"d": 100.0},
            {"a": "elsewhere"}, {"a": vm()}, {}, 300.0)
        assert (cpu["d"], ram["d"]) == (20.0, 40.0)

    def test_settle_for_compensation_does_not_wait_the_metric_window(self, tmp_path):
        journal, clock = journal_at(tmp_path)
        nova = FakeNova()
        recent = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(clock() - 20))
        nova.global_rows = [Row(id=1, status="completed", server_id="a", source_compute="s",
                                dest_compute="d", updated_at=recent)]
        before = clock()
        arrivals = journal.settle_for_compensation(nova, ["s", "d"], 300)
        assert clock() == before
        assert [a["vm_id"] for a in arrivals] == ["a"]


# ============================================================ F5 cooldown
def test_F5_rejected_cooldown_round_trips_and_expires(tmp_path, monkeypatch):
    path = tmp_path / "cool.json"
    store = CooldownStore(str(path))
    store.rejected["v"] = time.time()
    assert store.save()
    again = CooldownStore(str(path))
    again.load()
    assert "v" in again.rejected and again.remaining("v") > 0
    again.prune(now=time.time() + C.VM_REJECTED_MOVE_COOLDOWN_SECONDS + 1)
    assert "v" not in again.rejected


# ======================================================== OpenStack reads
def test_A1_reads_are_retried_before_failing(caplog):
    from compute_maintenance.adapters.openstack import retry_read
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("Request to /servers/detail timed out")
        return ["ok"]
    with caplog.at_level(logging.WARNING):
        assert retry_read("list", flaky, sleeper=lambda s: None) == ["ok"]
    assert calls["n"] == 3

    def dead():
        raise TimeoutError("down")
    with pytest.raises(TimeoutError):
        retry_read("list", dead, sleeper=lambda s: None)


def test_unsupported_flavor_resources_are_detected():
    from compute_maintenance.adapters.openstack import unsupported_flavor_resources
    assert unsupported_flavor_resources({"extra_specs": {}}) is None
    assert unsupported_flavor_resources({"extra_specs": {"hw:cpu_policy": "shared"}}) is None
    assert "dedicated" in unsupported_flavor_resources({"extra_specs": {"hw:cpu_policy": "dedicated"}})
    assert "pci_passthrough:alias" in unsupported_flavor_resources(
        {"extra_specs": {"pci_passthrough:alias": "gpu:1"}})
    assert "resources:VGPU" in unsupported_flavor_resources({"extra_specs": {"resources:VGPU": "1"}})


# ======================================================= rolling-maint
def _stage(tmp_path, servers, locked, reboot="always"):
    from compute_maintenance.adapters.persistence import HostState
    from compute_maintenance.application.stages import StageActions
    from compute_maintenance.profiles.schema import validate_profile
    calls = []
    runner = StageActions()
    runner.profile = validate_profile({"name": "t", "reboot": reboot})
    service = {"Status": "enabled", "State": "up"}
    runner.cloud = NS(
        nova_service=lambda _h: dict(service),
        servers_on=lambda _h: list(servers),
        locked_servers_on=lambda _h: list(servers),  # a client may ignore --locked
        server_locked=lambda vm_id: vm_id in {s["ID"] for s in locked},
        set_nova_enabled=lambda *a: calls.append(("set", a)),
    )
    state = HostState(str(tmp_path), "h1")
    state.save(run_id="tok", left_on_host=[])
    return runner, state, calls


def test_PL2_rolling_maint_refuses_locked_running_vm_before_disable(tmp_path):
    from compute_maintenance.domain.maintenance_errors import StepFailed
    locked = [{"ID": "db-1", "Name": "db", "Status": "ACTIVE"}]
    runner, state, calls = _stage(tmp_path, locked, locked)
    with pytest.raises(StepFailed, match="locked running VM"):
        runner.disable(NS(name="h1"), state)
    assert calls == [] and "disabled_reason" not in state.data


def test_PL2_locked_but_stopped_vm_does_not_block(tmp_path):
    locked = [{"ID": "old", "Name": "old", "Status": "SHUTOFF"}]
    runner, state, calls = _stage(tmp_path, locked, locked)
    runner.require_owned = lambda *a: None
    runner.disable(NS(name="h1"), state)
    assert calls and calls[0][1][1] is False  # the host was disabled


def test_R6_reboot_refused_while_qemu_guests_still_run(tmp_path):
    from compute_maintenance.domain.maintenance_errors import StepFailed
    runner, state, _calls = _stage(tmp_path, [], [])
    rebooted = []
    node = NS(name="h1", address="h1", boot_id=lambda: "b1", running_guest_processes=lambda: 2,
              reboot=lambda: rebooted.append(1))
    with pytest.raises(StepFailed, match="QEMU guest process"):
        runner.reboot(node, state)
    assert not rebooted and "boot_id_before" not in state.data


def test_R6_qemu_count_parsing():
    from compute_maintenance.adapters.commands import Node
    from compute_maintenance.domain.maintenance_errors import StepFailed
    node = Node("h1", "10.0.0.1", "root", None, "docker")
    node.ssh = lambda cmd, **k: subprocess_result("28\n")
    assert node.running_guest_processes() == 28
    node.ssh = lambda cmd, **k: subprocess_result("0\n")
    assert node.running_guest_processes() == 0
    node.ssh = lambda cmd, **k: subprocess_result("garbage")
    with pytest.raises(StepFailed):
        node.running_guest_processes()


def subprocess_result(stdout):
    import subprocess
    return subprocess.CompletedProcess(["ssh"], 0, stdout, "")


def test_compensation_prefers_the_pre_migration_sample_over_the_full_flavor():
    cpu, ram = {"d": 20.0}, {"d": 40.0}
    server = NS(flavor={"vcpus": 8, "ram": 32768})
    S.apply_arrival_compensation(
        [{"vm_id": "a", "dst_host": "d", "age": 0.0}], cpu, ram, {"d": 64}, {"d": 100.0},
        {"a": "d"}, {}, {"a": server}, 300.0,
        ambiguous_samples={"a": {"cpu_pct": 10.0, "ram_pct": 25.0}})
    # 8 vCPU * 10% = 0.8 core of 64; 32 GiB * 25% = 8 GiB of 100 GiB
    assert cpu["d"] == pytest.approx(20.0 + 0.8 / 64 * 100)
    assert ram["d"] == pytest.approx(48.0)


def test_dual_series_vm_sample_is_kept_aside(monkeypatch):
    payloads = iter([
        {"data": {"result": [row({"instanceId": "v", "host": "h1"}, 30), row({"instanceId": "v", "host": "h2"}, 35)]}},
        {"data": {"result": [row({"instanceId": "v", "host": "h1"}, 12), row({"instanceId": "v", "host": "h2"}, 8)]}},
    ])
    monkeypatch.setattr(P, "do_query", lambda url, query: next(payloads))
    monkeypatch.setattr(P, "fill_vm_rss_fallback", lambda *a: None)
    cfg = NS(prometheus_query_vm_ram_rss="", prometheus_query_url="u", prometheus_libvirt_job="lv")
    ambiguous = {}
    assert P.get_all_vm_metrics(cfg, ["h1", "h2"], ambiguous=ambiguous) == {}
    assert ambiguous == {"v": {"ram_pct": 35.0, "cpu_pct": 12.0}}


def test_drain_replans_after_a_refused_step_instead_of_aborting(monkeypatch):
    import compute_maintenance.application.balancing as B
    monkeypatch.setattr(C, "DRAIN_REFUSAL_REPLAN_DELAY", 0)
    plan_one = NS(feasible=True, steps=[NS(vm_id="v", src_host="a", dst_host="b")], unmovable=[],
                  infeasible_reasons=[], blocked=[])
    plan_done = NS(feasible=True, steps=[], unmovable=[], infeasible_reasons=[], blocked=[])
    plans = iter([plan_one, plan_one, plan_one, plan_done])
    monkeypatch.setattr(B, "evaluate_drain", lambda *a, **k: ({"hosts_in_agg": ["a", "b"]}, next(plans)))
    monkeypatch.setattr(B._planning_drain, "drain_plan_to_migration_plan", lambda p: p)
    for name in ("print_baseline_table", "print_drain_report", "append_migration_event"):
        monkeypatch.setattr(B._report, name, lambda *a, **k: None)
    monkeypatch.setattr(B._report, "build_drain_event", lambda *a, **k: {})
    monkeypatch.setattr(B._report, "format_drain_vm_label", lambda s, v: v)
    results = iter([None, True])
    monkeypatch.setattr(B._execution_migrate, "execute_migration_step", lambda *a, **k: next(results))
    rc = B.drain_host_workload(NS(), NS(), "agg", ["a", "b"], "a", assume_yes=True, disable_host=False)
    assert rc == C.DRAIN_EXIT_OK


@pytest.mark.parametrize("interval,expected", [("15s", 30.0), ("30s", 60.0), ("1m", 120.0)])
def test_cpu_compensation_window_follows_the_real_scrape_interval(monkeypatch, interval, expected):
    P._SCRAPE_INTERVAL_CACHE.clear()

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"activeTargets": [
                {"labels": {"job": "node"}, "scrapeInterval": interval},
                {"labels": {"job": "other"}, "scrapeInterval": "5m"}]}}
    seen = {}

    def get(url, params=None, timeout=None):
        seen["url"] = url
        return Resp()
    monkeypatch.setattr(P.requests, "get", get)
    cfg = NS(prometheus_query_url="http://prom:9090/api/v1/query", prometheus_node_job="node",
             compensation_cpu_window_seconds=None)
    assert S.cpu_compensation_window_seconds(cfg) == expected
    assert seen["url"] == "http://prom:9090/api/v1/targets"
    P._SCRAPE_INTERVAL_CACHE.clear()


def test_cpu_compensation_window_override_and_fallback(monkeypatch):
    P._SCRAPE_INTERVAL_CACHE.clear()
    assert S.cpu_compensation_window_seconds(NS(compensation_cpu_window_seconds=45.0)) == 45.0
    import requests

    def down(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(P.requests, "get", down)
    cfg = NS(prometheus_query_url="http://p/api/v1/query", prometheus_node_job="node",
             compensation_cpu_window_seconds=None)
    assert S.cpu_compensation_window_seconds(cfg) == C.COMPENSATION_CPU_WINDOW_SECONDS
    P._SCRAPE_INTERVAL_CACHE.clear()


def test_cpu_compensation_decays_over_the_given_window():
    cpu, ram = {"d": 20.0}, {"d": 40.0}
    S.apply_arrival_compensation(
        [{"vm_id": "a", "dst_host": "d", "age": 15.0}], cpu, ram, {"d": 32}, {"d": 100.0},
        {"a": "d"}, {"a": vm(8, 8, 50.0, 0.0)}, {}, 300.0, cpu_window=30.0)
    # 4 cores of 32 = 12.5%, half of the 30 s window left -> +6.25%
    assert cpu["d"] == pytest.approx(26.25)


def test_PL2_client_ignoring_the_locked_filter_does_not_block_unlocked_vms(tmp_path):
    # Lab: python-openstackclient 6.6.0 returned all 19 VMs for --locked.
    servers = [{"ID": "a", "Name": "a", "Status": "ACTIVE"}, {"ID": "b", "Name": "b", "Status": "ACTIVE"}]
    runner, state, calls = _stage(tmp_path, servers, [])
    runner.require_owned = lambda *a: None
    runner.disable(NS(name="h1"), state)
    assert calls  # disabled: nothing is really locked


def test_PL2_server_locked_parses_and_fails_closed():
    from compute_maintenance.adapters.compute_cli import Cloud
    from compute_maintenance.domain.maintenance_errors import StepFailed
    cloud = Cloud("openstack")
    for raw, expected in ((True, True), (False, False), ("True", True), ("false", False)):
        cloud.osc = lambda *a, raw=raw: {"locked": raw}
        assert cloud.server_locked("x") is expected
    cloud.osc = lambda *a: {}
    with pytest.raises(StepFailed):
        cloud.server_locked("x")


class TestMeasuredRamCompensation:
    """RAM compensation from the raw host sample (lab: mean error 0.10 pp)."""

    def run(self, ram_avg, instant, sample_age, arrivals, rss):
        now = 100_000.0
        cpu, ram = {"d": 20.0}, {"d": ram_avg}
        added = S.apply_arrival_compensation(
            [{"vm_id": v, "dst_host": "d", "age": age} for v, age in arrivals], cpu, ram, {"d": 64},
            {"d": 100.0}, {v: "d" for v, _ in arrivals}, {}, {}, 300.0, cpu_window=60.0, now=now,
            ram_instant={"d": instant}, ram_sample_ts={"d": now - sample_age}, vm_rss_gb=rss)
        return ram["d"], added["d"]

    def test_raw_sample_plus_rss_of_vms_that_arrived_after_it(self):
        # A (16 GiB) completed 60 s ago: already in the raw sample (20 s old).
        # B (8 GiB) completed 10 s ago: after the sample, so its RSS is added.
        ram, added = self.run(40.0, 48.0, 20.0, [("A", 60.0), ("B", 10.0)], {"A": 16.0, "B": 8.0})
        assert ram == pytest.approx(56.0)
        assert added["method"] == "measured+rss" and added["ram"] == pytest.approx(16.0)

    def test_fresh_raw_sample_is_the_answer(self):
        # Every arrival completed before the raw sample: the sample IS the
        # host's RAM (lab: worst +0.03 / -0.02 pp). A VM that left within the
        # averaging window no longer inflates the result (lab: was +2.55).
        ram, added = self.run(40.0, 37.0, 20.0, [("A", 120.0)], {"A": 16.0})
        assert ram == pytest.approx(37.0) and added["method"] == "measured"

    def test_stale_raw_sample_keeps_the_average_floor(self):
        ram, added = self.run(40.0, 37.0, 20.0, [("A", 5.0)], {"A": 16.0})
        assert ram == pytest.approx(40.0 + 16.0) and added["method"] == "measured+rss"

    def test_wait_for_a_raw_sample_newer_than_the_last_arrival(self, monkeypatch):
        clock = Clock(now=10_000.0)
        stamps = iter([9_990.0, 9_995.0, 10_002.0])
        monkeypatch.setattr(S._sources_prometheus, "get_scrape_interval_seconds", lambda *a: 30.0)
        monkeypatch.setattr(S._sources_prometheus, "get_metrics",
                            lambda url, q, hosts: {"d": {"value": next(stamps)}})
        cfg = NS(prometheus_query_url="u", prometheus_node_job="n", prometheus_query_mem_sample_timestamp="t")
        ok = S.wait_for_fresh_ram_samples(cfg, [{"vm_id": "a", "dst_host": "d", "age": 0.0}], ["d"],
                                          sleeper=clock.sleep, clock=clock)
        assert ok and clock() == 10_010.0  # two 5 s polls

    def test_wait_gives_up_after_three_scrapes(self, monkeypatch):
        clock = Clock(now=10_000.0)
        monkeypatch.setattr(S._sources_prometheus, "get_scrape_interval_seconds", lambda *a: 30.0)
        monkeypatch.setattr(S._sources_prometheus, "get_metrics", lambda url, q, hosts: {"d": {"value": 1.0}})
        cfg = NS(prometheus_query_url="u", prometheus_node_job="n", prometheus_query_mem_sample_timestamp="t")
        assert not S.wait_for_fresh_ram_samples(cfg, [{"vm_id": "a", "dst_host": "d", "age": 0.0}], ["d"],
                                                sleeper=clock.sleep, clock=clock)
        assert 90.0 <= clock() - 10_000.0 <= 100.0

    def test_host_rss_wins_over_guest_reported_usage(self):
        # Lab: lab-uyuni guest-reported 5.95 GiB, host RSS 31.77 GiB.
        cpu, ram = {"d": 0.0}, {"d": 10.0}
        S.apply_arrival_compensation(
            [{"vm_id": "u", "dst_host": "d", "age": 0.0}], cpu, ram, {"d": 64}, {"d": 100.0},
            {"u": "d"}, {"u": vm(8, 32, 0.0, 18.6)}, {}, 300.0, now=1000.0,
            vm_rss_gb={"u": 31.77})
        assert ram["d"] == pytest.approx(10.0 + 31.77)  # model fallback, full weight

    def test_measurement_failure_falls_back_to_the_model(self, monkeypatch):
        from compute_maintenance.domain.errors import PrometheusError

        def fail(*a, **k):
            raise PrometheusError("down")
        monkeypatch.setattr(S._sources_prometheus, "get_metrics", fail)
        cfg = NS(prometheus_query_url="u", prometheus_query_mem_used_instant="q",
                 prometheus_query_mem_sample_timestamp="t", prometheus_libvirt_job="lv")
        assert S.measure_receiving_hosts(cfg, [{"vm_id": "a", "dst_host": "d", "age": 1.0}], ["d"]) == ({}, {}, {})


class TestProgressAwareTimeout:
    """The 600 s limit only applies to migrations that stopped progressing."""

    @pytest.fixture
    def fast_clock(self, monkeypatch):
        clock = {"t": 0.0}

        def monotonic():
            clock["t"] += 1.0
            return clock["t"]
        monkeypatch.setattr(M.time, "monotonic", monotonic)
        monkeypatch.setattr(M.time, "sleep", lambda s: None)
        monkeypatch.setattr(C, "MIGRATION_HOST_ACTIVITY_INTERVAL", 0.0)
        return clock

    def nova(self, polls_until_done, remaining_changes=True, host_rows=None):
        state = {"n": 0}

        class Compute:
            def server_migrations(self, _vm):
                state["n"] += 1
                left = 1000 - (state["n"] if remaining_changes else 0)
                return [Row(id=7, status="running", memory_remaining_bytes=left, source_node="src",
                            dest_node="dst")]

            def get_server(self, _vm):
                host = "dst" if state["n"] >= polls_until_done else "src"
                return Row(status="ACTIVE", compute_host=host, task_state=None if host == "dst" else "migrating")

            def migrations(self, **query):
                if "host" in query and host_rows is not None:
                    return host_rows(state["n"])
                return []
        return NS(compute=Compute()), state

    def test_progressing_migration_outlives_the_minimum(self, fast_clock):
        conn, state = self.nova(polls_until_done=200)
        result = M.monitor_migration(conn, "vm", "vm", src_host="src", dst_host="dst", timeout=10,
                                     stall_timeout=30, max_wait=10_000, verbose=False,
                                     preexisting_migration_ids=set())
        assert result == C.MIGRATION_RESULT_SUCCESS and state["n"] >= 200

    def test_stalled_migration_times_out_after_the_stall_window(self, fast_clock):
        conn, _state = self.nova(polls_until_done=10_000, remaining_changes=False)
        result = M.monitor_migration(conn, "vm", "vm", src_host="src", dst_host="dst", timeout=10,
                                     stall_timeout=30, max_wait=10_000, verbose=False,
                                     preexisting_migration_ids=set())
        assert result == C.MIGRATION_RESULT_TIMEOUT
        assert fast_clock["t"] < 200  # gave up shortly after minimum + stall

    def test_queued_migration_waits_while_the_one_ahead_progresses(self, fast_clock):
        # Own row never changes (queued behind max_concurrent_live_migrations=1),
        # but the running migration on the same source host keeps updating.
        conn, _state = self.nova(
            polls_until_done=150, remaining_changes=False,
            host_rows=lambda n: [Row(id=6, status="running", updated_at=f"t{n}")])
        result = M.monitor_migration(conn, "vm", "vm", src_host="src", dst_host="dst", timeout=10,
                                     stall_timeout=30, max_wait=10_000, verbose=False,
                                     preexisting_migration_ids=set())
        assert result == C.MIGRATION_RESULT_SUCCESS

    def test_maximum_wait_is_absolute(self, fast_clock):
        conn, _state = self.nova(polls_until_done=10_000)
        result = M.monitor_migration(conn, "vm", "vm", src_host="src", dst_host="dst", timeout=10,
                                     stall_timeout=30, max_wait=100, verbose=False,
                                     preexisting_migration_ids=set())
        assert result == C.MIGRATION_RESULT_TIMEOUT and fast_clock["t"] < 400

    def test_journal_wait_extends_while_nova_rows_progress(self, tmp_path):
        clock = Clock()
        journal = R.MigrationJournal(tmp_path, settle_seconds=0, clock=clock, sleeper=clock.sleep,
                                     active_timeout=60, stall_timeout=30, max_wait=10_000)
        polls = {"n": 0}

        class Compute:
            def migrations(self, **query):
                polls["n"] += 1
                if polls["n"] < 100:  # ~1000 s of a progressing external migration
                    return [Row(id=1, status="running", updated_at=f"u{polls['n']}")]
                return [Row(id=1, status="completed", updated_at="1970-01-01T00:00:00")]
        journal.wait_settled(NS(compute=Compute()), hosts=["h"])
        assert polls["n"] >= 100

        stalled = R.MigrationJournal(tmp_path / "b", settle_seconds=0, clock=clock, sleeper=clock.sleep,
                                     active_timeout=60, stall_timeout=30, max_wait=10_000)
        with pytest.raises(MigrationExecutionError, match="stalled"):
            stalled.wait_settled(NS(compute=NS(migrations=lambda **q: [Row(id=1, status="running",
                                                                              updated_at="same")])),
                                 hosts=["h"])


def test_arrival_completion_time_is_absolute_not_a_stale_age():
    # Lab drain-only run: "age" computed at settle time, compensation applied
    # seconds later -> the VM looked newer than the raw sample (measured+rss).
    now = 100_000.0
    arrivals = R.recent_arrivals([(Row(id=1, status="completed", server_id="a", source_compute="s",
                                       dest_compute="d"), now - 40.0)], now, 300.0)
    assert arrivals[0]["done_at"] == now - 40.0
    cpu, ram = {"d": 20.0}, {"d": 40.0}
    added = S.apply_arrival_compensation(
        arrivals, cpu, ram, {"d": 64}, {"d": 100.0}, {"a": "d"}, {}, {}, 300.0,
        now=now + 25.0,  # applied 25 s later
        ram_instant={"d": 47.0}, ram_sample_ts={"d": now - 20.0}, vm_rss_gb={"a": 8.0})
    assert added["d"]["method"] == "measured" and ram["d"] == pytest.approx(47.0)


def test_cpu_of_a_vm_without_metrics_comes_from_before_its_migration(monkeypatch):
    # Lab: right after a move the new series has no irate() yet and the old
    # one carries QEMU's migration thread; the full-flavor fallback added
    # +7.9% CPU (4 vCPU on 48 cores) for VMs that were nearly idle.
    seen = {}

    def cpu_at(cfg, vm_id, at):
        seen[vm_id] = at
        return 12.5
    monkeypatch.setattr(S._sources_prometheus, "get_vm_cpu_pct_at", cpu_at)
    arrivals = [{"vm_id": "a", "dst_host": "d", "age": 5.0, "done_at": 995.0, "begin_at": 900.0},
                {"vm_id": "known", "dst_host": "d", "age": 5.0, "done_at": 995.0, "begin_at": 900.0}]
    pre = S.pre_migration_cpu(NS(), arrivals, {"known": vm()})
    assert pre == {"a": 12.5} and seen == {"a": 885.0}
    cpu, ram = {"d": 10.0}, {"d": 10.0}
    S.apply_arrival_compensation(arrivals[:1], cpu, ram, {"d": 48}, {"d": 100.0}, {"a": "d"}, {},
                                 {"a": NS(flavor={"vcpus": 4, "ram": 4096})}, 300.0,
                                 ambiguous_samples={"a": {"pre_cpu_pct": 12.5, "cpu_pct": 90.0}},
                                 cpu_window=60.0, now=1000.0)
    # 4 vCPU x 12.5% = 0.5 core of 48, (1 - 5/60) left
    assert cpu["d"] == pytest.approx(10.0 + 0.5 / 48 * 100 * (1 - 5 / 60))
