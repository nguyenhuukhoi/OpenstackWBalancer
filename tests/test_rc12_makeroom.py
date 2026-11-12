"""rc12 MAKE_ROOM review (REVIEW-MAKEROOM-2026-11-11): the step-2 refresh plans
nothing (L1), and the two-step executor end to end (L3, review gap #6).

The executor tests run the real balancing.auto_balance_aggregate,
migrations.execute_migration_step / wait_for_make_room_step / record_rejected_move
and a real MigrationJournal on disk; only Nova/Prometheus (the snapshot), the
revalidation verdicts and the migration monitor are faked.
"""
import time
from types import SimpleNamespace as NS

import pytest

import compute_maintenance.application.balancing as BAL
import compute_maintenance.application.migrations as MIG
import compute_maintenance.config as C
import compute_maintenance.presentation.reports as REP
from compute_maintenance.adapters.cooldown import CooldownStore
from compute_maintenance.application.recovery import MigrationJournal
from compute_maintenance.domain.errors import MigrationExecutionError
from compute_maintenance.domain.models import MigrationPlan, MigrationStep
from compute_maintenance.domain.plan_context import PlanContext

# --- L1: the step-2 refresh is a snapshot only -------------------------------------------

def test_make_room_step_two_refresh_does_not_run_the_planner(monkeypatch):
    seen = []
    monkeypatch.setattr(MIG._snapshot, "build_cluster_cache", lambda *a, **k: {})
    monkeypatch.setattr(MIG._snapshot, "evaluate_aggregate_state",
                        lambda **kw: seen.append(kw) or PlanContext(aggregate="agg", error="stop"))
    cfg = NS(placement_sync_timeout=0, placement_sync_interval=1)
    _state, ok, why = MIG.wait_for_make_room_step(cfg, object(), "agg", ["A", "B", "C"], object(),
                                                  exclude_unready_hosts=True)
    assert (ok, why) == (False, "refresh evaluation failed: stop")
    assert len(seen) == 1
    assert seen[0].get("plan_balance") is False      # a full planner run here would be thrown away


# --- Apply mode never prefetches the cluster cache it would discard ----------------------

def _run_balance_by_aggregate(monkeypatch, cfg, migrated=False):
    builds = []
    seen_caches = []
    monkeypatch.setattr(BAL._sources_openstack, "get_openstack_connection", lambda *a, **k: object())
    monkeypatch.setattr(BAL._sources_openstack, "get_aggregate_map",
                        lambda conn, aggregate_names=None, show_available_on_missing=False: {"agg": ["h1", "h2"]})
    monkeypatch.setattr(BAL._sources_openstack, "build_selected_host_list", lambda agg_map: ["h1", "h2"])
    monkeypatch.setattr(BAL._snapshot, "build_cluster_cache",
                        lambda *a, **k: builds.append(a) or {"host_resources": {}})
    monkeypatch.setattr(BAL, "auto_balance_aggregate",
                        lambda **kw: seen_caches.append(kw["cluster_cache"]) or migrated)
    BAL.balance_by_aggregate(cfg, dry_run=False, assume_yes=True)
    return builds, seen_caches


def test_apply_mode_does_not_prefetch_the_cluster_cache(monkeypatch):
    """With a journal, evaluate_aggregate_state discards any pre-barrier cache,
    so building one up front (and again after a migration) is pure waste."""
    builds, seen = _run_balance_by_aggregate(monkeypatch, NS(migration_journal=object()), migrated=True)
    assert builds == []
    assert seen == [None]


def test_journalless_mode_still_prefetches_and_refreshes_the_cache(monkeypatch):
    builds, seen = _run_balance_by_aggregate(monkeypatch, NS(), migrated=True)
    assert len(builds) == 2              # prefetch + post-migration refresh
    assert seen[0] is not None           # the aggregate got the prefetched cache


# --- L3: the two-step executor ----------------------------------------------------------

def step(vm_id, src, dst, reason):
    return MigrationStep(vm_id=vm_id, src_host=src, dst_host=dst, reason=reason, vcpus=2, vm_ram_gb=8.0,
                         real_vm_cores=0.5, real_vm_ram=2.0, cpu_impact_src=1.5, cpu_impact_dst=1.5,
                         ram_impact_src=0.8, ram_impact_dst=0.8, dst_free_vcpu=60.0, dst_free_ram_gb=100.0)


class Crash(BaseException):
    """The coordinator dies (kill -9, reboot) between the two steps."""


@pytest.fixture
def make_room(monkeypatch, tmp_path):
    """One MAKE_ROOM plan: step 1 moves `room` B -> C, step 2 moves `big` A -> B."""
    first = step("room", "B", "C", "make_room_release")
    second = step("big", "A", "B", "make_room_balance")
    plan = MigrationPlan(plan_type="make_room", steps=[first, second], baseline_weighted_mad=10.0,
                         final_weighted_mad=6.0, improvement=4.0, mad_cpu=6.0, mad_ram=6.0,
                         final_host_cpu={}, final_host_ram={}, search_scope="MAKE_ROOM FULL")
    world = {"vm_host": {"room": "B", "big": "A"}, "submits": [], "verdicts": {}, "on_refresh": None}
    details = {vm: {"vcpus": 2, "cpu_pct": 25.0, "ram_gb": 8.0, "ram_pct": 25.0, "ram_rss_gb": None,
                    "status": "ACTIVE", "project_id": "p-1", "project_name": "demo"} for vm in world["vm_host"]}

    def evaluate(*args, **kwargs):
        planning = kwargs.get("plan_balance", True)
        if not planning and world["on_refresh"] is not None:
            world["on_refresh"]()
        ctx = PlanContext(
            aggregate="agg", hosts_in_agg=["A", "B", "C"], exclude_unready_hosts=True,
            collected_at=time.time(), vm_host_map=dict(world["vm_host"]),
            all_vm_names={vm: vm + "-name" for vm in world["vm_host"]}, all_vm_details=details,
            host_cpu_pct={"A": 60.0, "B": 10.0, "C": 30.0}, host_ram_pct={"A": 50.0, "B": 20.0, "C": 40.0},
            host_cores={"A": 32, "B": 32, "C": 32}, host_ram={"A": 256.0, "B": 256.0, "C": 256.0},
            mode=C.MODE_MAD, step_improvement_threshold=0.15,
            improvement_threshold=0.3 if planning else 0.15,
            detail_status="UNBALANCED" if planning else "UNBALANCED_NO_VALID_MOVE",
            severity="WARNING" if planning else "CRITICAL", best_improvement=4.0 if planning else 0.0,
            has_valid_move=planning, best_plan=plan if planning else None,
            best_plan_candidate=plan if planning else None, soft_aa_heal=None)
        return ctx

    def validate(cfg, conn, state, migration_step, **kwargs):
        # One scripted verdict per revalidation of each VM, in call order.
        return world["verdicts"][migration_step.vm_id].pop(0)

    events = []
    monkeypatch.setattr(BAL._snapshot, "evaluate_aggregate_state", evaluate)
    monkeypatch.setattr(BAL._snapshot, "build_cluster_cache", lambda *a, **k: {})
    for name in ("print_baseline_table", "print_state_summary", "print_move_details"):
        monkeypatch.setattr(REP, name, lambda *a, **k: None)
    monkeypatch.setattr(REP, "append_migration_event", events.append)
    monkeypatch.setattr(MIG, "validate_migration_step", validate)
    monkeypatch.setattr(MIG, "get_server_migration_ids", lambda conn, vm_id: set())
    monkeypatch.setattr(MIG, "snapshot_action_ids", lambda conn, vm_id: set())
    monkeypatch.setattr(MIG._sources_openstack, "server_migration_rows", lambda conn, vm_id: [])
    monkeypatch.setattr(MIG._sources_openstack, "max_migration_id", lambda rows: 0)
    monkeypatch.setattr(MIG, "monitor_migration", lambda *a, **k: C.MIGRATION_RESULT_SUCCESS)

    class Compute:
        def live_migrate_server(self, server, host, block_migration):
            world["submits"].append((server, host))
            world["vm_host"][server] = host            # Nova moves the VM

    journal_dir = str(tmp_path / "operations")
    journal = MigrationJournal(journal_dir, 360)
    begun = []
    real_begin = journal.begin

    def recording_begin(migration_step, *args, **kwargs):
        begun.append(migration_step.vm_id)
        return real_begin(migration_step, *args, **kwargs)
    journal.begin = recording_begin
    cooldown = tmp_path / "cool.json"
    cfg = NS(cooldown_store=CooldownStore(str(cooldown)), migration_journal=journal,
             placement_sync_timeout=0, placement_sync_interval=1)

    def run():
        return BAL.auto_balance_aggregate(cfg, NS(compute=Compute()), "agg", ["A", "B", "C"], assume_yes=True)
    return NS(run=run, world=world, events=events, journal=journal, journal_dir=journal_dir, begun=begun,
              cooldown=cooldown, plan=plan)


def test_make_room_step_two_unsafe_after_refresh_is_recorded_once_and_raises(make_room):
    make_room.world["verdicts"] = {"room": [(True, "ok")], "big": [(False, "server_group")]}
    with pytest.raises(MigrationExecutionError, match="step 1 succeeded but step 2 was aborted: server_group"):
        make_room.run()
    assert make_room.world["submits"] == [("room", "C")]            # step 2 never reached Nova
    aborted = [e for e in make_room.events if e["result"] == "make_room_aborted_after_refresh"]
    assert len(aborted) == 1
    assert (aborted[0]["step_index"], aborted[0]["vm_id"], aborted[0]["detail"]) == (2, "big", "server_group")
    assert aborted[0]["plan_type"] == "make_room" and aborted[0]["detail_status"] == "UNBALANCED"
    assert [e["result"] for e in make_room.events] == ["migration_success", "make_room_aborted_after_refresh"]
    assert make_room.begun == ["room"] and make_room.journal.state.data["pending"] is None


def test_make_room_step_two_refused_before_submit_keeps_step_one_and_cools_the_vm(make_room):
    # The refresh loop accepts step 2; the executor's own pre-submit check refuses it.
    make_room.world["verdicts"] = {"room": [(True, "ok")], "big": [(True, "ok"), (False, "VM is locked")]}
    assert make_room.run() is True                                   # no exception: step 1 is a valid move
    assert make_room.world["submits"] == [("room", "C")]
    refused = make_room.events[-1]
    assert (refused["result"], refused["step_index"], refused["vm_id"], refused["detail"]) == (
        "step_revalidation_failed", 2, "big", "VM is locked")
    persisted = CooldownStore(str(make_room.cooldown))
    persisted.load()
    assert "big" in persisted.rejected and "room" not in persisted.rejected
    assert make_room.begun == ["room"] and make_room.journal.state.data["pending"] is None


def test_crash_between_the_steps_leaves_no_step_two_intent(make_room):
    make_room.world["verdicts"] = {"room": [(True, "ok")], "big": [(True, "ok"), (True, "ok")]}
    at_crash = {}

    def crash_once_step_one_is_done():
        if make_room.world["submits"]:                               # step 1 already migrated
            at_crash["pending"] = make_room.journal.state.data["pending"]
            raise Crash
    make_room.world["on_refresh"] = crash_once_step_one_is_done
    with pytest.raises(Crash):
        make_room.run()
    assert at_crash == {"pending": None}                             # step 1 intent closed before step 2
    assert make_room.world["submits"] == [("room", "C")]
    assert make_room.begun == ["room"]                               # step 2 was never journaled
    # The next run opens the journal: nothing pending, nothing to reconcile or resubmit.
    restarted = MigrationJournal(make_room.journal_dir, 360)
    assert restarted.state.data["pending"] is None
    assert restarted.state.data["last_outcome"] == "success"
    assert restarted.reconcile(object()) is None
