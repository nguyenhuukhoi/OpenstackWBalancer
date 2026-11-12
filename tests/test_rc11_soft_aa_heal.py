"""rc11: soft-anti-affinity heal (VMware DRS constraint-correction pattern).

Ladder: ram_critical > ram_hotspot > cpu_hotspot > soft_aa_heal > MAD. A broken
soft-anti-affinity group (two or more members on one host of the aggregate)
gets at most one heal move per aggregate per run, through the same filter
chain, revalidation, cooldowns and events as any other move. Unresolvable or
disabled heal is informational only. Tie-breaks: a near-best balancing move
that heals is preferred (hotspot/MAD); drain prefers the destination with fewer
soft-anti-affinity members when everything else is equal.
"""
import json
import logging
from types import SimpleNamespace as NS

import pytest
from click.testing import CliRunner

import compute_maintenance.application.balancing as BAL
import compute_maintenance.application.migrations as MIG
import compute_maintenance.config as C
import compute_maintenance.presentation.reports as REP
from compute_maintenance.domain import balance, drain, models, policies
from compute_maintenance.domain.errors import ConfigError
from compute_maintenance.domain.plan_context import PlanContext
from compute_maintenance.domain.snapshot import freeze_context_inputs

LIVE = ("RAM_HOTSPOT_THRESHOLD", "RAM_CRITICAL_THRESHOLD", "CPU_HOTSPOT_THRESHOLD",
        "AUTO_PRESSURE_THRESHOLD", "TARGET_AVG_MAD", "MIN_IMPROVEMENT_THRESHOLD",
        "PLACEMENT_PRESSURE_THRESHOLD", "DRAIN_MAX_HOST_RAM_PCT", "DRAIN_MAX_HOST_CPU_PCT",
        "DRAIN_ABSOLUTE_MAX_HOST_RAM_PCT",
        "HEAL_SOFT_ANTI_AFFINITY", "SOFT_AA_HEAL_MAD_TOLERANCE", "SOFT_AA_TIEBREAK_MARGIN")


@pytest.fixture(autouse=True)
def restore_live_settings(monkeypatch):
    # apply_*() mutates module globals; monkeypatch restores them.
    for name in LIVE:
        monkeypatch.setattr(C, name, getattr(C, name))
    for key, _a, _d in C.SOFT_AA_HEAL_KEYS:
        monkeypatch.delenv(key, raising=False)


def heal_off(monkeypatch):
    monkeypatch.setattr(C, "HEAL_SOFT_ANTI_AFFINITY", False)


# ------------------------------------------------------------------ helpers
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


def group(name, policy, members, rules=None):
    return {"id": name, "name": name, "policy": policy, "members": list(members), "rules": rules or {}}


def soft(name, members):
    return group(name, "soft-anti-affinity", members)


def index(*groups):
    out = {}
    for g in groups:
        for member in g["members"]:
            out.setdefault(member, []).append(g)
    return out


def ctx(hosts, cpu, ram, vms, vm_host, groups=(), place=None, drain_mode=False, host_az=None,
        ready=None, max_instances=None, cooldown=None, excluded=()):
    vm_to_groups = index(*groups)
    c = models.BalanceContext(
        effective_hosts=list(hosts), host_cpu_pct=dict(cpu), host_ram_pct=dict(ram),
        host_cores={h: 32 for h in hosts}, host_ram={h: 100.0 for h in hosts},
        host_free_cores={h: 0 for h in hosts}, host_free_ram={h: 0 for h in hosts},
        host_placement=place or {h: placement() for h in hosts}, vm_host_map=dict(vm_host),
        all_vm_details=vms, vm_to_groups={v: vm_to_groups.get(v, []) for v in vm_host},
        enforce_server_groups=True, dest_ready_hosts=set(ready if ready is not None else hosts),
        host_running_vms={h: sum(1 for x in vm_host.values() if x == h) for h in hosts},
        host_max_instances=dict(max_instances or {h: 0 for h in hosts}), excluded_vm_ids=set(excluded),
        reject_stats={}, dest_reject_stats={}, skip_stats={}, ignore_cooldown=drain_mode,
        vm_status_map={v: "ACTIVE" for v in vm_host},
        cooldown_remaining={v: (cooldown or {}).get(v, 0.0) for v in vm_host},
        host_az=host_az)
    return freeze_context_inputs(c)


def heal(c):
    base = balance.compute_baseline_metrics(dict(c.host_cpu_pct), dict(c.host_ram_pct))
    return balance.plan_soft_aa_heal(c, base["mode"], base["baseline_weighted_mad"],
                                     base["cpu_weight"], base["ram_weight"])


def healed(result):
    step = result["plan"].steps[0]
    return step.vm_id, step.src_host, step.dst_host


# ===================================================== A. violation counting
def test_violations_count_members_beyond_one_per_host_across_groups_and_hosts():
    big = soft("big", ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"])  # a6 not in snapshot
    pair = soft("pair", ["d1", "d2"])
    spread = soft("spread", ["b1", "b2"])
    hard = group("hard", "anti-affinity", ["c1", "c2"])
    aff = group("aff", "soft-affinity", ["e1", "e2"])
    vh = {"a1": "H1", "a2": "H1", "a3": "H1", "a4": "H2", "a5": "H2",
          "a7": "X", "a8": "X",                       # outside the aggregate: not counted
          "b1": "H1", "b2": "H3", "c1": "H1", "c2": "H1", "d1": "H3", "d2": "H3",
          "e1": "H2", "e2": "H2"}
    broken = policies.broken_soft_anti_affinity_groups(
        vh, index(big, pair, spread, hard, aff), ["H1", "H2", "H3"])
    assert [(g["name"], g["violations"], g["worst_count"]) for g in broken] == [
        ("big", 3, 3),    # H1: 3 members -> 2, H2: 2 -> 1; X ignored; a6 absent
        ("pair", 1, 2),
    ]
    assert broken[0]["members_by_host"] == {"H1": ["a1", "a2", "a3"], "H2": ["a4", "a5"]}


def test_violation_counting_works_on_the_frozen_planner_context():
    g = soft("db", ["m1", "m2", "m3"])
    c = ctx(["A", "B"], {"A": 30, "B": 30}, {"A": 30, "B": 30},
            {"m1": vm(), "m2": vm(), "m3": vm()}, {"m1": "A", "m2": "A", "m3": "A"}, [g])
    broken = policies.broken_soft_anti_affinity_groups(c.vm_host_map, c.vm_to_groups, c.effective_hosts)
    assert [(g["name"], g["violations"]) for g in broken] == [("db", 2)]


def test_group_with_most_crowded_host_is_ordered_first():
    g1, g2 = soft("two-hosts", ["a", "b", "c", "d"]), soft("three", ["x", "y", "z"])
    vh = {"a": "H1", "b": "H1", "c": "H2", "d": "H2", "x": "H3", "y": "H3", "z": "H3"}
    broken = policies.broken_soft_anti_affinity_groups(vh, index(g1, g2), ["H1", "H2", "H3"])
    assert [g["name"] for g in broken] == ["three", "two-hosts"]   # worst host 3 > 2


@pytest.mark.parametrize("src,dst,expected", [
    ("A", "C", 1),   # leaves a member behind, lands on an empty host
    ("A", "B", 0),   # leaves one behind, lands next to one
    ("B", "A", -1),  # lone member joins the crowded host
])
def test_move_reduction_follows_the_violation_formula(src, dst, expected):
    g = soft("db", ["m1", "m2", "m3"])
    vh = {"m1": "A", "m2": "A", "m3": "B"}
    mover = "m1" if src == "A" else "m3"
    assert policies.soft_anti_affinity_move_reduction(mover, src, dst, vh, index(g)) == expected


# ================================================ C. source / VM / destination
def crowded():
    # db: 3 members on A (16/8/4 GiB flavors), one on B; C and D are member-free.
    g = soft("db", ["d16", "d8", "d4", "db-b"])
    vms = {"d16": vm(ram_gb=16, ram_pct=10.0), "d8": vm(ram_gb=8, ram_pct=10.0),
           "d4": vm(ram_gb=4, ram_pct=10.0), "db-b": vm()}
    vh = {"d16": "A", "d8": "A", "d4": "A", "db-b": "B"}
    cpu = {"A": 40.0, "B": 30.0, "C": 30.0, "D": 30.0}
    ram = {"A": 60.0, "B": 40.0, "C": 30.0, "D": 50.0}
    return g, vms, vh, cpu, ram


def test_smallest_ram_member_on_the_most_crowded_host_moves_to_the_freest_member_free_host():
    g, vms, vh, cpu, ram = crowded()
    result = heal(ctx("ABCD", cpu, ram, vms, vh, [g]))
    assert healed(result) == ("d4", "A", "C")          # C: most headroom, no member
    plan = result["plan"]
    assert plan.plan_type == C.SOFT_AA_HEAL_PLAN_TYPE == "soft_aa_heal"
    assert plan.steps[0].reason == "soft_aa_heal"
    assert plan.soft_aa_heal["group_violations_before"] == 2
    assert plan.soft_aa_heal["group_violations_after"] == 1


@pytest.mark.parametrize("blocker", [
    {"vms_extra": {"d4": {"locked": True}}},
    {"cooldown": {"d4": 3600.0}},
    {"excluded": {"d4"}},
    {"vms_extra": {"d4": {"unsupported_resources": "resources:CUSTOM_X=1"}}},
])
def test_unmovable_smallest_member_is_skipped_for_the_next_smallest(blocker):
    g, vms, vh, cpu, ram = crowded()
    for vm_id, extra in blocker.get("vms_extra", {}).items():
        vms[vm_id] = dict(vms[vm_id], **extra)
    c = ctx("ABCD", cpu, ram, vms, vh, [g], cooldown=blocker.get("cooldown"),
            excluded=blocker.get("excluded", ()))
    assert healed(heal(c)) == ("d8", "A", "C")
    assert c.skip_stats == {}       # heal probing never pollutes the balancing statistics


def test_destination_with_any_member_is_never_chosen_even_with_more_headroom():
    g = soft("db", ["m1", "m2", "m3"])
    vms = {"m1": vm(), "m2": vm(), "m3": vm()}
    vh = {"m1": "A", "m2": "A", "m3": "B"}
    cpu = {"A": 40.0, "B": 5.0, "C": 35.0}
    ram = {"A": 60.0, "B": 5.0, "C": 55.0}      # B is far emptier but holds m3
    assert heal(ctx("ABC", cpu, ram, vms, vh, [g]))["plan"].steps[0].dst_host == "C"
    # Without a member-free host there is nothing to heal with.
    result = heal(ctx("AB", {"A": 40.0, "B": 5.0}, {"A": 60.0, "B": 5.0}, vms, vh, [g]))
    assert result["plan"] is None
    assert "every host of the aggregate already holds a member" in result["unresolvable"][0]["reason"]


def single_free_host(**kw):
    """A holds m1+m2 (crowded); B is the only member-free destination."""
    g = soft("db", ["m1", "m2"])
    groups = [g] + kw.pop("extra_groups", [])
    vms = {"m1": vm(**kw.pop("vm1", {})), "m2": vm(**kw.pop("vm2", {}))}
    vms.update(kw.pop("vms", {}))
    vh = {"m1": "A", "m2": "A"}
    vh.update(kw.pop("vh", {}))
    cpu = kw.pop("cpu", {"A": 50.0, "B": 30.0})
    ram = kw.pop("ram", {"A": 60.0, "B": 40.0})
    return ctx("AB", cpu, ram, vms, vh, groups, **kw)


def test_feasible_single_destination_control_case():
    assert healed(heal(single_free_host())) == ("m1", "A", "B")


@pytest.mark.parametrize("setup,reason", [
    ({"ram": {"A": 60.0, "B": 79.5}}, "would_create_new_hotspot"),        # B would cross 80 %
    ({"cpu": {"A": 50.0, "B": 69.9}, "vm1": {"vcpus": 8, "cpu_pct": 100.0},
      "vm2": {"vcpus": 8, "cpu_pct": 100.0}}, "would_create_new_hotspot"),
    ({"place": {"A": placement(), "B": placement(mem_used=97280 - 1024)}}, "placement_memory"),
    ({"host_az": {"A": "az1", "B": "az2"}}, "availability_zone"),
    ({"max_instances": {"A": 0, "B": 1}, "vh": {"other": "B"}, "vms": {"other": vm()}}, "max_instances"),
    ({"ready": {"A"}}, "destination_not_ready"),
    ({"extra_groups": [group("hard", "anti-affinity", ["m1", "m2", "h3"])],
      "vh": {"h3": "B"}, "vms": {"h3": vm()}}, "server_group"),
])
def test_heal_moves_go_through_the_entire_filter_chain(setup, reason):
    result = heal(single_free_host(**setup))
    assert result["plan"] is None
    assert result["unresolvable"] == []                       # a filter is not structural
    assert reason in result["deferred_groups"][0]["reason"]
    assert result["deferred_groups"][0]["blocking"][reason] >= 1


@pytest.mark.parametrize("baseline,tolerance,target,limit", [
    (0.0, 0.05, 3, 3.0),        # balanced resource: it must stay at or below the target
    (2.98, 0.05, 3, 3.0),       # just under the target: still the target (closed edge, not 3.03)
    (3.0, 0.05, 3, 3.0),        # exactly at the target: not "above", stays bounded by it
    (3.5, 0.05, 3, 3.55),       # already above the target: at most +0.05
    (0.0, 0.05, 5, 5.0),        # follows TARGET_AVG_MAD
    (4.0, 1.0, 3, 5.0),         # follows SOFT_AA_HEAL_MAD_TOLERANCE above the target
    (2.0, 30.0, 3, 3.0),        # the tolerance never applies below the target
])
def test_heal_mad_limit_formula(baseline, tolerance, target, limit):
    assert balance.soft_aa_heal_mad_limit(baseline, tolerance, target) == pytest.approx(limit)


def test_heal_mad_limit_reads_the_live_settings(monkeypatch):
    C.apply_thresholds(C.Thresholds(target_mad=4))
    monkeypatch.setattr(C, "SOFT_AA_HEAL_MAD_TOLERANCE", 0.5)
    assert balance.soft_aa_heal_mad_limit(0.0) == 4.0
    assert balance.soft_aa_heal_mad_limit(3.8) == 4.0
    assert balance.soft_aa_heal_mad_limit(4.2) == pytest.approx(4.7)


@pytest.mark.parametrize("before,after,expected", [
    ((0.0, 0.0), (1.67, 1.56), (True, None)),                                 # balanced: both <= 3
    ((0.0, 0.0), (0.5, 3.6), (False, "RAM MAD 3.600 above heal limit 3.000")),   # weighted 2.98 still refused
    ((0.0, 0.0), (3.2, 0.1), (False, "CPU MAD 3.200 above heal limit 3.000")),
    ((0.0, 2.98), (0.1, 3.00), (True, None)),                                # edge: 2.98 -> 3.00 allowed
    ((0.0, 2.98), (0.1, 3.02), (False, "RAM MAD 3.020 above heal limit 3.000")),  # edge: max() allowed 3.03
    ((3.5, 0.0), (3.54, 2.0), (True, None)),                                 # CPU above target: +0.05 max
    ((3.5, 0.0), (3.6, 2.0), (False, "CPU MAD 3.600 above heal limit 3.550")),
    ((0.0, 3.5), (2.0, 3.51), (True, None)),                                 # RAM above target: 3.51 ok
    ((0.0, 3.5), (2.0, 3.6), (False, "RAM MAD 3.600 above heal limit 3.550")),
])
def test_heal_mad_check_is_per_resource(before, after, expected):
    assert balance.soft_aa_heal_mad_check(*before, *after) == expected


@pytest.mark.parametrize("member_pct,allowed", [
    (1.0, True),     # RAM MAD 2.98 -> 2.99 <= 3
    (4.0, False),    # RAM MAD 2.98 -> 3.02 >  3 (a max(before + 0.05, 3) rule allowed up to 3.03)
])
def test_heal_never_pushes_a_resource_just_under_the_target_above_it(member_pct, allowed):
    cpu, ram = {"A": 30.0, "B": 30.0}, {"A": 40.0, "B": 45.96}          # RAM MAD 2.98
    member = {"vcpus": 1, "ram_gb": 1, "cpu_pct": member_pct, "ram_pct": member_pct}
    assert balance.compute_baseline_metrics(cpu, ram)["baseline_ram_mad"] == pytest.approx(2.98)
    result = heal(single_free_host(cpu=cpu, ram=ram, vm1=member, vm2=dict(member)))
    assert (result["plan"] is not None) is allowed
    if allowed:
        assert result["plan"].mad_ram <= 3.0
        assert result["plan"].soft_aa_heal["ram_mad_limit"] == 3.0


def test_heal_allowed_on_a_balanced_aggregate_while_both_resources_stay_below_target():
    # Balanced pair: CPU MAD 0 -> 1.25, RAM MAD 0 -> 2.00, both far above 0 + 0.05
    # but <= TARGET_AVG_MAD 3, so the aggregate is still not UNBALANCED.
    plan = heal(single_free_host(cpu={"A": 30.0, "B": 30.0}, ram={"A": 40.0, "B": 40.0}))["plan"]
    assert plan is not None
    assert (plan.mad_cpu, plan.mad_ram) == (pytest.approx(1.25), pytest.approx(2.0))
    assert (plan.soft_aa_heal["cpu_mad_limit"], plan.soft_aa_heal["ram_mad_limit"]) == (3.0, 3.0)


def test_heal_blocked_when_a_resource_exceeds_both_limits(monkeypatch):
    big = {"vcpus": 8, "ram_gb": 16, "cpu_pct": 100.0, "ram_pct": 100.0}   # 25 CPU / 16 RAM points
    setup = {"cpu": {"A": 30.0, "B": 30.0}, "ram": {"A": 40.0, "B": 40.0}, "vm1": big, "vm2": big}
    result = heal(single_free_host(**setup))                     # CPU MAD 0 -> 25 > 3
    assert result["plan"] is None
    assert "mad_tolerance" in result["deferred_groups"][0]["reason"]
    # A resource below the target is bounded by the target alone: a huge
    # tolerance does not let the heal make the aggregate UNBALANCED.
    monkeypatch.setattr(C, "SOFT_AA_HEAL_MAD_TOLERANCE", 30.0)
    assert heal(single_free_host(**setup))["plan"] is None


# 1 vCPU at 16 % = 0.5 CPU point, 2 GiB at 80 % = 1.6 RAM points on 32-core / 100 GiB hosts.
BLIND_SPOT = {"vcpus": 1, "ram_gb": 2, "cpu_pct": 16.0, "ram_pct": 80.0}


def test_weighted_mad_blind_spot_is_blocked_per_resource():
    # CPU 30 / 30, RAM 40 / 44: weights 0.2 / 0.8. The heal ends at CPU MAD 0.5 /
    # RAM MAD 3.6 = weighted 2.98 <= 3 (a weighted gate would allow it), but RAM
    # 3.6 > 3 (RAM MAD 2.0 is below the target, so its limit is the target)
    # would make the aggregate UNBALANCED -> refused.
    cpu, ram = {"A": 30.0, "B": 30.0}, {"A": 40.0, "B": 44.0}
    base = balance.compute_baseline_metrics(cpu, ram)
    assert (base["cpu_weight"], base["ram_weight"]) == (pytest.approx(0.2), pytest.approx(0.8))
    c = single_free_host(cpu=cpu, ram=ram, vm1=dict(BLIND_SPOT), vm2=dict(BLIND_SPOT))
    p = balance.project_runtime_move(cpu, ram, c.host_cores, c.host_ram, BLIND_SPOT, "A", "B")
    mad_cpu = balance.mean_absolute_deviation(list(p["new_host_cpu"].values()))
    mad_ram = balance.mean_absolute_deviation(list(p["new_host_ram"].values()))
    assert (mad_cpu, mad_ram) == (pytest.approx(0.5), pytest.approx(3.6))
    assert 0.2 * mad_cpu + 0.8 * mad_ram == pytest.approx(2.98)
    result = heal(c)
    assert result["plan"] is None
    assert result["deferred_groups"][0]["reason"] == "no member-free destination passes the filters (mad_tolerance=2)"


def test_weighted_mad_blind_spot_passing_counterpart():
    # Same aggregate, member with 0.4 GiB used: CPU MAD 0.5, RAM MAD 2.4 -> both <= 3.
    member = dict(BLIND_SPOT, ram_gb=1, ram_pct=40.0)
    plan = heal(single_free_host(cpu={"A": 30.0, "B": 30.0}, ram={"A": 40.0, "B": 44.0},
                                 vm1=member, vm2=dict(member)))["plan"]
    assert plan is not None
    assert (plan.mad_cpu, plan.mad_ram) == (pytest.approx(0.5), pytest.approx(2.4))


@pytest.mark.parametrize("resource,member_pct,allowed", [
    ("ram", 1.0, True),      # RAM MAD 3.500 -> 3.510 <= 3.55
    ("ram", 10.0, False),    # RAM MAD 3.500 -> 3.600 >  3.55 (weighted 2.80 -> 2.94 would pass)
    ("cpu", 1.0, True),      # CPU MAD 3.500 -> 3.531 <= 3.55
    ("cpu", 10.0, False),    # CPU MAD 3.500 -> 3.813 >  3.55
])
def test_resource_above_target_may_grow_only_by_the_tolerance(resource, member_pct, allowed):
    cpu = {"A": 30.0, "B": 37.0 if resource == "cpu" else 30.0}       # MAD 3.5 on the chosen resource
    ram = {"A": 40.0, "B": 47.0 if resource == "ram" else 40.0}
    member = {"vcpus": 1, "ram_gb": 1, "cpu_pct": member_pct, "ram_pct": member_pct}
    result = heal(single_free_host(cpu=cpu, ram=ram, vm1=member, vm2=dict(member)))
    assert (result["plan"] is not None) is allowed
    if allowed:
        plan = result["plan"]
        after = plan.mad_cpu if resource == "cpu" else plan.mad_ram
        assert 3.5 < after <= 3.55
        assert plan.soft_aa_heal[resource + "_mad_limit"] == pytest.approx(3.55)


def test_first_deferred_group_does_not_stop_the_next_one():
    stuck = soft("stuck", ["s1", "s2", "s3"])           # 3 on A, but every member locked
    other = soft("other", ["o1", "o2"])
    vms = {"s1": vm(locked=True), "s2": vm(locked=True), "s3": vm(locked=True), "o1": vm(), "o2": vm()}
    vh = {"s1": "A", "s2": "A", "s3": "A", "o1": "A", "o2": "A"}
    result = heal(ctx("ABC", {"A": 50.0, "B": 30.0, "C": 30.0}, {"A": 60.0, "B": 40.0, "C": 40.0},
                      vms, vh, [stuck, other]))
    assert healed(result)[0] in ("o1", "o2")
    assert result["unresolvable"] == []
    assert [(e["group_name"], e["reason"], e["blocking"]) for e in result["deferred_groups"]] == [
        ("stuck", "no movable member on the crowded host(s) (locked=3)", {"locked": 3})]


@pytest.mark.parametrize("blocker,blocking", [
    ({"cooldown": {"m1": 3600.0, "m2": 3600.0}}, {"cooldown": 2}),
    ({"vms": {"m1": vm(locked=True), "m2": vm(locked=True)}}, {"locked": 2}),
    ({"missing": True}, {"missing_vm_details": 2}),   # e.g. series on two hosts right after a manual move
])
def test_temporary_causes_are_deferred_not_unresolvable(blocker, blocking):
    g = soft("db", ["m1", "m2"])
    vms = blocker.get("vms") or {"m1": vm(), "m2": vm()}
    if blocker.get("missing"):
        vms = {}                                           # no planner details for either member
    c = ctx("AB", {"A": 50.0, "B": 30.0}, {"A": 60.0, "B": 40.0}, vms, {"m1": "A", "m2": "A"}, [g],
            cooldown=blocker.get("cooldown"))
    result = heal(c)
    assert result["plan"] is None and result["unresolvable"] == []
    assert result["deferred_groups"][0]["blocking"] == blocking
    assert result["deferred_groups"][0]["reason"].startswith("no movable member on the crowded host(s)")


def test_only_the_structural_case_is_unresolvable():
    g = soft("etcd", ["e1", "e2", "e3"])                   # 3 members, 2 hosts, both occupied
    c = ctx("AB", {"A": 50.0, "B": 30.0}, {"A": 60.0, "B": 40.0}, {"e1": vm(), "e2": vm(), "e3": vm()},
            {"e1": "A", "e2": "A", "e3": "B"}, [g])
    result = heal(c)
    assert result["deferred_groups"] == []
    assert [(e["group_name"], e["reason"]) for e in result["unresolvable"]] == [
        ("etcd", "every host of the aggregate already holds a member (2 host(s), 3 member(s))")]


# ================================================================ B. ladder
def test_hotspot_outranks_heal():
    c = single_free_host(ram={"A": 85.0, "B": 40.0})
    result = heal(c)
    assert result["plan"] is None and result["deferred"] == "hotspot"
    assert result["violations"] == 1


def test_heal_disabled_detects_but_never_plans(monkeypatch):
    heal_off(monkeypatch)
    result = heal(single_free_host())
    assert result["plan"] is None and result["deferred"] == "disabled"
    assert [g["name"] for g in result["broken_groups"]] == ["db"]


def test_no_server_groups_means_no_heal():
    c = single_free_host()
    object.__setattr__(c, "enforce_server_groups", False)
    assert heal(c)["broken_groups"] == []


def test_budget_exhaustion_is_reported_not_raised():
    c = single_free_host()
    c.search_budget = 0
    result = heal(c)
    assert result["plan"] is None
    assert result["deferred_groups"][0]["reason"] == "planner candidate budget exhausted"
    assert result["deferred_groups"][0]["blocking"] == {"budget_exhausted": 1}


# =============================================================== D. tie-break
def tiebreak_ctx(m1_cpu_pct, m1_ram_gb):
    """Two hosts. x (no group) has the best improvement 10.25; m1 heals db."""
    g = soft("db", ["m1", "m2"])
    vms = {"x": vm(vcpus=4, ram_gb=8, cpu_pct=100.0, ram_pct=100.0),     # 12.5 % CPU, 8 % RAM
           "m1": vm(vcpus=4, ram_gb=m1_ram_gb, cpu_pct=m1_cpu_pct, ram_pct=100.0),
           "m2": vm(vcpus=1, ram_gb=1, cpu_pct=10.0, ram_pct=100.0)}
    vh = {"x": "A", "m1": "A", "m2": "A"}
    return ctx("AB", {"A": 60.0, "B": 20.0}, {"A": 60.0, "B": 20.0}, vms, vh, [g])


def best_direct_move(c):
    base = balance.compute_baseline_metrics(dict(c.host_cpu_pct), dict(c.host_ram_pct))
    move, improvement, _scope, _src = balance.search_best_move_for_state(
        c, base["mode"], base["baseline_weighted_mad"], base["cpu_weight"], base["ram_weight"], 0.15)
    return move, improvement


def test_tiebreak_prefers_a_healing_move_within_the_margin():
    move, improvement = best_direct_move(tiebreak_ctx(80.0, 8))   # m1: 9.00 vs x: 10.25 (-12 %)
    assert move.vm_id == "m1"
    assert improvement == pytest.approx(9.0)


def test_tiebreak_does_not_reach_outside_the_margin():
    move, improvement = best_direct_move(tiebreak_ctx(40.0, 4))   # m1: 4.50 vs 10.25 (-56 %)
    assert move.vm_id == "x"
    assert improvement == pytest.approx(10.25)


def test_tiebreak_margin_is_configurable(monkeypatch):
    monkeypatch.setattr(C, "SOFT_AA_TIEBREAK_MARGIN", 0.10)   # 9.00 is 12 % below 10.25
    assert best_direct_move(tiebreak_ctx(80.0, 8))[0].vm_id == "x"


@pytest.mark.parametrize("x_pct,x_valid", [
    (10.0, True),    # x improves 0.21 >= 0.15; m1 (0.02) heals but is below the bar
    (5.0, False),    # x improves 0.10 < 0.15: nothing is valid, nothing is reordered
])
def test_tiebreak_never_admits_a_move_below_the_improvement_threshold(monkeypatch, x_pct, x_valid):
    monkeypatch.setattr(C, "SOFT_AA_TIEBREAK_MARGIN", 1.0)       # widest band
    g = soft("db", ["m1", "m2"])
    vms = {"x": vm(vcpus=1, ram_gb=1, cpu_pct=x_pct, ram_pct=x_pct),
           "m1": vm(vcpus=1, ram_gb=1, cpu_pct=1.0, ram_pct=1.0),
           "m2": vm(vcpus=8, ram_gb=64, cpu_pct=100.0, ram_pct=100.0)}   # far too big to move
    c = ctx("AB", {"A": 51.0, "B": 49.0}, {"A": 51.0, "B": 49.0}, vms, {"x": "A", "m1": "A", "m2": "A"}, [g])
    base = balance.compute_baseline_metrics(dict(c.host_cpu_pct), dict(c.host_ram_pct))
    move, improvement = balance.find_best_move(c, base["mode"], ["A"], base["baseline_weighted_mad"],
                                               base["cpu_weight"], base["ram_weight"], improvement_threshold=0.15)
    assert move.vm_id == "x"
    assert (improvement >= 0.15) is x_valid


def test_tiebreak_also_applies_in_hotspot_mode():
    c = tiebreak_ctx(80.0, 8)
    c = ctx("AB", {"A": 60.0, "B": 20.0}, {"A": 85.0, "B": 20.0}, dict(c.all_vm_details),
            dict(c.vm_host_map), [soft("db", ["m1", "m2"])])
    move, _ = best_direct_move(c)
    base = balance.compute_baseline_metrics(dict(c.host_cpu_pct), dict(c.host_ram_pct))
    assert base["mode"] == C.MODE_RAM_HOTSPOT
    assert move.vm_id == "m1"


def test_tiebreak_off_with_heal_disabled_keeps_rc10_choice(monkeypatch):
    heal_off(monkeypatch)
    move, improvement = best_direct_move(tiebreak_ctx(80.0, 8))
    assert move.vm_id == "x" and improvement == pytest.approx(10.25)


# ========================================================= E. drain tie-break
def drain_ctx(cpu_y=20.0):
    """v leaves D. X holds a (soft-anti-affinity with v) and b (soft-affinity
    with v): soft penalty X = 1 (a) and Y = 1 (no b) -> equal. X and Y are
    identical, so the packing score is equal too."""
    aa = soft("aa", ["v", "a"])
    sa = group("sa", "soft-affinity", ["v", "b"])
    vms = {"v": vm(), "a": vm(), "b": vm()}
    vh = {"v": "D", "a": "X", "b": "X"}
    return ctx(["D", "X", "Y"], {"D": 20.0, "X": 20.0, "Y": cpu_y}, {"D": 20.0, "X": 20.0, "Y": 20.0},
               vms, vh, [aa, sa], drain_mode=True)


def drain_destination(c):
    plan = drain.plan_host_drain(c, "agg", "D", 0.5, 0.5)
    assert plan.feasible
    return plan.steps[0].dst_host


def test_drain_tiebreak_prefers_fewer_soft_anti_affinity_members_on_equal_scores():
    assert drain_destination(drain_ctx()) == "Y"


def test_drain_tiebreak_does_not_override_a_better_packing_score():
    assert drain_destination(drain_ctx(cpu_y=40.0)) == "X"    # X keeps the remaining hosts balanced


def test_drain_tiebreak_off_with_heal_disabled(monkeypatch):
    heal_off(monkeypatch)
    assert drain_destination(drain_ctx()) == "X"              # rc10: first host on a full tie


def test_drain_soft_penalty_still_comes_first():
    # rc5 behaviour unchanged: a destination without a soft conflict always wins.
    g = soft("etcd", ["m1", "m2", "m3"])
    c = ctx(["A", "B", "C", "D"], {h: 20.0 for h in "ABCD"}, {h: 20.0 for h in "ABCD"},
            {"m1": vm(), "m2": vm(), "m3": vm()}, {"m1": "A", "m2": "A", "m3": "B"}, [g], drain_mode=True)
    plan = drain.plan_host_drain(c, "agg", "A", 0.5, 0.5)
    assert {s.dst_host for s in plan.steps} == {"C", "D"} and not plan.soft_policy_warnings


# ======================================== end to end: snapshot / monitor (G)
def host_resource(cores=32, ram_gb=256):
    record = policies.build_placement_inventory_record
    placement_ = {"resource_provider_uuid": "rp",
                  "VCPU": record({"total": cores, "reserved": 0, "allocation_ratio": 1.0}, 0, 0),
                  "MEMORY_MB": record({"total": ram_gb * 1024, "reserved": 0, "allocation_ratio": 1.0}, 0, 0)}
    return {"total_cores": cores, "free_cores": cores, "total_ram_gb": ram_gb, "free_ram_gb": ram_gb,
            "running_vms": 0, "placement": placement_}


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    """Real evaluate_aggregate_state over fake Prometheus/Nova/Placement with
    VMs and server groups (worked example of docs/algorithm.md section 5.4)."""
    import compute_maintenance.adapters.prometheus as P
    import compute_maintenance.application.snapshots as S
    from compute_maintenance.adapters.cooldown import CooldownStore
    env = NS(hosts=["A", "B", "C"], series={"CPU": {"A": 30.0, "B": 25.0, "C": 28.0},
                                             "MEM": {"A": 70.0, "B": 40.0, "C": 45.0}},
             vms={"db1": vm(vcpus=8, ram_gb=16, cpu_pct=50.0, ram_pct=90.0),
                  "db2": vm(vcpus=4, ram_gb=8, cpu_pct=20.0, ram_pct=75.0)},
             vm_host={"db1": "A", "db2": "A"}, groups=[soft("db", ["db1", "db2"])])

    def do_query(url, query, at=None):
        rows = [{"metric": {"alias": host}, "value": [1_700_000_000, str(value)]}
                for host, value in env.series[query].items()]
        return {"status": "success", "data": {"result": rows}}
    monkeypatch.setattr(P, "do_query", do_query)
    monkeypatch.setattr(S._sources_openstack, "is_nova_compute_service_ready", lambda conn, host: (True, ""))
    monkeypatch.setattr(S._sources_prometheus, "get_all_vm_metrics", lambda *a, **k: {})
    monkeypatch.setattr(S, "build_aggregate_vm_inventory", lambda *a, **k: (
        {v: v for v in env.vm_host}, {v: env.vms[v] for v in env.vm_host if v in env.vms}, dict(env.vm_host)))
    env.cfg = NS(prometheus_query_url="http://prom.test/api/v1/query", prometheus_query_cpu_used="CPU",
                 prometheus_query_mem_used="MEM", cooldown_store=CooldownStore(str(tmp_path / "cool.json")),
                 max_instances_per_host={}, max_instances_per_host_default=0,
                 placement_final_allocation_candidate_check=False, planner_candidate_budget=100000,
                 make_room_min_improvement=0.3, enable_make_room=False)

    def cache():
        return {"host_resources": {h: host_resource() for h in env.hosts}, "host_resource_errors": {},
                "host_az": None, "server_cache": {}, "server_groups": list(env.groups), "project_cache": {}}
    env.cache = cache

    def evaluate(**kw):
        return S.evaluate_aggregate_state(env.cfg, object(), "agg", list(env.hosts),
                                          enforce_server_groups=True, cluster_cache=cache(),
                                          exclude_unready_hosts=True, **kw)
    env.evaluate = evaluate

    def monitor():
        import compute_maintenance.application.monitoring as MON
        return MON.monitor_aggregate(env.cfg, object(), "agg", list(env.hosts), enforce_server_groups=True,
                                     cluster_cache=cache())
    env.monitor = monitor
    return env


def balanced_with_tiny_member(env):
    """Balanced aggregate; heal of an idle member stays within the 0.05 tolerance."""
    env.series = {"CPU": {"A": 30.0, "B": 30.0, "C": 30.0}, "MEM": {"A": 50.0, "B": 50.0, "C": 50.0}}
    env.vms = {"t1": vm(vcpus=1, ram_gb=1, cpu_pct=1.0, ram_pct=5.0), "t2": vm(vcpus=1, ram_gb=1)}
    env.vm_host = {"t1": "A", "t2": "A"}
    env.groups = [soft("tiny", ["t1", "t2"])]


def test_worked_example_heal_move_is_the_plan_and_raises_warning(cloud):
    state = cloud.evaluate()
    plan = state["best_plan"]
    assert state["has_valid_move"] and plan.plan_type == "soft_aa_heal"
    assert (plan.steps[0].vm_id, plan.steps[0].src_host, plan.steps[0].dst_host) == ("db2", "A", "B")
    assert plan.baseline_weighted_mad == pytest.approx(10.1333, abs=1e-4)
    assert plan.final_weighted_mad == pytest.approx(8.5722, abs=1e-4)
    assert state["detail_status"] == "SOFT_AA_HEAL"
    assert (state["severity"], state["exit_code"]) == ("WARNING", C.ICINGA_WARNING)
    assert state["load_status"]["detail_status"] == "UNBALANCED"     # the MAD rung's own verdict
    assert state["message"].startswith("WARNING - soft-anti-affinity spread broken")
    assert "heal db: db2 (A -> B), group violations 1 -> 0" in state["message"]


def test_monitor_warning_when_balanced_aggregate_has_a_feasible_heal(cloud):
    balanced_with_tiny_member(cloud)
    result = cloud.monitor()
    assert result["exit_code"] == C.ICINGA_WARNING and result["detail_status"] == "SOFT_AA_HEAL"
    assert "Load status: BALANCED (OK)" in result["message"]


def test_balanced_aggregate_heals_an_8_gib_member_docs_example(cloud):
    # docs/algorithm.md 8.4: RAM 50/50/50, db2 (8 GiB flavor, 6 GiB used) moves:
    # CPU MAD 0 -> 1.67 and RAM MAD 0 -> 1.56, both <= TARGET_AVG_MAD 3 -> allowed.
    cloud.series = {"CPU": {"A": 30.0, "B": 30.0, "C": 30.0}, "MEM": {"A": 50.0, "B": 50.0, "C": 50.0}}
    state = cloud.evaluate()
    plan = state["best_plan"]
    assert plan.plan_type == "soft_aa_heal" and plan.steps[0].vm_id == "db2"
    assert plan.mad_ram == pytest.approx(1.5625) and plan.final_weighted_mad == pytest.approx(1.6146, abs=1e-4)
    assert (state["severity"], state["load_status"]["detail_status"]) == ("WARNING", "BALANCED")


def test_heal_that_would_unbalance_the_aggregate_stays_blocked_docs_example(cloud):
    # Counterexample: two 32 GiB members using all of it -> weighted MAD 0 -> 8.33 > 3.
    cloud.series = {"CPU": {"A": 30.0, "B": 30.0, "C": 30.0}, "MEM": {"A": 50.0, "B": 50.0, "C": 50.0}}
    cloud.vms = {"db1": vm(vcpus=8, ram_gb=32, cpu_pct=50.0, ram_pct=100.0),
                 "db2": vm(vcpus=8, ram_gb=32, cpu_pct=50.0, ram_pct=100.0)}
    state = cloud.evaluate()
    assert state["best_plan"] is None and (state["severity"], state["exit_code"]) == ("OK", C.ICINGA_OK)
    assert "db (1 violation(s)): deferred: no member-free destination passes the filters (mad_tolerance=4)" \
        in state["message"]


def test_monitor_note_only_when_the_group_cannot_be_fixed(cloud):
    balanced_with_tiny_member(cloud)
    cloud.vms.update(t3=vm(), t4=vm())
    cloud.vm_host.update(t3="B", t4="C")
    cloud.groups = [soft("tiny", ["t1", "t2", "t3", "t4"])]        # 4 members, 3 hosts
    result = cloud.monitor()
    assert result["exit_code"] == C.ICINGA_OK and result["detail_status"] == "BALANCED"
    assert "Soft-anti-affinity: 1 broken group(s), 1 violation(s)" in result["message"]
    assert "tiny (1 violation(s)): unresolvable: every host of the aggregate already holds a member" \
        in result["message"]


def test_monitor_note_only_when_heal_is_disabled(cloud, monkeypatch):
    balanced_with_tiny_member(cloud)
    heal_off(monkeypatch)
    result = cloud.monitor()
    assert result["exit_code"] == C.ICINGA_OK and result["detail_status"] == "BALANCED"
    assert "heal disabled (HEAL_SOFT_ANTI_AFFINITY=false)" in result["message"]


def test_heal_warning_never_masks_a_critical_load_state(cloud, monkeypatch):
    C.apply_thresholds(C.Thresholds(pressure=50))                    # makes pressure mode reachable
    balanced_with_tiny_member(cloud)
    cloud.series = {"CPU": {"A": 20.0, "B": 20.0, "C": 20.0}, "MEM": {"A": 60.0, "B": 60.0, "C": 60.0}}
    state = cloud.evaluate()
    assert state["best_plan"].plan_type == "soft_aa_heal"            # the ladder still heals
    assert state["load_status"]["detail_status"] == "PRESSURE_NO_VALID_MOVE"
    assert (state["severity"], state["exit_code"]) == ("CRITICAL", C.ICINGA_CRITICAL)
    assert "Load status: PRESSURE_NO_VALID_MOVE (CRITICAL)" in state["message"]


def test_heal_falls_through_to_mad_when_infeasible(cloud):
    cloud.vms["db1"] = dict(cloud.vms["db1"], locked=True)
    cloud.vms["db2"] = dict(cloud.vms["db2"], locked=True)
    cloud.vms["web"] = vm(vcpus=8, ram_gb=32, cpu_pct=50.0, ram_pct=90.0)
    cloud.vm_host["web"] = "A"
    state = cloud.evaluate()
    assert state["best_plan"].plan_type == "direct"
    assert state["best_plan"].steps[0].vm_id == "web"
    assert state["detail_status"] == "UNBALANCED" and state["load_status"] is None
    assert state["soft_aa_heal"]["unresolvable"] == []
    assert state["soft_aa_heal"]["deferred_groups"][0]["reason"].startswith("no movable member")
    assert "db (1 violation(s)): deferred: no movable member on the crowded host(s) (locked=2)" in state["message"]


def test_hotspot_defers_heal_end_to_end(cloud):
    cloud.series["MEM"]["A"] = 85.0
    state = cloud.evaluate()
    assert state["mode"] == C.MODE_RAM_HOTSPOT
    assert state["best_plan"].plan_type == "direct"
    assert "heal deferred while a hotspot is handled" in state["message"]


def test_heal_disabled_restores_the_rc10_decision(cloud, monkeypatch):
    on = cloud.evaluate()
    heal_off(monkeypatch)
    off = cloud.evaluate()
    assert off["best_plan"].plan_type == "direct"
    assert off["best_plan"].steps[0].vm_id == "db1"                    # plain best MAD move
    assert off["best_improvement"] == pytest.approx(2.0)
    assert (off["detail_status"], off["severity"], off["exit_code"]) == ("UNBALANCED", "WARNING", 1)
    assert off["load_status"] is None
    assert on["best_plan"].plan_type == "soft_aa_heal"
    # Everything the MAD rung itself computed is identical with heal on or off.
    for key in ("reject_stats", "skip_stats", "best_move", "baseline_weighted_mad", "mode"):
        assert on[key] == off[key], key


def test_monitor_only_writes_no_events(cloud, monkeypatch):
    monkeypatch.setattr(REP, "append_migration_event", lambda e: pytest.fail("monitor wrote an event"))
    assert cloud.monitor()["exit_code"] == C.ICINGA_WARNING


# ======================================= H. execution path, events, logging
def test_dry_run_records_the_heal_plan_with_mode_and_logs_it(cloud, monkeypatch, caplog):
    events = []
    monkeypatch.setattr(REP, "append_migration_event", events.append)
    with caplog.at_level(logging.INFO):
        assert BAL.auto_balance_aggregate(cloud.cfg, object(), "agg", cloud.hosts, dry_run=True,
                                          cluster_cache=cloud.cache()) is False
    assert [e["result"] for e in events] == ["dry_run"]
    assert events[0]["mode"] == "soft_aa_heal" and events[0]["plan_type"] == "soft_aa_heal"
    assert events[0]["soft_aa_heal"]["group_name"] == "db"
    assert events[0]["vm_id"] == "db2" and events[0]["step_reason"] == "soft_aa_heal"
    assert "soft_aa_heal: group db (1 violation(s)) -> move db2 A -> B" in caplog.text


def test_unresolvable_group_is_logged_and_recorded(cloud, monkeypatch, caplog):
    balanced_with_tiny_member(cloud)
    cloud.vms.update(t3=vm(), t4=vm())
    cloud.vm_host.update(t3="B", t4="C")
    cloud.groups = [soft("tiny", ["t1", "t2", "t3", "t4"])]
    events = []
    monkeypatch.setattr(REP, "append_migration_event", events.append)
    with caplog.at_level(logging.INFO):
        BAL.auto_balance_aggregate(cloud.cfg, object(), "agg", cloud.hosts, dry_run=True,
                                   cluster_cache=cloud.cache())
    assert [e["result"] for e in events] == ["soft_aa_heal_unresolvable"]
    assert events[0]["mode"] == "soft_aa_heal"
    assert events[0]["soft_aa_unresolvable"][0]["group_name"] == "tiny"
    assert "tiny: unresolvable: every host" in events[0]["detail"]
    assert "soft_aa_heal: group tiny (1 violation(s)) unresolvable:" in caplog.text
    json.dumps(events[0])   # JSONL-serialisable


def test_deferred_group_is_logged_and_recorded_with_its_blocking_counts(cloud, monkeypatch, caplog):
    # Both members locked: a temporary cause -> "deferred", never "unresolvable".
    cloud.vms["db1"] = dict(cloud.vms["db1"], locked=True)
    cloud.vms["db2"] = dict(cloud.vms["db2"], locked=True)
    events = []
    monkeypatch.setattr(REP, "append_migration_event", events.append)
    with caplog.at_level(logging.INFO):
        BAL.auto_balance_aggregate(cloud.cfg, object(), "agg", cloud.hosts, dry_run=True,
                                   cluster_cache=cloud.cache())
    deferred = [e for e in events if e["result"] == "soft_aa_heal_deferred"]
    assert len(deferred) == 1 and not [e for e in events if e["result"] == "soft_aa_heal_unresolvable"]
    assert deferred[0]["mode"] == "soft_aa_heal"
    assert deferred[0]["soft_aa_deferred"] == [{
        "group_id": "db", "group_name": "db", "violations": 1,
        "reason": "no movable member on the crowded host(s) (locked=2)", "blocking": {"locked": 2}}]
    assert deferred[0]["detail"] == "db: deferred: no movable member on the crowded host(s) (locked=2)"
    assert "soft_aa_heal: group db (1 violation(s)) deferred: no movable member" in caplog.text
    assert "unresolvable" not in caplog.text
    json.dumps(deferred[0])


# --- rc11 lab fix: migration_success / drain events carry the EXECUTED plan's labels ---
@pytest.fixture
def execute(cloud, monkeypatch):
    """Run the real execute_migration_step with a journal, so the pre-submit
    refresh is the real evaluate_aggregate_state(plan_balance=False)."""
    import compute_maintenance.application.snapshots as S
    events = []
    monkeypatch.setattr(MIG._report, "append_migration_event", events.append)
    monkeypatch.setattr(MIG, "validate_migration_step", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(MIG, "get_server_migration_ids", lambda conn, vm_id: set())
    monkeypatch.setattr(MIG, "snapshot_action_ids", lambda conn, vm_id: set())
    monkeypatch.setattr(MIG._sources_openstack, "server_migration_rows", lambda conn, vm_id: [])
    monkeypatch.setattr(MIG._sources_openstack, "max_migration_id", lambda rows: 0)
    monkeypatch.setattr(MIG, "monitor_migration", lambda *a, **k: C.MIGRATION_RESULT_SUCCESS)
    real_evaluate = S.evaluate_aggregate_state
    refreshed = []

    def refresh(cfg, conn, agg, hosts, **kw):
        if "plan_balance" not in kw:                    # the planning call itself
            return real_evaluate(cfg, conn, agg, hosts, **kw)
        state = real_evaluate(cloud.cfg, conn, agg, hosts, enforce_server_groups=True,
                              cluster_cache=cloud.cache(), exclude_unready_hosts=kw["exclude_unready_hosts"],
                              plan_balance=kw["plan_balance"])
        refreshed.append(dict(state))
        return state
    monkeypatch.setattr(MIG._snapshot, "evaluate_aggregate_state", refresh)
    journal = NS(begin=lambda *a, **k: None, finish=lambda *a, **k: None, state=NS(data={}))
    cfg = NS(**vars(cloud.cfg), migration_journal=journal)
    conn = NS(compute=NS(live_migrate_server=lambda **kw: None))

    def run(state):
        plan = state["best_plan"]
        assert MIG.execute_migration_step(cfg, conn, state, plan.steps[0], plan, 1) is True
        success = [e for e in events if e["result"] == "migration_success"]
        assert len(success) == 1
        return success[0], refreshed[-1]
    return run


def test_heal_migration_success_event_keeps_the_planned_labels(cloud, execute):
    planned = cloud.evaluate()
    dry = REP.build_migration_event(planned, "dry_run", plan=planned["best_plan"])
    event, refreshed = execute(planned)
    # The lab bug: the no-plan refresh classifies itself "no technical candidate".
    assert (refreshed["severity"], refreshed["detail_status"]) == ("CRITICAL", "UNBALANCED_NO_VALID_MOVE")
    assert refreshed["best_improvement"] == 0.0
    # The event describes the executed heal plan, exactly like its dry_run event.
    assert event["mode"] == "soft_aa_heal" and event["plan_type"] == "soft_aa_heal"
    assert (event["severity"], event["detail_status"]) == ("WARNING", "SOFT_AA_HEAL")
    assert event["best_improvement"] == pytest.approx(planned["best_plan"].improvement)
    for key in ("mode", "severity", "detail_status", "best_improvement", "improvement_threshold"):
        assert event[key] == dry[key], key


def test_direct_migration_success_event_keeps_the_planned_labels(cloud, execute, monkeypatch):
    heal_off(monkeypatch)
    planned = cloud.evaluate()
    assert planned["best_plan"].plan_type == "direct"
    event, refreshed = execute(planned)
    assert refreshed["severity"] == "CRITICAL"
    assert (event["mode"], event["severity"], event["detail_status"]) == ("mad", "WARNING", "UNBALANCED")
    assert event["best_improvement"] == pytest.approx(2.0)
    assert "soft_aa_heal" not in event


def test_drain_events_carry_drain_labels_not_a_balance_verdict(cloud, monkeypatch, capsys):
    events = []
    monkeypatch.setattr(REP, "append_migration_event", events.append)
    code = BAL.drain_host_workload(cloud.cfg, object(), "agg", cloud.hosts, "A", dry_run=True,
                                   cluster_cache=cloud.cache())
    assert code == C.DRAIN_EXIT_OK
    assert [e["result"] for e in events] == ["drain_dry_run"]
    event = events[0]
    assert (event["mode"], event["detail_status"], event["severity"]) == ("drain", "DRAIN_FEASIBLE", None)
    assert event["best_improvement"] is None and event["improvement_threshold"] is None
    assert event["drain_feasible"] is True


def test_heal_disabled_writes_no_heal_event(cloud, monkeypatch):
    balanced_with_tiny_member(cloud)
    heal_off(monkeypatch)
    events = []
    monkeypatch.setattr(REP, "append_migration_event", events.append)
    BAL.auto_balance_aggregate(cloud.cfg, object(), "agg", cloud.hosts, dry_run=True,
                               cluster_cache=cloud.cache())
    assert events == []


def test_heal_plan_executes_through_the_direct_path_without_the_improvement_gate(monkeypatch):
    step = NS(vm_id="db2", src_host="A", dst_host="B")
    plan = NS(plan_type="soft_aa_heal", steps=[step])
    state = PlanContext(has_valid_move=True, best_plan=plan, all_vm_names={}, exclude_unready_hosts=True)
    calls = []
    monkeypatch.setattr(BAL._snapshot, "evaluate_aggregate_state", lambda **kw: state)
    for name in ("print_baseline_table", "print_state_summary", "print_move_details"):
        monkeypatch.setattr(BAL._report, name, lambda *a, **k: None)
    monkeypatch.setattr(BAL._execution_migrate, "execute_migration_step",
                        lambda *a, **kw: calls.append((a[3], kw)) or True)
    assert BAL.auto_balance_aggregate(NS(), object(), "agg", ["A", "B"], assume_yes=True) is True
    assert len(calls) == 1
    assert calls[0][0] is step
    assert calls[0][1]["require_runtime_improvement"] is False
    assert calls[0][1]["prevent_new_hotspot"] is True


def test_executor_asks_for_heal_revalidation(monkeypatch, tmp_path):
    from compute_maintenance.adapters.cooldown import CooldownStore
    seen = []
    monkeypatch.setattr(MIG, "validate_migration_step", lambda *a, **kw: seen.append(kw) or (False, "stop"))
    monkeypatch.setattr(MIG._report, "append_migration_event", lambda e: None)
    cfg = NS(cooldown_store=CooldownStore(str(tmp_path / "cool.json")))
    step = NS(vm_id="vm", src_host="A", dst_host="B", vcpus=1, vm_ram_gb=1, reason="soft_aa_heal")
    for plan_type, expected in (("soft_aa_heal", True), ("direct", False)):
        assert MIG.execute_migration_step(cfg, object(), PlanContext(all_vm_details={}), step,
                                          NS(plan_type=plan_type, steps=[step], improvement=0.0), 1) is None
        assert seen[-1]["soft_aa_heal"] is expected


@pytest.fixture
def revalidate(monkeypatch):
    """validate_migration_step for a heal of m1 A -> B against fake fresh Nova."""
    fresh = {"vm_hosts": {"m1": "A", "m2": "A"}}
    g = soft("db", ["m1", "m2"])
    monkeypatch.setattr(MIG._sources_openstack, "build_nova_host_to_hypervisor_map",
                        lambda conn, hosts: {h: h for h in hosts})
    monkeypatch.setattr(MIG._sources_openstack, "is_nova_compute_service_ready", lambda conn, host: (True, ""))
    monkeypatch.setattr(MIG._sources_openstack, "fetch_placement_host_resource", lambda *a, **k: placement())
    monkeypatch.setattr(MIG._sources_openstack, "get_fresh_server_group_context",
                        lambda conn, vm_id, known_hosts=None, **_: (dict(fresh["vm_hosts"]), {"m1": [g]}))
    server = {"status": "ACTIVE", "compute_host": "A", "task_state": None}
    conn = NS(compute=NS(get_server=lambda _id: server))
    step = NS(vm_id="m1", src_host="A", dst_host="B", vcpus=2, vm_ram_gb=4)

    def run(cpu=None, ram=None, baseline=None, details=None):
        cpu = cpu or {"A": 50.0, "B": 30.0}
        ram = ram or {"A": 60.0, "B": 40.0}
        base = balance.compute_baseline_metrics(cpu, ram)
        state = PlanContext(all_vm_details={"m1": details or vm()}, host_az=None, host_max_instances={},
                            host_running_vms={}, host_cpu_pct=cpu, host_ram_pct=ram,
                            host_cores={"A": 32, "B": 32}, host_ram={"A": 100.0, "B": 100.0},
                            mode=base["mode"], max_cpu=base["max_cpu"], max_ram=base["max_ram"],
                            cpu_weight=base["cpu_weight"], ram_weight=base["ram_weight"],
                            # evaluate_aggregate_state records the per-resource baseline the gate uses
                            baseline_cpu_mad=base["baseline_cpu_mad"], baseline_ram_mad=base["baseline_ram_mad"],
                            baseline_weighted_mad=base["baseline_weighted_mad"] if baseline is None else baseline)
        return MIG.validate_migration_step(NS(placement_final_allocation_candidate_check=False), conn, state,
                                           step, enforce_server_groups=True, require_runtime_improvement=False,
                                           prevent_new_hotspot=True, enforce_soft_groups=True, soft_aa_heal=True)
    return NS(run=run, fresh=fresh)


def test_revalidation_accepts_a_heal_that_still_heals(revalidate):
    assert revalidate.run() == (True, "ok")


def test_revalidation_refuses_a_heal_the_group_no_longer_needs(revalidate):
    revalidate.fresh["vm_hosts"]["m2"] = "C"           # m2 moved away meanwhile
    assert revalidate.run() == (False, "soft_aa_heal_no_longer_heals")


def test_revalidation_allows_a_balanced_aggregate_to_stay_below_target(revalidate):
    # Fresh CPU MAD 0 -> 1.25, RAM MAD 0 -> 2.00: both <= TARGET_AVG_MAD 3 -> same verdict as the planner.
    assert revalidate.run(cpu={"A": 30.0, "B": 30.0}, ram={"A": 40.0, "B": 40.0}) == (True, "ok")


def test_revalidation_refuses_a_resource_above_its_limit(revalidate):
    big = vm(vcpus=8, ram_gb=16, cpu_pct=100.0, ram_pct=100.0)
    ok, why = revalidate.run(cpu={"A": 30.0, "B": 30.0}, ram={"A": 40.0, "B": 40.0}, details=big)
    assert ok is False and why == "refreshed CPU MAD 25.000 above heal limit 3.000"


def test_revalidation_blocks_the_weighted_blind_spot(revalidate):
    # Weights 0.2 / 0.8: CPU MAD 0.5 / RAM MAD 3.6 = weighted 2.98 <= 3, but RAM 3.6 > 3.
    ok, why = revalidate.run(cpu={"A": 30.0, "B": 30.0}, ram={"A": 40.0, "B": 44.0}, details=dict(BLIND_SPOT))
    assert ok is False and why == "refreshed RAM MAD 3.600 above heal limit 3.000"
    passing = dict(BLIND_SPOT, ram_gb=1, ram_pct=40.0)                       # RAM MAD 2.4
    assert revalidate.run(cpu={"A": 30.0, "B": 30.0}, ram={"A": 40.0, "B": 44.0}, details=passing) == (True, "ok")


@pytest.mark.parametrize("member_pct,expected_ok", [(1.0, True), (10.0, False)])
def test_revalidation_above_target_allows_only_the_tolerance(revalidate, member_pct, expected_ok):
    member = vm(vcpus=1, ram_gb=1, cpu_pct=member_pct, ram_pct=member_pct)
    ok, why = revalidate.run(cpu={"A": 30.0, "B": 30.0}, ram={"A": 40.0, "B": 47.0}, details=member)
    assert ok is expected_ok
    if not ok:
        assert why == "refreshed RAM MAD 3.600 above heal limit 3.550"


@pytest.mark.parametrize("member_pct,expected", [
    (1.0, (True, "ok")),                                                       # RAM MAD 2.98 -> 2.99
    (4.0, (False, "refreshed RAM MAD 3.020 above heal limit 3.000")),          # 2.98 -> 3.02
])
def test_revalidation_never_pushes_a_resource_just_under_the_target_above_it(revalidate, member_pct, expected):
    member = vm(vcpus=1, ram_gb=1, cpu_pct=member_pct, ram_pct=member_pct)
    assert revalidate.run(cpu={"A": 30.0, "B": 30.0}, ram={"A": 40.0, "B": 45.96}, details=member) == expected


@pytest.mark.parametrize("cpu_b,ram_b,details", [
    (30.0, 40.0, None),
    (30.0, 45.96, vm(vcpus=1, ram_gb=1, cpu_pct=1.0, ram_pct=1.0)),       # closed edge: allowed
    (30.0, 45.96, vm(vcpus=1, ram_gb=1, cpu_pct=4.0, ram_pct=4.0)),       # closed edge: blocked
    (30.0, 40.0, vm(vcpus=8, ram_gb=16, cpu_pct=100.0, ram_pct=100.0)),
    (30.0, 44.0, vm(vcpus=1, ram_gb=2, cpu_pct=16.0, ram_pct=80.0)),      # weighted blind spot: blocked
    (30.0, 44.0, vm(vcpus=1, ram_gb=1, cpu_pct=16.0, ram_pct=40.0)),      # its passing counterpart
    (30.0, 47.0, vm(vcpus=1, ram_gb=1, cpu_pct=1.0, ram_pct=1.0)),
    (30.0, 47.0, vm(vcpus=1, ram_gb=1, cpu_pct=10.0, ram_pct=10.0)),
    (37.0, 40.0, vm(vcpus=1, ram_gb=1, cpu_pct=1.0, ram_pct=1.0)),
    (37.0, 40.0, vm(vcpus=1, ram_gb=1, cpu_pct=10.0, ram_pct=10.0)),
])
def test_planner_and_revalidation_share_one_gate(revalidate, cpu_b, ram_b, details):
    cpu, ram = {"A": 30.0, "B": cpu_b}, {"A": 40.0, "B": ram_b}
    member = details or vm()
    planned = heal(single_free_host(cpu=cpu, ram=ram, vm1=dict(member), vm2=dict(member)))["plan"] is not None
    assert revalidate.run(cpu=cpu, ram=ram, details=member)[0] is planned


def test_heal_events_carry_the_mode(cloud):
    state = cloud.evaluate()
    event = REP.build_migration_event(state, "migration_success", plan=state["best_plan"])
    assert event["mode"] == "soft_aa_heal"
    assert event["soft_aa_heal"]["group_violations_after"] == 0
    direct = REP.build_migration_event(state, "x", plan=NS(plan_type="direct", improvement=1.0, steps=[]))
    assert direct["mode"] == "mad"          # the planning snapshot's load mode


# ================================================================ F. config
def write_config(tmp_path, **extra):
    lines = ["PROMETHEUS_QUERY_URL=http://prom.test:9090/api/v1/query",
             "PROMETHEUS_NODE_JOB=node", "PROMETHEUS_LIBVIRT_JOB=libvirt"]
    lines += [f"{k}={v}" for k, v in extra.items()]
    path = tmp_path / "wbalancer.conf"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_config_defaults():
    assert C.SoftAntiAffinityHeal().as_dict() == {
        "HEAL_SOFT_ANTI_AFFINITY": True, "SOFT_AA_HEAL_MAD_TOLERANCE": 0.05, "SOFT_AA_TIEBREAK_MARGIN": 0.15}
    assert (C.HEAL_SOFT_ANTI_AFFINITY, C.SOFT_AA_HEAL_MAD_TOLERANCE, C.SOFT_AA_TIEBREAK_MARGIN) == (
        True, 0.05, 0.15)


def test_config_file_without_keys_keeps_defaults(tmp_path):
    cfg = C.LoadbalancerConfig.load_config(write_config(tmp_path))
    assert cfg.soft_aa_heal.as_dict() == C.SoftAntiAffinityHeal().as_dict()


def test_config_keys_are_read_and_environment_wins(tmp_path, monkeypatch):
    cfg = C.LoadbalancerConfig.load_config(write_config(
        tmp_path, HEAL_SOFT_ANTI_AFFINITY="false", SOFT_AA_HEAL_MAD_TOLERANCE=0.5, SOFT_AA_TIEBREAK_MARGIN=0))
    assert cfg.soft_aa_heal.as_dict() == {
        "HEAL_SOFT_ANTI_AFFINITY": False, "SOFT_AA_HEAL_MAD_TOLERANCE": 0.5, "SOFT_AA_TIEBREAK_MARGIN": 0.0}
    monkeypatch.setenv("SOFT_AA_TIEBREAK_MARGIN", "1")
    assert C.LoadbalancerConfig.load_config(write_config(tmp_path)).soft_aa_heal.tiebreak_margin == 1.0


@pytest.mark.parametrize("extra,message", [
    ({"SOFT_AA_HEAL_MAD_TOLERANCE": -0.01}, "SOFT_AA_HEAL_MAD_TOLERANCE must be a finite number >= 0"),
    ({"SOFT_AA_HEAL_MAD_TOLERANCE": "nan"}, "SOFT_AA_HEAL_MAD_TOLERANCE"),
    ({"SOFT_AA_HEAL_MAD_TOLERANCE": "lots"}, "SOFT_AA_HEAL_MAD_TOLERANCE must be a number"),
    ({"SOFT_AA_TIEBREAK_MARGIN": 1.5}, "SOFT_AA_TIEBREAK_MARGIN must be between 0 and 1"),
    ({"SOFT_AA_TIEBREAK_MARGIN": -0.1}, "SOFT_AA_TIEBREAK_MARGIN must be between 0 and 1"),
    ({"HEAL_SOFT_ANTI_AFFINITY": "maybe"}, "HEAL_SOFT_ANTI_AFFINITY must be a boolean"),
])
def test_invalid_heal_settings_are_refused(tmp_path, extra, message):
    with pytest.raises(ConfigError, match=message):
        C.LoadbalancerConfig.load_config(write_config(tmp_path, **extra))


def test_apply_makes_the_settings_live():
    C.apply_soft_aa_heal(C.SoftAntiAffinityHeal(enabled=False, mad_tolerance=0.2, tiebreak_margin=0.3))
    assert (C.HEAL_SOFT_ANTI_AFFINITY, C.SOFT_AA_HEAL_MAD_TOLERANCE, C.SOFT_AA_TIEBREAK_MARGIN) == (
        False, 0.2, 0.3)
    with pytest.raises(TypeError):
        C.SoftAntiAffinityHeal(tolerance=1)


def test_cli_applies_the_heal_settings(tmp_path, monkeypatch):
    import compute_maintenance.cli.balance as CLI
    seen = {}

    def fake_monitor(cfg, **kw):
        seen.update(enabled=C.HEAL_SOFT_ANTI_AFFINITY, tolerance=C.SOFT_AA_HEAL_MAD_TOLERANCE)
        return 0
    monkeypatch.setattr(CLI._execution_monitor, "monitor_by_aggregate", fake_monitor)
    conf = write_config(tmp_path, HEAL_SOFT_ANTI_AFFINITY="no", SOFT_AA_HEAL_MAD_TOLERANCE=0.25)
    result = CliRunner().invoke(CLI.main, ["--config", conf, "--monitor-only",
                                           "--cooldown-file", str(tmp_path / "cool.json")])
    assert result.exit_code == 0, result.output
    assert seen == {"enabled": False, "tolerance": 0.25}


def test_example_documents_the_heal_keys_with_their_defaults():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "wbalancer.conf.example").read_text(encoding="utf-8")
    for line in ("#HEAL_SOFT_ANTI_AFFINITY=true", "#SOFT_AA_HEAL_MAD_TOLERANCE=0.05",
                 "#SOFT_AA_TIEBREAK_MARGIN=0.15"):
        assert line in text.splitlines(), line
