"""rc11 soft-anti-affinity heal: the ladder rung between the hotspots and MAD
balancing (detection, per-resource MAD gate, one heal move per run)."""
import compute_maintenance.config as _config
import compute_maintenance.domain.models as _models
import compute_maintenance.domain.policies as _planning_constraints
from compute_maintenance.domain.balance.filters import (
    check_direct_destination,
    make_step_from_projection,
    vm_candidate_details,
)
from compute_maintenance.domain.balance.modes import mean_absolute_deviation


def soft_aa_heal_vm_details(ctx, vm_id):
    """vm_candidate_details() for the heal search, without adding to the
    balancing skip statistics. Returns ``(details, skip_reason)``."""
    probe = {}
    saved = ctx.skip_stats
    ctx.skip_stats = probe
    try:
        details = vm_candidate_details(ctx, vm_id, record=True)
    finally:
        ctx.skip_stats = saved
    return details, next(iter(probe), None)


def soft_aa_heal_headroom(new_host_cpu, new_host_ram, host):
    """Percentage points ``host`` keeps below the nearer hotspot threshold."""
    return min(_config.runtime().cpu_hotspot_threshold - new_host_cpu[host],
               _config.runtime().ram_hotspot_threshold - new_host_ram[host])


def soft_aa_heal_mad_limit(baseline_mad, tolerance=None, target=None):
    """Highest MAD ONE resource (CPU or RAM) may reach after a heal move.

    A resource at or below TARGET_AVG_MAD must stay at or below it (limit =
    TARGET_AVG_MAD); only a resource already above it may grow, by at most
    SOFT_AA_HEAL_MAD_TOLERANCE (limit = baseline + tolerance). So a heal
    never pushes a resource's MAD above the target unless it already was.
    Examples (target 3, tolerance 0.05): baseline 0 -> 3.00; baseline 2.98
    -> 3.00 (not 3.03); baseline 3.50 -> 3.55.
    """
    tolerance = _config.runtime().soft_aa_heal_mad_tolerance if tolerance is None else tolerance
    target = float(_config.runtime().target_avg_mad if target is None else target)
    return baseline_mad + tolerance if baseline_mad > target else target


def soft_aa_heal_mad_check(cpu_mad_before, ram_mad_before, cpu_mad_after, ram_mad_after,
                           tolerance=None, target=None):
    """Per-resource MAD gate of a heal move, shared by planner and revalidation.

    Allowed iff, for CPU and for RAM separately, ``MAD after <=
    soft_aa_heal_mad_limit(MAD before)``. The MAD rung is triggered by
    ``cpu_mad > TARGET_AVG_MAD or ram_mad > TARGET_AVG_MAD`` (classify_cluster_
    detail_status), so a heal never makes an aggregate newly UNBALANCED.
    Returns ``(True, None)`` or ``(False, "RAM MAD 3.600 above heal limit 3.000")``.
    """
    for name, before, after in (("CPU", cpu_mad_before, cpu_mad_after), ("RAM", ram_mad_before, ram_mad_after)):
        limit = soft_aa_heal_mad_limit(before, tolerance, target)
        if after > limit + 1e-12:
            return False, f"{name} MAD {after:.3f} above heal limit {limit:.3f}"
    return True, None


# Why a broken group got no heal move this run.
SOFT_AA_UNRESOLVABLE = "unresolvable"   # structural: every host already holds a member


SOFT_AA_DEFERRED = "deferred"           # temporary: may clear by itself on a later run


def _format_counts(counts):
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def find_soft_aa_heal_move(ctx, group, baseline_weighted_mad, cpu_weight, ram_weight, mad_tolerance):
    """One move that removes a violation of the broken soft-anti-affinity ``group``.

    Source: the host holding the most members of the group (other crowded
    hosts are tried after it); on it the smallest-RAM (flavor) member that
    is movable - excluded, cooling-down, locked, busy, pinned/PCI/resources:*
    VMs are not. Destination: an aggregate host holding no member of the
    group that passes the whole direct-move filter chain
    (check_direct_destination), keeps the CPU MAD and the RAM MAD each within
    soft_aa_heal_mad_check() and lowers the VM's soft-anti-affinity
    violations in total. The destination with the most headroom below the
    hotspot thresholds wins (then the lower weighted MAD, then host order).
    Returns ``(MigrationPlan, None)`` or ``(None, outcome)`` with ``outcome``
    = ``{"status", "reason", "blocking"}``: status ``unresolvable`` only for
    the structural case (every host already holds a member), ``deferred``
    for causes that can clear by themselves (cooldown, lock, metrics still
    on two hosts after a move, destination filters, the MAD gate).
    """
    by_host = group["members_by_host"]
    free_hosts = [h for h in ctx.effective_hosts if h not in by_host]
    if not free_hosts:
        members = sum(len(v) for v in by_host.values())
        return None, {
            "status": SOFT_AA_UNRESOLVABLE,
            "reason": "every host of the aggregate already holds a member "
                      f"({len(ctx.effective_hosts)} host(s), {members} member(s))",
            "blocking": {},
        }

    sources = sorted((h for h, members in by_host.items() if len(members) >= 2),
                     key=lambda h: (-len(by_host[h]), h))
    base_cpu_mad = mean_absolute_deviation(list(ctx.host_cpu_pct.values()))
    base_ram_mad = mean_absolute_deviation(list(ctx.host_ram_pct.values()))
    skipped = {}
    rejected = {}
    for src_host in sources:
        movable = []
        for vm_id in by_host[src_host]:
            details, skip = soft_aa_heal_vm_details(ctx, vm_id)
            if details is None:
                key = skip or "not_movable"
                skipped[key] = skipped.get(key, 0) + 1
                continue
            ram_mb = round(float(details.get("ram_gb", 2.0) or 2.0) * 1024.0)
            movable.append((ram_mb, vm_id, details))

        for _ram_mb, vm_id, details in sorted(movable, key=lambda item: (item[0], item[1])):
            best = None
            for dst_host in free_hosts:
                ctx.consume_candidate()
                reason, projection, placement = check_direct_destination(
                    ctx, vm_id, details, src_host, dst_host
                )
                reduction = 0
                if reason is None:
                    reduction = _planning_constraints.soft_anti_affinity_move_reduction(
                        vm_id, src_host, dst_host, ctx.vm_host_map, ctx.vm_to_groups)
                    if reduction <= 0:
                        reason = "no_net_violation_reduction"
                if reason is None:
                    new_host_cpu = projection["new_host_cpu"]
                    new_host_ram = projection["new_host_ram"]
                    mad_cpu = mean_absolute_deviation(list(new_host_cpu.values()))
                    mad_ram = mean_absolute_deviation(list(new_host_ram.values()))
                    new_weighted_mad = (mad_cpu * cpu_weight) + (mad_ram * ram_weight)
                    # Heal must not buy redundancy with an unbalanced aggregate.
                    within, _why = soft_aa_heal_mad_check(base_cpu_mad, base_ram_mad, mad_cpu, mad_ram, mad_tolerance)
                    if not within:
                        reason = "mad_tolerance"
                if reason is not None:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                key = (soft_aa_heal_headroom(new_host_cpu, new_host_ram, dst_host), -new_weighted_mad)
                if best is None or key > best["key"]:
                    best = {
                        "key": key, "dst_host": dst_host, "projection": projection,
                        "placement": placement, "reduction": reduction,
                        "new_weighted_mad": new_weighted_mad, "mad_cpu": mad_cpu, "mad_ram": mad_ram,
                    }
            if best is None:
                continue
            projection = best["projection"]
            step = make_step_from_projection(
                vm_id, src_host, best["dst_host"], _config.MODE_SOFT_AA_HEAL,
                projection, best["placement"],
            )
            plan = _models.MigrationPlan(
                plan_type=_config.SOFT_AA_HEAL_PLAN_TYPE,
                steps=[step],
                baseline_weighted_mad=baseline_weighted_mad,
                final_weighted_mad=best["new_weighted_mad"],
                improvement=baseline_weighted_mad - best["new_weighted_mad"],
                mad_cpu=best["mad_cpu"],
                mad_ram=best["mad_ram"],
                final_host_cpu=projection["new_host_cpu"],
                final_host_ram=projection["new_host_ram"],
                search_scope=f"SOFT_AA_HEAL ({group['name'] or group['id']})",
                soft_aa_heal={
                    "group_id": group["id"],
                    "group_name": group["name"],
                    "group_violations_before": group["violations"],
                    "group_violations_after": group["violations"] - 1,
                    "violation_reduction": best["reduction"],
                    "members_on_source": len(by_host[src_host]),
                    "cpu_mad_before": base_cpu_mad,
                    "ram_mad_before": base_ram_mad,
                    "cpu_mad_limit": soft_aa_heal_mad_limit(base_cpu_mad, mad_tolerance),
                    "ram_mad_limit": soft_aa_heal_mad_limit(base_ram_mad, mad_tolerance),
                },
            )
            return plan, None

    if rejected:
        reason = "no member-free destination passes the filters (" + _format_counts(rejected) + ")"
        if skipped:
            reason += "; unmovable members (" + _format_counts(skipped) + ")"
    elif skipped:
        reason = "no movable member on the crowded host(s) (" + _format_counts(skipped) + ")"
    else:
        reason = "no heal candidate"
    blocking = dict(skipped)
    for key, count in rejected.items():
        blocking[key] = blocking.get(key, 0) + count
    return None, {"status": SOFT_AA_DEFERRED, "reason": reason, "blocking": blocking}


def plan_soft_aa_heal(ctx, mode, baseline_weighted_mad, cpu_weight, ram_weight):
    """rc11 ladder rung between the hotspots and MAD balancing.

    Detects broken soft-anti-affinity groups on the aggregate and, when
    HEAL_SOFT_ANTI_AFFINITY is on and no hotspot is being handled, plans at
    most one heal move (group with the most crowded host first). Returns a
    dict: ``enabled``, ``broken_groups``, ``violations``, ``plan`` (or None),
    ``group`` (the healed group), ``unresolvable`` (groups that can never be
    fixed in this aggregate: every host already holds a member),
    ``deferred_groups`` (groups not fixable this run for a temporary reason,
    with the blocking counts) and ``deferred`` (the whole rung: "disabled" /
    "hotspot" / None). It never raises on the candidate budget: heal is
    advisory, the budget answer stays with balance.
    """
    result = {
        "enabled": bool(_config.runtime().heal_soft_anti_affinity),
        "broken_groups": [],
        "violations": 0,
        "plan": None,
        "group": None,
        "unresolvable": [],
        "deferred_groups": [],
        "deferred": None,
    }
    if not ctx.enforce_server_groups:
        return result
    broken = _planning_constraints.broken_soft_anti_affinity_groups(
        ctx.vm_host_map, ctx.vm_to_groups, ctx.effective_hosts)
    result["broken_groups"] = broken
    result["violations"] = sum(g["violations"] for g in broken)
    if not broken:
        return result
    if not result["enabled"]:
        result["deferred"] = "disabled"
        return result
    if mode in _config.HOTSPOT_MODES:
        # Resource safety first: a hotspot outranks redundancy repair.
        result["deferred"] = "hotspot"
        return result

    from compute_maintenance.domain.errors import PlanningBudgetExceeded
    budget_exhausted = {"status": SOFT_AA_DEFERRED, "reason": "planner candidate budget exhausted",
                        "blocking": {"budget_exhausted": 1}}
    for group in broken:
        try:
            plan, outcome = find_soft_aa_heal_move(
                ctx, group, baseline_weighted_mad, cpu_weight, ram_weight,
                _config.runtime().soft_aa_heal_mad_tolerance)
        except PlanningBudgetExceeded:
            plan, outcome = None, budget_exhausted
        if plan is not None:
            result["plan"] = plan
            result["group"] = group
            break
        entry = {
            "group_id": group["id"],
            "group_name": group["name"],
            "violations": group["violations"],
            "reason": outcome["reason"],
        }
        if outcome["status"] == SOFT_AA_UNRESOLVABLE:
            result["unresolvable"].append(entry)
        else:
            entry["blocking"] = dict(outcome["blocking"])
            result["deferred_groups"].append(entry)
        if outcome is budget_exhausted:
            break
    return result
