"""planning.drain — extracted domain responsibilities."""
from copy import deepcopy

import compute_maintenance.config as _config
import compute_maintenance.domain.balance as _planning_balance
import compute_maintenance.domain.models as _models
import compute_maintenance.domain.policies as _planning_constraints


def build_drain_context(state, enforce_server_groups=True):
    """Rebuild a BalanceContext from an evaluated aggregate state (PlanContext).

    Drain deliberately clears ``excluded_vm_ids`` and sets ``ignore_cooldown``:
    a maintenance evacuation must be able to move every VM on the host, and
    both of those mechanisms exist only to damp automatic balancing.
    """
    from compute_maintenance.domain.snapshot import freeze_context_inputs
    return freeze_context_inputs(_models.BalanceContext(
        effective_hosts=list(state.hosts_in_agg),
        host_cpu_pct=dict(state.host_cpu_pct),
        host_ram_pct=dict(state.host_ram_pct),
        host_cores=dict(state.host_cores),
        host_ram=dict(state.host_ram),
        host_free_cores=dict(state.host_free_cores),
        host_free_ram=dict(state.host_free_ram),
        host_placement=state.host_placement,
        vm_host_map=dict(state.vm_host_map),
        all_vm_details=state.all_vm_details,
        vm_to_groups=state.vm_to_groups,
        enforce_server_groups=enforce_server_groups,
        dest_ready_hosts=set(state.dest_ready_hosts),
        host_running_vms=dict(state.host_running_vms),
        host_max_instances=dict(state.host_max_instances),
        excluded_vm_ids=set(),
        reject_stats={},
        dest_reject_stats={},
        skip_stats={},
        ignore_cooldown=True,
        vm_status_map=dict(state.vm_status_map or {}),
        search_budget=(state.planner_candidate_budget
                       if state.planner_candidate_budget is not None else 100000),
        host_az=state.host_az,
    ))


def classify_drain_vm(ctx, vm_id):
    """Return (details, blocking_reason) for a VM that must leave the host."""
    details = ctx.all_vm_details.get(vm_id)
    status = (getattr(ctx, "vm_status_map", None) or {}).get(vm_id)
    if status is None and details:
        status = details.get("status")
    if status != _config.ACTIVE_VM_STATUS:
        return None, "not_active:" + str(status or "UNKNOWN")
    if not details:
        return None, "missing_vm_details"
    if not details.get("metadata_complete", False):
        return None, "incomplete_metadata"
    if not details.get("metrics_complete", False):
        return None, "incomplete_metrics"
    # Locked VMs and VMs with a running Nova task cannot be live-migrated now.
    # Reporting them here keeps the dry-run honest instead of letting the real
    # drain stop half way at executor revalidation.
    blocked = _planning_balance.vm_planning_block_reason(details)
    if blocked:
        return None, blocked
    return details, None


def vm_flavor_request(details):
    """Placement request (vcpus, ram_mb) implied by a VM's flavor."""
    vcpus = int(details.get("vcpus", 1) or 1)
    ram_mb = round(float(details.get("ram_gb", 2.0) or 2.0) * 1024.0)
    return vcpus, ram_mb


def subset_weighted_mad(host_cpu_pct, host_ram_pct, hosts, cpu_weight, ram_weight):
    """Weighted MAD restricted to a host subset.

    A drained host no longer participates in the aggregate's balance, so the
    drain objective is measured over the *remaining* hosts only.
    """
    if not hosts:
        return 0.0
    cpu_mad = _planning_balance.mean_absolute_deviation([host_cpu_pct[h] for h in hosts])
    ram_mad = _planning_balance.mean_absolute_deviation([host_ram_pct[h] for h in hosts])
    return (cpu_mad * cpu_weight) + (ram_mad * ram_weight)


def drain_capacity_budget(ctx, movable, remaining_hosts):
    """Aggregate Placement arithmetic for the whole evacuation.

    This is the cheap, unambiguous check that answers "is there simply not
    enough resource left in the aggregate?" before the per-VM packing search
    runs. Packing can still fail on fragmentation when this check passes.
    """
    required_vcpu = 0
    required_ram_mb = 0
    for _vm_id, details in movable:
        vcpus, ram_mb = vm_flavor_request(details)
        required_vcpu += vcpus
        required_ram_mb += ram_mb

    available_vcpu = 0.0
    available_ram_mb = 0.0
    for host in remaining_hosts:
        placement = ctx.host_placement[host]
        available_vcpu += max(placement["VCPU"]["free"], 0.0)
        available_ram_mb += max(placement["MEMORY_MB"]["free"], 0.0)

    return {
        "required_vcpu": required_vcpu,
        "required_ram_mb": required_ram_mb,
        "available_vcpu": available_vcpu,
        "available_ram_mb": available_ram_mb,
        "vcpu_shortfall": max(required_vcpu - available_vcpu, 0.0),
        "ram_mb_shortfall": max(required_ram_mb - available_ram_mb, 0.0),
    }


def drain_host_caps(allow_hotspot):
    """Runtime (Prometheus) ceilings a receiving host must stay under."""
    if allow_hotspot:
        # --drain-allow-hotspot trades balance for the ability to finish the
        # evacuation, but never lifts the absolute RAM safety ceiling.
        return 100.0, _config.runtime().drain_absolute_max_host_ram_pct
    return _config.runtime().drain_max_host_cpu_pct, _config.runtime().drain_max_host_ram_pct


def pack_drain_vms(
    ctx,
    drain_host,
    remaining_hosts,
    ordered_movable,
    cpu_weight,
    ram_weight,
    allow_hotspot,
):
    """Greedy best-fit pass: place every VM, minimizing weighted MAD per step.

    The simulation advances Placement usage, runtime CPU/RAM percentages, the
    instance count per host and the VM->host topology after each placement, so
    later VMs are checked against the state the earlier ones would create.
    """
    cpu_cap, ram_cap = drain_host_caps(allow_hotspot)

    sim_cpu = dict(ctx.host_cpu_pct)
    sim_ram = dict(ctx.host_ram_pct)
    sim_placement = {h: deepcopy(ctx.host_placement[h]) for h in remaining_hosts}
    sim_counts = dict(ctx.host_running_vms or {})
    sim_vm_host = dict(ctx.vm_host_map)

    steps = []
    blocked = []
    soft_warnings = []
    # rc11: last tie-break key only (soft penalty and packing score equal).
    aa_tiebreak = bool(ctx.enforce_server_groups and _config.runtime().heal_soft_anti_affinity)

    for vm_id, details in ordered_movable:
        vcpus, ram_mb = vm_flavor_request(details)
        best = None
        reasons = {}

        for dst_host in remaining_hosts:
            ctx.consume_candidate()
            if ctx.dest_ready_hosts is not None and dst_host not in ctx.dest_ready_hosts:
                reasons[dst_host] = "destination_not_ready"
                continue

            if not _planning_balance.max_instances_allows(ctx, dst_host, current_counts=sim_counts):
                reasons[dst_host] = "max_instances"
                continue

            if ctx.enforce_server_groups and not _planning_constraints.is_server_group_move_compliant(
                vm_id, drain_host, dst_host, sim_vm_host, ctx.vm_to_groups
            ):
                reasons[dst_host] = "server_group"
                continue

            if not _planning_balance.availability_zone_allows(ctx, details, drain_host, dst_host):
                reasons[dst_host] = "availability_zone"
                continue

            fits, fit_reason = _planning_constraints.placement_can_fit(sim_placement[dst_host], vcpus, ram_mb)
            if not fits:
                reasons[dst_host] = fit_reason
                continue

            projection = _planning_balance.project_runtime_move(
                sim_cpu, sim_ram, ctx.host_cores, ctx.host_ram, details, drain_host, dst_host
            )
            if projection is None:
                reasons[dst_host] = "runtime_capacity_missing"
                continue

            new_host_cpu = projection["new_host_cpu"]
            new_host_ram = projection["new_host_ram"]
            if new_host_cpu[dst_host] > cpu_cap:
                reasons[dst_host] = f"cpu_cap_exceeded:{new_host_cpu[dst_host]:.1f}>{cpu_cap:.1f}"
                continue
            if new_host_ram[dst_host] > ram_cap:
                reasons[dst_host] = f"ram_cap_exceeded:{new_host_ram[dst_host]:.1f}>{ram_cap:.1f}"
                continue

            score = subset_weighted_mad(
                new_host_cpu, new_host_ram, remaining_hosts, cpu_weight, ram_weight
            )
            # Soft-(anti-)affinity first, balance second: a destination that
            # keeps the soft groups intact always wins over a better MAD.
            soft = (
                _planning_constraints.soft_group_penalty(vm_id, dst_host, sim_vm_host, ctx.vm_to_groups)
                if ctx.enforce_server_groups else 0
            )
            # Equal soft penalty and equal score: fewer members of the VM's
            # soft-anti-affinity group(s) on the destination wins (anti-affinity
            # before affinity). Never decides anything the first two keys decide.
            aa = (
                _planning_constraints.soft_anti_affinity_members_on(
                    vm_id, dst_host, sim_vm_host, ctx.vm_to_groups)
                if aa_tiebreak else 0
            )
            if best is None or (soft, score, aa) < (best["soft"], best["score"], best["aa"]):
                best = {
                    "dst_host": dst_host,
                    "score": score,
                    "soft": soft,
                    "aa": aa,
                    "projection": projection,
                    "placement": sim_placement[dst_host],
                    "vcpus": vcpus,
                    "ram_mb": ram_mb,
                }

        if best is None:
            blocked.append(
                {
                    "vm_id": vm_id,
                    "vcpus": vcpus,
                    "ram_gb": ram_mb / 1024.0,
                    "reasons": reasons,
                }
            )
            continue

        dst_host = best["dst_host"]
        projection = best["projection"]
        if best["soft"] > 0:
            soft_warnings.append({"vm_id": vm_id, "dst_host": dst_host, "penalty": best["soft"]})
        steps.append(
            _planning_balance.make_step_from_projection(
                vm_id, drain_host, dst_host, _config.DRAIN_STEP_REASON, projection, best["placement"]
            )
        )
        sim_cpu = projection["new_host_cpu"]
        sim_ram = projection["new_host_ram"]
        sim_placement[dst_host] = _planning_balance.placement_with_usage_delta(
            sim_placement[dst_host],
            {"VCPU": best["vcpus"], "MEMORY_MB": best["ram_mb"]},
        )
        sim_counts[dst_host] = sim_counts.get(dst_host, 0) + 1
        sim_counts[drain_host] = max(sim_counts.get(drain_host, 1) - 1, 0)
        sim_vm_host[vm_id] = dst_host

    final_weighted_mad = subset_weighted_mad(
        sim_cpu, sim_ram, remaining_hosts, cpu_weight, ram_weight
    )
    return {
        "steps": steps,
        "blocked": blocked,
        "soft_warnings": soft_warnings,
        "final_host_cpu": sim_cpu,
        "final_host_ram": sim_ram,
        "final_weighted_mad": final_weighted_mad,
    }


def drain_vm_orderings(movable):
    """Candidate placement orders for the greedy packer.

    Greedy best-fit is order-sensitive, so a few deterministic orders are tried
    and the best feasible one wins. This is still a heuristic, not an exact
    bin-packing solver: a reported infeasibility means "this planner found no
    safe packing", which is the fail-closed answer we want for maintenance.
    """
    def key_flavor_ram(item):
        vcpus, ram_mb = vm_flavor_request(item[1])
        return (-ram_mb, -vcpus, item[0])

    def key_flavor_vcpu(item):
        vcpus, ram_mb = vm_flavor_request(item[1])
        return (-vcpus, -ram_mb, item[0])

    def key_runtime_ram(item):
        details = item[1]
        rss = details.get("ram_rss_gb")
        if rss is not None:
            used = float(rss)
        else:
            used = (
                float(details.get("ram_gb", 0.0) or 0.0)
                * float(details.get("ram_pct", 0.0) or 0.0)
                / 100.0
            )
        return (-used, item[0])

    return [
        ("flavor_ram_desc", sorted(movable, key=key_flavor_ram)),
        ("flavor_vcpu_desc", sorted(movable, key=key_flavor_vcpu)),
        ("runtime_ram_desc", sorted(movable, key=key_runtime_ram)),
    ]


def drain_affinity_conflicts(ctx, drain_host):
    """Affinity groups with two or more members on the host being drained.

    Live migration moves one VM at a time, so the first member to leave would
    already split the group across hosts. The planner therefore cannot evacuate
    such a group without violating the affinity policy it is told to enforce.
    This is reported explicitly instead of being silently worked around.
    """
    conflicts = []
    if not ctx.enforce_server_groups:
        return conflicts

    seen_groups = set()
    for vm_id, host in ctx.vm_host_map.items():
        if host != drain_host:
            continue
        for group in ctx.vm_to_groups.get(vm_id) or []:
            if group.get("policy") != "affinity":
                continue
            group_id = group.get("id")
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            on_host = [
                member for member in (group.get("members") or [])
                if ctx.vm_host_map.get(member) == drain_host
            ]
            if len(on_host) >= 2:
                conflicts.append(
                    {
                        "group_id": group_id,
                        "group_name": group.get("name"),
                        "vm_ids": on_host,
                    }
                )
    return conflicts


def plan_host_drain(
    ctx,
    agg_name,
    drain_host,
    cpu_weight,
    ram_weight,
    allow_hotspot=False,
):
    """Plan a full evacuation of ``drain_host`` that keeps the rest balanced.

    Fail-closed by design: the returned plan is only ``feasible`` when every
    live-migratable VM on the host has a destination that satisfies Placement
    capacity, server groups, instance caps and the runtime hotspot ceilings.
    """
    infeasible_reasons = []
    remaining_hosts = [h for h in ctx.effective_hosts if h != drain_host]
    baseline_weighted_mad = subset_weighted_mad(
        ctx.host_cpu_pct, ctx.host_ram_pct, remaining_hosts, cpu_weight, ram_weight
    )

    movable = []
    unmovable = []
    for vm_id, host in ctx.vm_host_map.items():
        if host != drain_host:
            continue
        details, reason = classify_drain_vm(ctx, vm_id)
        if reason is not None:
            unmovable.append({"vm_id": vm_id, "reason": reason})
            continue
        movable.append((vm_id, details))

    affinity_conflicts = drain_affinity_conflicts(ctx, drain_host)
    shared = {
        "host": drain_host,
        "aggregate": agg_name,
        "remaining_hosts": remaining_hosts,
        "unmovable": unmovable,
        "affinity_conflicts": affinity_conflicts,
        "baseline_weighted_mad": baseline_weighted_mad,
    }

    if not remaining_hosts:
        infeasible_reasons.append(
            drain_host + " is the only host with usable metrics in " + agg_name
            + "; there is no destination to drain to"
        )
        return _models.DrainPlan(
            steps=[],
            feasible=False,
            blocked=[],
            capacity={},
            final_weighted_mad=baseline_weighted_mad,
            final_host_cpu=dict(ctx.host_cpu_pct),
            final_host_ram=dict(ctx.host_ram_pct),
            hotspot_hosts=[],
            infeasible_reasons=infeasible_reasons,
            **shared,
        )

    ready_hosts = [
        h for h in remaining_hosts
        if ctx.dest_ready_hosts is None or h in ctx.dest_ready_hosts
    ]
    if not ready_hosts:
        infeasible_reasons.append(
            "no destination-ready nova-compute host left in " + agg_name
            + " (candidates: " + ", ".join(remaining_hosts) + ")"
        )

    capacity = drain_capacity_budget(ctx, movable, ready_hosts)
    if capacity["vcpu_shortfall"] > 0:
        infeasible_reasons.append(
            "not enough VCPU: need {:.0f}, only {:.0f} safe-free on {} host(s) "
            "(short {:.0f})".format(
                capacity["required_vcpu"],
                capacity["available_vcpu"],
                len(ready_hosts),
                capacity["vcpu_shortfall"],
            )
        )
    if capacity["ram_mb_shortfall"] > 0:
        infeasible_reasons.append(
            "not enough RAM: need {:.1f} GiB, only {:.1f} GiB safe-free on {} host(s) "
            "(short {:.1f} GiB)".format(
                capacity["required_ram_mb"] / 1024.0,
                capacity["available_ram_mb"] / 1024.0,
                len(ready_hosts),
                capacity["ram_mb_shortfall"] / 1024.0,
            )
        )

    if infeasible_reasons:
        # Aggregate arithmetic already says no; skip the packing search so the
        # operator gets the unambiguous capacity answer instead of a per-VM one.
        return _models.DrainPlan(
            steps=[],
            feasible=False,
            blocked=[
                {
                    "vm_id": vm_id,
                    "vcpus": vm_flavor_request(details)[0],
                    "ram_gb": vm_flavor_request(details)[1] / 1024.0,
                    "reasons": {"*": "aggregate_capacity_shortfall"},
                }
                for vm_id, details in movable
            ],
            capacity=capacity,
            final_weighted_mad=baseline_weighted_mad,
            final_host_cpu=dict(ctx.host_cpu_pct),
            final_host_ram=dict(ctx.host_ram_pct),
            hotspot_hosts=[],
            infeasible_reasons=infeasible_reasons,
            **shared,
        )

    best_result = None
    best_ordering = ""
    for ordering, ordered_movable in drain_vm_orderings(movable):
        result = pack_drain_vms(
            ctx,
            drain_host,
            remaining_hosts,
            ordered_movable,
            cpu_weight,
            ram_weight,
            allow_hotspot,
        )
        if best_result is None:
            best_result, best_ordering = result, ordering
            continue
        # Fewest blocked VMs wins, then fewest soft-group violations; ties are
        # broken by the most balanced projected end state.
        candidate_key = (len(result["blocked"]), len(result["soft_warnings"]),
                         result["final_weighted_mad"])
        best_key = (len(best_result["blocked"]), len(best_result["soft_warnings"]),
                    best_result["final_weighted_mad"])
        if candidate_key < best_key:
            best_result, best_ordering = result, ordering

    blocked = best_result["blocked"]
    _cpu_cap, ram_cap = drain_host_caps(allow_hotspot)
    hotspot_hosts = []
    for host in remaining_hosts:
        cpu_pct = best_result["final_host_cpu"][host]
        ram_pct = best_result["final_host_ram"][host]
        if cpu_pct > _config.runtime().drain_max_host_cpu_pct or ram_pct > ram_cap:
            hotspot_hosts.append({"host": host, "cpu_pct": cpu_pct, "ram_pct": ram_pct})

    if blocked:
        infeasible_reasons.append(
            "aggregate has enough total capacity but no safe placement exists for "
            + str(len(blocked))
            + " VM(s); capacity is fragmented or blocked by policy"
        )
        if affinity_conflicts:
            # Most likely cause when destinations are rejected on policy only:
            # the group must stay co-located but is being evacuated one VM at
            # a time, so no single move can keep it together.
            infeasible_reasons.append(
                "affinity group(s) with 2+ members on "
                + drain_host
                + " cannot be evacuated one VM at a time: "
                + ", ".join(
                    "{}[{} VM(s)]".format(
                        c["group_name"] or c["group_id"], len(c["vm_ids"])
                    )
                    for c in affinity_conflicts
                )
                + " - migrate the group manually or re-run with --no-server-groups"
            )

    # Only the absolute RAM ceiling makes an otherwise-complete plan infeasible.
    # Soft hotspot overshoot is reported but does not block, because the packer
    # already refused every placement above the configured cap.
    critical_hosts = [
        h for h in hotspot_hosts
        if h["ram_pct"] > _config.runtime().drain_absolute_max_host_ram_pct
    ]
    if critical_hosts:
        infeasible_reasons.append(
            "projected RAM above the absolute ceiling "
            + f"{_config.runtime().drain_absolute_max_host_ram_pct:.1f}% on: "
            + ", ".join(
                "{}={:.1f}%".format(h["host"], h["ram_pct"]) for h in critical_hosts
            )
        )

    return _models.DrainPlan(
        steps=best_result["steps"],
        feasible=not infeasible_reasons,
        blocked=blocked,
        capacity=capacity,
        final_weighted_mad=best_result["final_weighted_mad"],
        final_host_cpu=best_result["final_host_cpu"],
        final_host_ram=best_result["final_host_ram"],
        hotspot_hosts=hotspot_hosts,
        infeasible_reasons=infeasible_reasons,
        ordering=best_ordering,
        soft_policy_warnings=best_result["soft_warnings"],
        **shared,
    )


def drain_plan_to_migration_plan(plan):
    """Wrap a DrainPlan so the shared execution/audit path can consume it."""
    if plan.remaining_hosts:
        final_cpu_mad = _planning_balance.mean_absolute_deviation(
            [plan.final_host_cpu[h] for h in plan.remaining_hosts]
        )
        final_ram_mad = _planning_balance.mean_absolute_deviation(
            [plan.final_host_ram[h] for h in plan.remaining_hosts]
        )
    else:
        final_cpu_mad = 0.0
        final_ram_mad = 0.0
    return _models.MigrationPlan(
        plan_type=_config.DRAIN_PLAN_TYPE,
        steps=list(plan.steps),
        baseline_weighted_mad=plan.baseline_weighted_mad,
        final_weighted_mad=plan.final_weighted_mad,
        improvement=plan.baseline_weighted_mad - plan.final_weighted_mad,
        mad_cpu=final_cpu_mad,
        mad_ram=final_ram_mad,
        final_host_cpu=plan.final_host_cpu,
        final_host_ram=plan.final_host_ram,
        search_scope="DRAIN " + plan.host,
    )
