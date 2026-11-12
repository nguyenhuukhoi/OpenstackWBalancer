"""MAKE_ROOM: the conservative two-step plan (move a small VM B->C to free
Placement capacity on B, then the primary VM A->B)."""
from copy import deepcopy

import compute_maintenance.config as _config
import compute_maintenance.domain.models as _models
import compute_maintenance.domain.policies as _planning_constraints
from compute_maintenance.domain.balance.filters import (
    availability_zone_allows,
    destination_hot_after,
    is_hotspot_guardrail_satisfied,
    make_step_from_projection,
    max_instances_allows,
    project_runtime_move,
    vm_candidate_details,
    would_create_any_new_hotspot,
    would_create_new_hotspot,
)
from compute_maintenance.domain.balance.modes import (
    get_hotspot_source_host,
    mean_absolute_deviation,
)


def placement_with_usage_delta(placement, delta):
    cloned = deepcopy(placement)
    for rc in ("VCPU", "MEMORY_MB"):
        inv = cloned[rc]
        inv["used"] = max(inv["used"] + float(delta.get(rc, 0.0)), 0.0)
        inv["free"] = max(inv["policy_capacity"] - inv["used"], 0.0)
        inv["nominal_free"] = max(inv["nominal_capacity"] - inv["used"], 0.0)
        inv["allocation_pct"] = (
            inv["used"] / inv["nominal_capacity"] * 100.0
            if inv["nominal_capacity"] > 0
            else 100.0
        )
    return cloned


def find_best_make_room_plan(
    ctx,
    cfg,
    mode,
    source_hosts,
    baseline_weighted_mad,
    cpu_weight,
    ram_weight,
    search_scope,
):
    """Search a conservative two-step B->C then A->B make-room plan."""
    if not cfg.enable_make_room:
        return None

    best_plan = None
    best_improvement = 0.0
    now = None  # cooldown is already captured in the input context
    current_max_cpu = max(ctx.host_cpu_pct.values()) if ctx.host_cpu_pct else 0.0
    current_max_ram = max(ctx.host_ram_pct.values()) if ctx.host_ram_pct else 0.0

    # vm_host_map holds every server of the cloud (server-group topology).
    # Only VMs on aggregate hosts can be planned, so index those once instead
    # of scanning (and charging the search budget for) the whole cloud for
    # every (primary VM, blocked target) pair.
    effective = set(ctx.effective_hosts)
    vms_on_host = {}
    for vm_id, host in ctx.vm_host_map.items():
        if host in effective:
            vms_on_host.setdefault(host, []).append(vm_id)

    for primary_vm, primary_src in ctx.vm_host_map.items():
        if primary_src not in source_hosts:
            continue
        primary = vm_candidate_details(ctx, primary_vm, now=now, record=False)
        if primary is None:
            continue

        primary_vcpus = int(primary.get("vcpus", 1) or 1)
        primary_ram_mb = round(float(primary.get("ram_gb", 2.0) or 2.0) * 1024.0)

        for target in ctx.effective_hosts:
            ctx.consume_candidate()
            if target == primary_src:
                continue
            if ctx.dest_ready_hosts is not None and target not in ctx.dest_ready_hosts:
                continue

            # MAKE_ROOM is specifically for a destination that is attractive at
            # runtime but currently blocked by Placement capacity.
            direct_fits, direct_reason = _planning_constraints.placement_can_fit(
                ctx.host_placement[target], primary_vcpus, primary_ram_mb
            )
            if direct_fits or direct_reason not in ("placement_vcpu", "placement_memory"):
                continue

            if not availability_zone_allows(ctx, primary, primary_src, target):
                continue

            for room_vm in vms_on_host.get(target, ()):
                ctx.consume_candidate()
                if room_vm == primary_vm:
                    continue
                room = vm_candidate_details(ctx, room_vm, now=now, record=False)
                if room is None:
                    continue

                room_cpu_pct = float(room.get("cpu_pct", 0.0) or 0.0)
                room_ram_pct = float(room.get("ram_pct", 0.0) or 0.0)
                if room_cpu_pct > cfg.make_room_max_vm_cpu_percent:
                    continue
                if room_ram_pct > cfg.make_room_max_vm_ram_percent:
                    continue

                room_vcpus = int(room.get("vcpus", 1) or 1)
                room_ram_mb = round(float(room.get("ram_gb", 2.0) or 2.0) * 1024.0)

                for buffer_host in ctx.effective_hosts:
                    ctx.consume_candidate()
                    # The buffer must be a third host. Parking the room VM on the
                    # primary's own (hot) source makes it hotter between steps.
                    if buffer_host in (target, primary_src):
                        continue
                    if not availability_zone_allows(ctx, room, target, buffer_host):
                        continue
                    if ctx.dest_ready_hosts is not None and buffer_host not in ctx.dest_ready_hosts:
                        continue
                    if not max_instances_allows(ctx, buffer_host):
                        continue

                    if ctx.enforce_server_groups and not _planning_constraints.is_server_group_move_compliant(
                        room_vm,
                        target,
                        buffer_host,
                        ctx.vm_host_map,
                        ctx.vm_to_groups,
                    ):
                        continue
                    if ctx.enforce_server_groups and _planning_constraints.soft_group_move_delta(
                        room_vm, target, buffer_host, ctx.vm_host_map, ctx.vm_to_groups
                    ) > 0:
                        continue

                    room_fits, _ = _planning_constraints.placement_can_fit(
                        ctx.host_placement[buffer_host], room_vcpus, room_ram_mb
                    )
                    if not room_fits:
                        continue

                    projection1 = project_runtime_move(
                        ctx.host_cpu_pct,
                        ctx.host_ram_pct,
                        ctx.host_cores,
                        ctx.host_ram,
                        room,
                        target,
                        buffer_host,
                    )
                    if projection1 is None:
                        continue
                    if would_create_new_hotspot(
                        ctx.host_cpu_pct,
                        ctx.host_ram_pct,
                        projection1["new_host_cpu"],
                        projection1["new_host_ram"],
                        buffer_host,
                    ) or destination_hot_after(
                        projection1["new_host_cpu"], projection1["new_host_ram"], buffer_host
                    ):
                        continue

                    # Simulate Placement after B->C. The target releases the
                    # make-room VM allocation; the buffer consumes it.
                    target_delta = {
                        "VCPU": -room_vcpus,
                        "MEMORY_MB": -room_ram_mb,
                    }
                    target_after_release = placement_with_usage_delta(
                        ctx.host_placement[target], target_delta
                    )
                    primary_fits, _ = _planning_constraints.placement_can_fit(
                        target_after_release, primary_vcpus, primary_ram_mb
                    )
                    if not primary_fits:
                        continue

                    # Re-evaluate server-group policy for the primary move after
                    # the room VM has moved to C.
                    simulated_vm_hosts = dict(ctx.vm_host_map)
                    simulated_vm_hosts[room_vm] = buffer_host
                    if ctx.enforce_server_groups and not _planning_constraints.is_server_group_move_compliant(
                        primary_vm,
                        primary_src,
                        target,
                        simulated_vm_hosts,
                        ctx.vm_to_groups,
                    ):
                        continue
                    if ctx.enforce_server_groups and _planning_constraints.soft_group_move_delta(
                        primary_vm, primary_src, target, simulated_vm_hosts, ctx.vm_to_groups
                    ) > 0:
                        continue

                    # Instance-count simulation: B loses one in step 1 and gets
                    # one back in step 2; C transiently and finally gains one.
                    simulated_counts = dict(ctx.host_running_vms or {})
                    if target in simulated_counts:
                        simulated_counts[target] = max(simulated_counts[target] - 1, 0)
                    if buffer_host in simulated_counts:
                        simulated_counts[buffer_host] += 1
                    if not max_instances_allows(ctx, target, current_counts=simulated_counts):
                        continue

                    projection2 = project_runtime_move(
                        projection1["new_host_cpu"],
                        projection1["new_host_ram"],
                        ctx.host_cores,
                        ctx.host_ram,
                        primary,
                        primary_src,
                        target,
                    )
                    if projection2 is None:
                        continue

                    final_cpu = projection2["new_host_cpu"]
                    final_ram = projection2["new_host_ram"]
                    if would_create_any_new_hotspot(
                        ctx.host_cpu_pct,
                        ctx.host_ram_pct,
                        final_cpu,
                        final_ram,
                        ctx.effective_hosts,
                    ) or destination_hot_after(final_cpu, final_ram, target) \
                            or destination_hot_after(final_cpu, final_ram, buffer_host):
                        continue

                    if mode in _config.HOTSPOT_MODES and not is_hotspot_guardrail_satisfied(
                        mode,
                        final_cpu,
                        final_ram,
                        target,
                        current_max_cpu,
                        current_max_ram,
                    ):
                        continue

                    mad_cpu = mean_absolute_deviation(list(final_cpu.values()))
                    mad_ram = mean_absolute_deviation(list(final_ram.values()))
                    final_weighted_mad = (mad_cpu * cpu_weight) + (mad_ram * ram_weight)
                    improvement = baseline_weighted_mad - final_weighted_mad
                    if improvement <= best_improvement:
                        continue

                    step1 = make_step_from_projection(
                        room_vm,
                        target,
                        buffer_host,
                        "make_room_release",
                        projection1,
                        ctx.host_placement[buffer_host],
                    )
                    step2 = make_step_from_projection(
                        primary_vm,
                        primary_src,
                        target,
                        "make_room_balance",
                        projection2,
                        target_after_release,
                    )
                    best_improvement = improvement
                    best_plan = _models.MigrationPlan(
                        plan_type="make_room",
                        steps=[step1, step2],
                        baseline_weighted_mad=baseline_weighted_mad,
                        final_weighted_mad=final_weighted_mad,
                        improvement=improvement,
                        mad_cpu=mad_cpu,
                        mad_ram=mad_ram,
                        final_host_cpu=final_cpu,
                        final_host_ram=final_ram,
                        search_scope=search_scope,
                    )

    return best_plan


def search_best_make_room_for_state(
    ctx,
    cfg,
    mode,
    baseline_weighted_mad,
    cpu_weight,
    ram_weight,
):
    hotspot_source_host = get_hotspot_source_host(mode, ctx.host_cpu_pct, ctx.host_ram_pct)
    if hotspot_source_host:
        plan = find_best_make_room_plan(
            ctx,
            cfg,
            mode,
            [hotspot_source_host],
            baseline_weighted_mad,
            cpu_weight,
            ram_weight,
            f"MAKE_ROOM HOTSPOT-FIRST ({hotspot_source_host})",
        )
        if plan:
            return plan
        return find_best_make_room_plan(
            ctx,
            cfg,
            mode,
            ctx.effective_hosts,
            baseline_weighted_mad,
            cpu_weight,
            ram_weight,
            f"MAKE_ROOM FALLBACK-FULL (after {hotspot_source_host})",
        )

    return find_best_make_room_plan(
        ctx,
        cfg,
        mode,
        ctx.effective_hosts,
        baseline_weighted_mad,
        cpu_weight,
        ram_weight,
        "MAKE_ROOM FULL",
    )
