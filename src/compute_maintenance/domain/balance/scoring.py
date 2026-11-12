"""Direct-move search: scoring by weighted MAD improvement, the rc11
soft-anti-affinity tie-break and the hotspot-first / full-search strategy."""
import compute_maintenance.config as _config
import compute_maintenance.domain.models as _models
import compute_maintenance.domain.policies as _planning_constraints
from compute_maintenance.domain.balance.filters import (
    check_direct_destination,
    is_hotspot_guardrail_satisfied,
    make_step_from_projection,
    record_rejection,
    vm_candidate_details,
)
from compute_maintenance.domain.balance.modes import (
    determine_improvement_threshold,
    get_hotspot_source_host,
    mean_absolute_deviation,
)


def soft_aa_tiebreak_enabled(ctx):
    """rc11 tie-break: only with heal on and server groups enforced."""
    return bool(_config.runtime().heal_soft_anti_affinity and ctx.enforce_server_groups)


def choose_with_soft_aa_tiebreak(best_move, candidates, improvement_threshold, margin):
    """Prefer a move that heals soft-anti-affinity among near-best moves.

    ``candidates`` is ``[(BestMove, violation_reduction), ...]`` in search
    order. Only moves that already pass ``improvement_threshold`` and lie
    within ``margin`` (relative) of the best improvement compete; the one
    removing the most violations wins, equal reductions keep the higher
    improvement (the original best on a full tie). It never admits a move
    the plain search would have rejected.
    """
    if best_move is None or best_move.mad_improvement < improvement_threshold:
        return best_move
    floor = max(improvement_threshold, best_move.mad_improvement * (1.0 - margin))
    chosen, chosen_reduction = best_move, None
    for move, reduction in candidates:
        if move is best_move:
            chosen_reduction = reduction
            break
    if chosen_reduction is None:
        return best_move
    for move, reduction in candidates:
        if move.mad_improvement < floor:
            continue
        if reduction > chosen_reduction or (
            reduction == chosen_reduction and move.mad_improvement > chosen.mad_improvement
        ):
            chosen, chosen_reduction = move, reduction
    return chosen


def find_best_move(ctx, mode, source_hosts, baseline_weighted_mad, cpu_weight, ram_weight,
                   improvement_threshold=None):
    """Search the best direct VM move.

    Runtime impact is scored from Prometheus-observed usage. Destination
    feasibility is a hard Placement VCPU/MEMORY_MB constraint using the VM
    flavor requirement.

    rc11: with HEAL_SOFT_ANTI_AFFINITY on, a move within
    SOFT_AA_TIEBREAK_MARGIN of the best improvement that removes more
    soft-anti-affinity violations is preferred (choose_with_soft_aa_tiebreak).
    """
    best_move = None
    best_improvement = 0.0
    current_max_cpu = max(ctx.host_cpu_pct.values()) if ctx.host_cpu_pct else 0.0
    current_max_ram = max(ctx.host_ram_pct.values()) if ctx.host_ram_pct else 0.0
    now = None  # cooldown is already captured in the input context
    candidates = [] if soft_aa_tiebreak_enabled(ctx) else None

    for vm_id, src_host in ctx.vm_host_map.items():
        if src_host not in source_hosts:
            continue

        details = vm_candidate_details(ctx, vm_id, now=now, record=True)
        if details is None:
            continue

        for dst_host in ctx.effective_hosts:
            ctx.consume_candidate()
            if dst_host == src_host:
                continue

            reason, projection, placement = check_direct_destination(
                ctx, vm_id, details, src_host, dst_host
            )
            if reason is not None:
                record_rejection(ctx, reason, dst_host)
                continue

            vcpus = int(details.get("vcpus", 1) or 1)
            new_host_cpu = projection["new_host_cpu"]
            new_host_ram = projection["new_host_ram"]

            if mode in _config.HOTSPOT_MODES and not is_hotspot_guardrail_satisfied(
                mode,
                new_host_cpu,
                new_host_ram,
                dst_host,
                current_max_cpu,
                current_max_ram,
            ):
                record_rejection(ctx, "hotspot_guardrail", dst_host)
                continue

            ctx.technical_candidate_count += 1

            mad_cpu = mean_absolute_deviation(list(new_host_cpu.values()))
            mad_ram = mean_absolute_deviation(list(new_host_ram.values()))
            new_weighted_mad = (mad_cpu * cpu_weight) + (mad_ram * ram_weight)
            mad_improvement = baseline_weighted_mad - new_weighted_mad

            is_best = mad_improvement > best_improvement
            if not is_best and not (candidates is not None and mad_improvement > 0):
                continue
            move = _models.BestMove(
                vm_id=vm_id,
                src_host=src_host,
                dst_host=dst_host,
                cpu_impact_src=projection["cpu_impact_src"],
                cpu_impact_dst=projection["cpu_impact_dst"],
                ram_impact_src=projection["ram_impact_src"],
                ram_impact_dst=projection["ram_impact_dst"],
                new_host_cpu=new_host_cpu,
                new_host_ram=new_host_ram,
                real_vm_cores=projection["real_vm_cores"],
                vcpus=vcpus,
                real_vm_ram=projection["real_vm_ram"],
                vm_ram_gb=projection["vm_ram_gb"],
                dst_free_cores=placement["VCPU"]["free"],
                dst_free_ram_gb=placement["MEMORY_MB"]["free"] / 1024.0,
                new_weighted_mad=new_weighted_mad,
                mad_improvement=mad_improvement,
                mad_cpu=mad_cpu,
                mad_ram=mad_ram,
            )
            if candidates is not None and mad_improvement > 0:
                candidates.append((move, _planning_constraints.soft_anti_affinity_move_reduction(
                    vm_id, src_host, dst_host, ctx.vm_host_map, ctx.vm_to_groups)))
            if is_best:
                best_improvement = mad_improvement
                best_move = move

    if candidates:
        threshold = (determine_improvement_threshold() if improvement_threshold is None
                     else improvement_threshold)
        best_move = choose_with_soft_aa_tiebreak(
            best_move, candidates, threshold, _config.runtime().soft_aa_tiebreak_margin)
        best_improvement = best_move.mad_improvement if best_move is not None else best_improvement

    return best_move, best_improvement


def direct_move_to_plan(move, baseline_weighted_mad, search_scope):
    if move is None:
        return None
    placement_stub = {
        "VCPU": {"free": move.dst_free_cores},
        "MEMORY_MB": {"free": move.dst_free_ram_gb * 1024.0},
    }
    projection = {
        "vcpus": move.vcpus,
        "vm_ram_gb": move.vm_ram_gb,
        "real_vm_cores": move.real_vm_cores,
        "real_vm_ram": move.real_vm_ram,
        "cpu_impact_src": move.cpu_impact_src,
        "cpu_impact_dst": move.cpu_impact_dst,
        "ram_impact_src": move.ram_impact_src,
        "ram_impact_dst": move.ram_impact_dst,
    }
    step = make_step_from_projection(
        move.vm_id,
        move.src_host,
        move.dst_host,
        "direct_balance",
        projection,
        placement_stub,
    )
    return _models.MigrationPlan(
        plan_type="direct",
        steps=[step],
        baseline_weighted_mad=baseline_weighted_mad,
        final_weighted_mad=move.new_weighted_mad,
        improvement=move.mad_improvement,
        mad_cpu=move.mad_cpu,
        mad_ram=move.mad_ram,
        final_host_cpu=move.new_host_cpu,
        final_host_ram=move.new_host_ram,
        search_scope=search_scope,
    )


def search_best_move_for_state(
    ctx,
    mode,
    baseline_weighted_mad,
    cpu_weight,
    ram_weight,
    improvement_threshold,
):
    """Run the current hotspot-first/full-search strategy unchanged."""
    effective_hosts = ctx.effective_hosts
    hotspot_source_host = get_hotspot_source_host(mode, ctx.host_cpu_pct, ctx.host_ram_pct)

    def search_move(source_hosts, scope_label):
        move, improvement = find_best_move(
            ctx,
            mode,
            source_hosts,
            baseline_weighted_mad,
            cpu_weight,
            ram_weight,
            improvement_threshold=improvement_threshold,
        )
        return move, improvement, scope_label

    best_move = None
    best_improvement = 0.0
    search_scope = "FULL"

    if hotspot_source_host:
        move1, improvement1, scope1 = search_move(
            [hotspot_source_host],
            f"HOTSPOT-FIRST ({hotspot_source_host})",
        )
        has_valid_move1 = bool(move1 and improvement1 >= improvement_threshold)

        if has_valid_move1:
            best_move, best_improvement, search_scope = move1, improvement1, scope1
        else:
            move2, improvement2, scope2 = search_move(
                effective_hosts,
                f"FALLBACK-FULL (after {hotspot_source_host})",
            )
            if improvement2 >= improvement1:
                best_move, best_improvement, search_scope = move2, improvement2, scope2
            else:
                best_move, best_improvement, search_scope = move1, improvement1, scope1
    else:
        best_move, best_improvement, search_scope = search_move(effective_hosts, "FULL")

    return best_move, best_improvement, search_scope, hotspot_source_host
