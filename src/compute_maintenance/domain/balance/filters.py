"""Candidate eligibility and the destination filter chain shared by direct
moves, the soft-anti-affinity heal and MAKE_ROOM (readiness, instance cap,
server groups, availability zone, Placement fit, runtime projection, hotspot
guards), plus the rejection/skip statistics they record."""
import compute_maintenance.config as _config
import compute_maintenance.domain.models as _models
import compute_maintenance.domain.policies as _planning_constraints


def is_hotspot_guardrail_satisfied(mode, new_host_cpu, new_host_ram, dst_host, current_max_cpu, current_max_ram):
    """
    For hotspot modes:
    - hotspot metric must go down
    - destination must not become hotspot
    """
    if mode in _config.RAM_HOTSPOT_MODES:
        new_max_ram = max(new_host_ram.values())
        dst_ram = new_host_ram[dst_host]
        if new_max_ram >= current_max_ram:
            return False
        if dst_ram >= _config.runtime().ram_hotspot_threshold:
            return False

    elif mode == _config.MODE_CPU_HOTSPOT:
        new_max_cpu = max(new_host_cpu.values())
        dst_cpu = new_host_cpu[dst_host]
        if new_max_cpu >= current_max_cpu:
            return False
        if dst_cpu >= _config.runtime().cpu_hotspot_threshold:
            return False

    return True


def _inc_counter(mapping, key, amount=1):
    if mapping is None:
        return
    mapping[key] = mapping.get(key, 0) + amount


def record_rejection(ctx, reason, dst_host=None):
    _inc_counter(ctx.reject_stats, reason)
    if dst_host is not None and ctx.dest_reject_stats is not None:
        per_host = ctx.dest_reject_stats.setdefault(dst_host, {})
        _inc_counter(per_host, reason)


def record_skip(ctx, reason):
    _inc_counter(ctx.skip_stats, reason)


def vm_candidate_details(ctx, vm_id, now=None, record=True):
    """Return VM details if it is eligible to be moved, otherwise None."""
    if vm_id in (ctx.excluded_vm_ids or set()):
        if record:
            record_skip(ctx, "excluded")
        return None

    # Drain mode sets ignore_cooldown: the per-VM cooldown only exists to damp
    # churn from automatic balancing, and a maintenance evacuation must not be
    # blocked by it.
    if not ctx.ignore_cooldown:
        cooldown_left = (ctx.cooldown_remaining or {}).get(vm_id, 0.0)
        if cooldown_left > 0:
            if record:
                record_skip(ctx, "cooldown")
            return None

    details = ctx.all_vm_details.get(vm_id)
    if not details:
        if record:
            record_skip(ctx, "missing_vm_details")
        return None

    if details.get("status") != _config.ACTIVE_VM_STATUS:
        if record:
            record_skip(ctx, "not_active")
        return None

    if not details.get("metadata_complete", False):
        if record:
            record_skip(ctx, "incomplete_metadata")
        return None

    if not details.get("metrics_complete", False):
        if record:
            record_skip(ctx, "incomplete_metrics")
        return None

    # Nova refuses live migration of a locked VM and of a VM with a running
    # task; planning them only produces a move the executor must reject.
    blocked = vm_planning_block_reason(details)
    if blocked:
        if record:
            record_skip(ctx, blocked)
        return None

    return details


def vm_planning_block_reason(details):
    """Why Nova/the planner cannot move this VM now, else None."""
    if details.get("locked"):
        return "locked"
    if details.get("task_state"):
        return "task_in_progress"
    if details.get("unsupported_resources"):
        return "unsupported_resources"
    return None


def availability_zone_allows(ctx, details, src_host, dst_host):
    """Keep a VM inside its availability zone.

    The VM's zone comes from Nova (OS-EXT-AZ); when Nova does not report one
    the source host's zone is used. An unknown destination zone is refused.
    """
    host_az = getattr(ctx, "host_az", None)
    if host_az is None:
        return True
    dst_az = host_az.get(dst_host)
    if dst_az is None:
        return False
    vm_az = (details or {}).get("availability_zone") or host_az.get(src_host)
    if vm_az is None:
        # Zone unknown (should not happen for a scheduled VM): only safe when
        # every host of the aggregate is in one and the same zone.
        return len(set(host_az.values())) == 1
    return vm_az == dst_az


def destination_hot_after(after_cpu, after_ram, host):
    """True when ``host`` would sit above a CPU or RAM hotspot threshold.

    Unlike would_create_new_hotspot this also refuses a destination that is
    already above a threshold: adding load to it makes it hotter.
    """
    return (after_cpu.get(host, 0.0) > _config.runtime().cpu_hotspot_threshold
            or after_ram.get(host, 0.0) > _config.runtime().ram_hotspot_threshold)


def max_instances_allows(ctx, dst_host, current_counts=None):
    max_instances = (ctx.host_max_instances or {}).get(dst_host, 0)
    if not max_instances or max_instances <= 0:
        return True
    counts = current_counts if current_counts is not None else (ctx.host_running_vms or {})
    current_vms = counts.get(dst_host)
    if current_vms is None:
        return False
    return current_vms + 1 <= max_instances


def project_runtime_move(host_cpu_pct, host_ram_pct, host_cores, host_ram, details, src_host, dst_host):
    vcpus = float(details.get("vcpus", 1) or 1)
    vm_cpu_pct = float(details.get("cpu_pct", 0.0) or 0.0)
    vm_ram_gb = float(details.get("ram_gb", 2.0) or 2.0)
    vm_ram_pct = float(details.get("ram_pct", 0.0) or 0.0)
    vm_ram_rss_gb = details.get("ram_rss_gb")

    real_vm_cores = vcpus * (vm_cpu_pct / 100.0)
    real_vm_ram = (
        float(vm_ram_rss_gb)
        if vm_ram_rss_gb is not None
        else vm_ram_gb * (vm_ram_pct / 100.0)
    )

    if host_cores.get(src_host, 0) <= 0 or host_cores.get(dst_host, 0) <= 0:
        return None
    if host_ram.get(src_host, 0) <= 0 or host_ram.get(dst_host, 0) <= 0:
        return None

    cpu_impact_src = (real_vm_cores / host_cores[src_host]) * 100.0
    cpu_impact_dst = (real_vm_cores / host_cores[dst_host]) * 100.0
    ram_impact_src = (real_vm_ram / host_ram[src_host]) * 100.0
    ram_impact_dst = (real_vm_ram / host_ram[dst_host]) * 100.0

    new_host_cpu = host_cpu_pct.copy()
    new_host_ram = host_ram_pct.copy()
    new_host_cpu[src_host] = max(new_host_cpu[src_host] - cpu_impact_src, 0.0)
    new_host_cpu[dst_host] = max(new_host_cpu[dst_host] + cpu_impact_dst, 0.0)
    new_host_ram[src_host] = max(new_host_ram[src_host] - ram_impact_src, 0.0)
    new_host_ram[dst_host] = max(new_host_ram[dst_host] + ram_impact_dst, 0.0)

    return {
        "vcpus": vcpus,
        "vm_ram_gb": vm_ram_gb,
        "real_vm_cores": real_vm_cores,
        "real_vm_ram": real_vm_ram,
        "cpu_impact_src": cpu_impact_src,
        "cpu_impact_dst": cpu_impact_dst,
        "ram_impact_src": ram_impact_src,
        "ram_impact_dst": ram_impact_dst,
        "new_host_cpu": new_host_cpu,
        "new_host_ram": new_host_ram,
    }


def make_step_from_projection(vm_id, src_host, dst_host, reason, projection, placement):
    return _models.MigrationStep(
        vm_id=vm_id,
        src_host=src_host,
        dst_host=dst_host,
        reason=reason,
        vcpus=projection["vcpus"],
        vm_ram_gb=projection["vm_ram_gb"],
        real_vm_cores=projection["real_vm_cores"],
        real_vm_ram=projection["real_vm_ram"],
        cpu_impact_src=projection["cpu_impact_src"],
        cpu_impact_dst=projection["cpu_impact_dst"],
        ram_impact_src=projection["ram_impact_src"],
        ram_impact_dst=projection["ram_impact_dst"],
        dst_free_vcpu=placement["VCPU"]["free"],
        dst_free_ram_gb=placement["MEMORY_MB"]["free"] / 1024.0,
    )


def check_direct_destination(ctx, vm_id, details, src_host, dst_host):
    """Filter chain of one direct move ``vm_id`` src -> dst.

    Shared by the balancing search and the soft-anti-affinity heal so both
    apply exactly the same rules: destination readiness, instance cap, hard
    server groups, soft groups never worse, availability zone, Placement
    capacity with headroom, runtime capacity, no new/worse hotspot.
    Returns ``(rejection_reason, projection, placement)``; the reason is None
    when the move passes.
    """
    if ctx.dest_ready_hosts is not None and dst_host not in ctx.dest_ready_hosts:
        return "destination_not_ready", None, None

    if not max_instances_allows(ctx, dst_host):
        return "max_instances", None, None

    if ctx.enforce_server_groups and not _planning_constraints.is_server_group_move_compliant(
        vm_id, src_host, dst_host, ctx.vm_host_map, ctx.vm_to_groups
    ):
        return "server_group", None, None

    # Balancing must never make a soft-(anti-)affinity group worse.
    if ctx.enforce_server_groups and _planning_constraints.soft_group_move_delta(
        vm_id, src_host, dst_host, ctx.vm_host_map, ctx.vm_to_groups
    ) > 0:
        return "soft_server_group", None, None

    if not availability_zone_allows(ctx, details, src_host, dst_host):
        return "availability_zone", None, None

    vcpus = int(details.get("vcpus", 1) or 1)
    ram_mb = round(float(details.get("ram_gb", 2.0) or 2.0) * 1024.0)
    placement = ctx.host_placement[dst_host]
    fits, fit_reason = _planning_constraints.placement_can_fit(placement, vcpus, ram_mb)
    if not fits:
        return fit_reason, None, None

    projection = project_runtime_move(
        ctx.host_cpu_pct,
        ctx.host_ram_pct,
        ctx.host_cores,
        ctx.host_ram,
        details,
        src_host,
        dst_host,
    )
    if projection is None:
        return "runtime_capacity_missing", None, None

    if would_create_new_hotspot(
        ctx.host_cpu_pct,
        ctx.host_ram_pct,
        projection["new_host_cpu"],
        projection["new_host_ram"],
        dst_host,
    ) or destination_hot_after(projection["new_host_cpu"], projection["new_host_ram"], dst_host):
        return "would_create_new_hotspot", None, None

    return None, projection, placement


def would_create_new_hotspot(before_cpu, before_ram, after_cpu, after_ram, host):
    if before_cpu.get(host, 0.0) <= _config.runtime().cpu_hotspot_threshold < after_cpu.get(host, 0.0):
        return True
    return before_ram.get(host, 0.0) <= _config.runtime().ram_hotspot_threshold < after_ram.get(host, 0.0)


def would_create_any_new_hotspot(before_cpu, before_ram, after_cpu, after_ram, hosts):
    return any(
        would_create_new_hotspot(before_cpu, before_ram, after_cpu, after_ram, host)
        for host in hosts
    )
