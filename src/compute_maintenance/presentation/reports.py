"""report — extracted domain responsibilities."""
import json
import logging
import os
import time
from contextvars import ContextVar
from datetime import datetime

import compute_maintenance.adapters.selection as _selection
import compute_maintenance.config as _config
import compute_maintenance.domain.balance as _planning_balance
import compute_maintenance.domain.drain as _planning_drain
import compute_maintenance.domain.plan_context as _plan_context
import compute_maintenance.domain.severity as _severity

event_path = ContextVar("migration_event_path", default=None)
_log = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s')


def format_migration_duration(duration_seconds):
    if duration_seconds is None:
        return "n/a"

    try:
        seconds = max(float(duration_seconds), 0.0)
    except (TypeError, ValueError):
        return "n/a"

    return f"{seconds:.1f} seconds ({seconds / 60.0:.2f} minutes)"


def format_migration_timestamp(value):
    if value is None:
        return "n/a"

    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d %H:%M:%S')

    try:
        return datetime.fromtimestamp(float(value)).strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError, OSError):
        return "n/a"


def format_mode_label(mode):
    if mode == _config.MODE_RAM_HOTSPOT:
        return f"RAM_HOTSPOT (host RAM above {_config.runtime().ram_hotspot_threshold:.1f}%)"

    if mode == _config.MODE_RAM_CRITICAL_HOTSPOT:
        return f"RAM_CRITICAL_HOTSPOT (host RAM above {_config.runtime().ram_critical_threshold:.1f}%)"

    if mode == _config.MODE_CPU_HOTSPOT:
        return f"CPU_HOTSPOT (host CPU above {_config.runtime().cpu_hotspot_threshold:.1f}%)"

    if mode == _config.MODE_PRESSURE:
        return f"PRESSURE (cluster average above {_config.runtime().auto_pressure_threshold:.1f}%)"

    return "BALANCING (no pressure or hotspot)"


def format_cluster_status(status):
    labels = {
        "RAM_HOTSPOT_OPTIMIZABLE": "RAM_HOTSPOT_OPTIMIZABLE",
        "RAM_HOTSPOT_NO_VALID_MOVE": "RAM_HOTSPOT_NO_VALID_MOVE",
        "CPU_HOTSPOT_OPTIMIZABLE": "CPU_HOTSPOT_OPTIMIZABLE",
        "CPU_HOTSPOT_NO_VALID_MOVE": "CPU_HOTSPOT_NO_VALID_MOVE",

        "RAM_PRESSURE": "RAM_PRESSURE",
        "CPU_PRESSURE": "CPU_PRESSURE",
        "PRESSURE_NO_VALID_MOVE": "PRESSURE_NO_VALID_MOVE",

        "BALANCED": "BALANCED",
        "BALANCED_OPTIMIZABLE": "BALANCED_OPTIMIZABLE",

        "UNBALANCED": "UNBALANCED",
        "UNBALANCED_NO_VALID_MOVE": "UNBALANCED_NO_VALID_MOVE",

        "UNKNOWN": "UNKNOWN",
    }
    return labels.get(status, status)


def format_excluded_vm_lines(excluded_vms, total_vms, limit=_config.EXCLUDED_VM_DISPLAY_LIMIT):
    """Render the excluded VMs as report lines, one per exclude entry.

    Long groups are truncated so a wildcard pinning dozens of VMs cannot flood
    an Icinga check result.
    """
    if not excluded_vms:
        return []

    lines = [
        f"Excluded: {len(excluded_vms)} of {total_vms} VM(s) pinned by the exclude file"
    ]
    for pattern, names in _selection.group_excluded_vms_by_pattern(excluded_vms).items():
        shown = ", ".join(names[:limit])
        if len(names) > limit:
            shown = f"{shown}, +{len(names) - limit} more"
        lines.append(f"  - {pattern} ({len(names)}): {shown}")

    return lines


def print_baseline_table(state):
    hosts = state.hosts_in_agg
    print("Baseline runtime load + Placement scheduler capacity:")
    header = (
        f"{'Host':<18} {'CPU%':>6} {'RAM%':>6} {'rCPU':>6} {'rRAMG':>7} "
        f"{'VCPU used/cap':>16} {'safeFree':>8} {'MEM used/cap G':>18} {'safeFreeG':>10} {'Capacity':>22}"
    )
    print(header)
    print("-" * len(header))
    for h in hosts:
        p = state.host_placement[h]
        v = p["VCPU"]
        m = p["MEMORY_MB"]
        print(
            f"{h:<18} "
            f"{state.host_cpu_pct[h]:6.2f} "
            f"{state.host_ram_pct[h]:6.2f} "
            f"{state.host_cores[h]:6d} "
            f"{state.host_ram[h]:7.1f} "
            f"{v['used']:6.0f}/{v['nominal_capacity']:<8.0f} "
            f"{v['free']:8.1f} "
            f"{m['used']/1024.0:7.1f}/{m['nominal_capacity']/1024.0:<8.1f} "
            f"{m['free']/1024.0:10.1f} "
            f"{state.placement_status_by_host[h]:>22}"
        )
    print()


def format_reason(reason, state=None):
    if reason == "no_technical_candidate":
        return "no feasible VM can be moved"

    if reason == "no_hotspot_candidate_above_improvement_threshold":
        if state:
            return (
                "no move reduces hotspot enough because best improvement "
                f"({state.best_improvement:.3f}) is below threshold "
                f"({state.improvement_threshold:.3f})"
            )
        return "no move reduces hotspot enough"

    if reason == "no_pressure_candidate_above_improvement_threshold":
        return "no move reduces pressure enough because no candidate meets required improvement"

    if reason == "no_candidate_above_improvement_threshold":
        return "no migration provides enough improvement"

    return reason


def get_hot_compute(state):
    mode = state.mode
    host_cpu_pct = state.host_cpu_pct or {}
    host_ram_pct = state.host_ram_pct or {}

    if mode in _config.RAM_HOTSPOT_MODES and host_ram_pct:
        return max(host_ram_pct, key=host_ram_pct.get)

    if mode == _config.MODE_CPU_HOTSPOT and host_cpu_pct:
        return max(host_cpu_pct, key=host_cpu_pct.get)

    return None


def format_candidate_line(state):
    plan = state.best_plan_candidate
    if not plan:
        return "Candidate: none"

    parts = []
    for step in plan.steps:
        vm_name = state.all_vm_names.get(step.vm_id, step.vm_id)
        parts.append(f"{vm_name} ({step.src_host} -> {step.dst_host})")
    joined = " ; ".join(parts)
    label = "Proposed plan" if state.has_valid_move else "Best plan candidate"
    suffix = "" if state.has_valid_move else " [below threshold]"
    return f"{label}: {plan.plan_type.upper()} | {joined}{suffix}"


def build_headline(state):
    severity = state.severity
    status = state.detail_status
    best = state.best_improvement
    threshold = state.improvement_threshold
    max_cpu = state.max_cpu
    max_ram = state.max_ram
    
    hot_compute = get_hot_compute(state)
    hot_str = f" | hot_compute={hot_compute}" if hot_compute else ""

    if status == _severity.SOFT_AA_HEAL_DETAIL_STATUS:
        heal = state.soft_aa_heal or {}
        group = heal.get("group") or {}
        return (
            f"{severity} - soft-anti-affinity spread broken "
            f"({heal.get('violations', 0)} violation(s) in {len(heal.get('broken_groups') or [])} group(s)) "
            f"and can be healed automatically (next: group {group.get('name') or group.get('id')})"
        )

    if status == "RAM_HOTSPOT_NO_VALID_MOVE":
        return (
            f"{severity} - RAM hotspot {max_ram:.2f}% "
            f"but no effective migration (best={best:.3f} < {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "RAM_HOTSPOT_OPTIMIZABLE":
        return (
            f"{severity} - RAM hotspot {max_ram:.2f}% "
            f"and can be improved automatically (best={best:.3f} >= {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "CPU_HOTSPOT_NO_VALID_MOVE":
        return (
            f"{severity} - CPU hotspot {max_cpu:.2f}% "
            f"but no effective migration (best={best:.3f} < {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "CPU_HOTSPOT_OPTIMIZABLE":
        return (
            f"{severity} - CPU hotspot {max_cpu:.2f}% "
            f"and can be improved automatically (best={best:.3f} >= {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "PRESSURE_NO_VALID_MOVE":
        return (
            f"{severity} - resource pressure but no effective migration "
            f"(best={best:.3f} < {threshold:.3f})"
            f"{hot_str}"
        )

    if status in ("RAM_PRESSURE", "CPU_PRESSURE"):
        return (
            f"{severity} - resource pressure and can be improved automatically "
            f"(best={best:.3f} >= {threshold:.3f})"
            f"{hot_str}"
        )

    if status == "UNBALANCED_NO_VALID_MOVE":
        return (
            f"{severity} - cluster unbalanced but no effective migration "
            f"(best={best:.3f} < {threshold:.3f})"
        )

    if status == "UNBALANCED":
        return (
            f"{severity} - cluster unbalanced and can be improved automatically "
            f"(best={best:.3f} >= {threshold:.3f})"
        )

    if status == "BALANCED_OPTIMIZABLE":
        return (
            f"{severity} - cluster balanced with small optimization available "
            f"(best={best:.3f} >= {threshold:.3f})"
        )

    return f"{severity} - cluster balanced"


def build_detail_lines(state):
    ps = state.placement_summary or {}
    lines = [
        f"Mode: {format_mode_label(state.mode)}",
        f"Peak: CPU={state.max_cpu:.2f}% | RAM={state.max_ram:.2f}%",
        "",
        f"Cluster: CPU avg={state.cpu_mean:.2f}% | RAM avg={state.ram_mean:.2f}%",
        (
            f"Spread : CPU MAD={state.baseline_cpu_mad:.2f}% | "
            f"RAM MAD={state.baseline_ram_mad:.2f}% | "
            f"Weighted MAD={state.baseline_weighted_mad:.2f}"
        ),
        f"Priority: CPU {state.cpu_weight*100:.0f}% | RAM {state.ram_weight*100:.0f}%",
        (
            "Placement: "
            f"VCPU>={_config.runtime().placement_pressure_threshold:g}% hosts={ps.get('vcpu_pressure_hosts', 0)} | "
            f"MEM>={_config.runtime().placement_pressure_threshold:g}% hosts={ps.get('memory_pressure_hosts', 0)} | "
            f"VCPU safe-full={ps.get('vcpu_full_hosts', 0)} | "
            f"MEM safe-full={ps.get('memory_full_hosts', 0)}"
        ),
        "",
        f"Status: {'can be improved automatically' if state.has_valid_move else 'cannot be fixed automatically'}",
        format_candidate_line(state),
    ]

    if state.reason:
        lines.append(f"Reason: {format_reason(state.reason, state)}")

    load = state.load_status
    if load:
        lines.append(f"Load status: {format_cluster_status(load['detail_status'])} ({load['severity']})")
    lines.extend(format_soft_aa_lines(state))

    reject_stats = state.reject_stats or {}
    if reject_stats:
        ranked = sorted(reject_stats.items(), key=lambda item: (-item[1], item[0]))
        lines.append(
            "Rejected candidate pairs: " + ", ".join(f"{k}={v}" for k, v in ranked[:8])
        )

    # Surface the hosts that looked attractive at runtime but were most often
    # rejected as direct destinations by Placement capacity.
    dest_reject = state.dest_reject_stats or {}
    capacity_rows = []
    for host, reasons in dest_reject.items():
        count = reasons.get("placement_vcpu", 0) + reasons.get("placement_memory", 0)
        if count:
            capacity_rows.append((count, host, reasons))
    for count, host, reasons in sorted(capacity_rows, reverse=True)[:3]:
        lines.append(
            f"  - {host}: Placement-capacity rejects={count} "
            f"(VCPU={reasons.get('placement_vcpu', 0)}, MEM={reasons.get('placement_memory', 0)})"
        )

    lines.extend(
        format_excluded_vm_lines(
            state.excluded_vms or [],
            len(state.all_vm_names or {}),
        )
    )

    return lines


def format_soft_aa_lines(state):
    """rc11: broken soft-anti-affinity groups and what the heal rung decided.

    Informational for unresolvable/deferred groups and a disabled/deferred
    rung: these lines never change the severity (DRS "could not fix"
    semantics without alert spam). "unresolvable" = structural (every host
    already holds a member); "deferred" = may clear by itself on a later run.
    """
    heal = state.soft_aa_heal or {}
    broken = heal.get("broken_groups") or []
    if not broken:
        return []
    head = f"Soft-anti-affinity: {len(broken)} broken group(s), {heal.get('violations', 0)} violation(s)"
    if heal.get("deferred") == "disabled":
        head += " | heal disabled (HEAL_SOFT_ANTI_AFFINITY=false)"
    elif heal.get("deferred") == "hotspot":
        head += " | heal deferred while a hotspot is handled"
    lines = [head]
    plan = heal.get("plan")
    if plan is not None:
        info = getattr(plan, "soft_aa_heal", None) or {}
        step = plan.steps[0]
        vm_name = (state.all_vm_names or {}).get(step.vm_id, step.vm_id)
        lines.append(
            f"  - heal {info.get('group_name') or info.get('group_id')}: {vm_name} "
            f"({step.src_host} -> {step.dst_host}), group violations "
            f"{info.get('group_violations_before')} -> {info.get('group_violations_after')}, "
            f"weighted MAD {plan.baseline_weighted_mad:.2f} -> {plan.final_weighted_mad:.2f}"
        )
    for label, key in (("unresolvable", "unresolvable"), ("deferred", "deferred_groups")):
        for entry in heal.get(key) or []:
            lines.append(
                f"  - {entry.get('group_name') or entry.get('group_id')} "
                f"({entry.get('violations')} violation(s)): {label}: {entry.get('reason')}"
            )
    return lines


def build_monitor_message(state):
    lines = [build_headline(state), ""]
    lines.extend(build_detail_lines(state))
    return "\n".join(lines)


def print_state_summary(state):
    print(build_monitor_message(state))
    print()


def print_move_details(state, migration_round):
    plan = state.best_plan
    if not plan or not state.has_valid_move:
        return

    print(
        f"[Round {migration_round}] [{format_cluster_status(state.detail_status)}] "
        f"Plan={plan.plan_type.upper()} steps={len(plan.steps)}"
    )
    for idx, step in enumerate(plan.steps, 1):
        vm_name = state.all_vm_names.get(step.vm_id, step.vm_id)
        print(
            f"   Step {idx}: {vm_name} ({step.src_host} -> {step.dst_host}) "
            f"reason={step.reason}"
        )
        print(
            f"      Flavor need: {step.vcpus:.0f} VCPU, {step.vm_ram_gb:.2f} GB RAM | "
            f"Observed impact basis: {step.real_vm_cores:.2f} CPU cores, {step.real_vm_ram:.2f} GB RAM"
        )
        print(
            f"      Destination Placement safe-free before step: "
            f"{step.dst_free_vcpu:.2f} VCPU, {step.dst_free_ram_gb:.2f} GB"
        )
        print(
            f"      Runtime impact: CPU src -{step.cpu_impact_src:.2f}% / dst +{step.cpu_impact_dst:.2f}% | "
            f"RAM src -{step.ram_impact_src:.2f}% / dst +{step.ram_impact_dst:.2f}%"
        )

    new_cpu_mean = sum(plan.final_host_cpu.values()) / len(plan.final_host_cpu)
    new_ram_mean = sum(plan.final_host_ram.values()) / len(plan.final_host_ram)
    print(f"   Search scope: {plan.search_scope}")
    print(f"   Final means: CPU={new_cpu_mean:.2f}% RAM={new_ram_mean:.2f}%")
    print(
        f"   Final MAD: CPU={plan.mad_cpu:.2f} RAM={plan.mad_ram:.2f} "
        f"Weighted={plan.final_weighted_mad:.2f}"
    )
    heal = getattr(plan, "soft_aa_heal", None)
    if heal:
        print(
            f"   Soft-anti-affinity heal: group {heal.get('group_name') or heal.get('group_id')} "
            f"violations {heal.get('group_violations_before')} -> {heal.get('group_violations_after')}"
        )
        cpu_limit = heal.get("cpu_mad_limit")
        ram_limit = heal.get("ram_mad_limit")
        if cpu_limit is None or ram_limit is None:
            cpu_limit = _planning_balance.soft_aa_heal_mad_limit(heal.get("cpu_mad_before", 0.0))
            ram_limit = _planning_balance.soft_aa_heal_mad_limit(heal.get("ram_mad_before", 0.0))
        print(
            f"   Heal MAD gate: CPU {plan.mad_cpu:.2f} <= {cpu_limit:.2f}, RAM {plan.mad_ram:.2f} <= "
            f"{ram_limit:.2f} (per resource: TARGET_AVG_MAD {_config.runtime().target_avg_mad:g}, or before + "
            f"{_config.runtime().soft_aa_heal_mad_tolerance:g} when already above it)"
        )
    print(
        f"   Improvement: {plan.baseline_weighted_mad:.3f} - "
        f"{plan.final_weighted_mad:.3f} = {plan.improvement:.3f}\n"
    )


# Classification fields an audit event takes from the snapshot its plan was
# made from (see plan_event_labels): the fields of plan_context.PlanLabels.
PLAN_EVENT_LABEL_KEYS = ("mode", "detail_status", "severity", "best_improvement", "improvement_threshold")


def plan_event_labels(state):
    """Event labels (a ``PlanLabels``) of the snapshot a plan was MADE from.

    A snapshot refreshed for revalidation is evaluated with
    ``plan_balance=False``: it searches no move, so its own classification
    is "no technical candidate" (CRITICAL). Events of the executed plan must
    keep the planning verdict instead; labels already carried by ``state``
    (``event_labels``) win, so they survive any number of refreshes.
    """
    if state.event_labels is not None:
        return state.event_labels
    return _plan_context.PlanLabels.of(state)


def build_migration_event(
    state,
    result,
    duration_seconds=None,
    started_at=None,
    ended_at=None,
    detail=None,
    step=None,
    plan=None,
    step_index=None,
):
    plan = plan or state.best_plan or state.best_plan_candidate
    labels = plan_event_labels(state)
    event = {
        "event_time": format_migration_timestamp(ended_at or datetime.now()),
        "event_ts": time.time(),
        "aggregate": state.aggregate,
        "result": result,
        "detail_status": labels.detail_status,
        "severity": labels.severity,
        "duration_seconds": duration_seconds,
        "started_at": format_migration_timestamp(started_at),
        "ended_at": format_migration_timestamp(ended_at),
        "best_improvement": labels.best_improvement,
        "improvement_threshold": labels.improvement_threshold,
    }
    if labels.mode is not None:
        event["mode"] = labels.mode

    if plan:
        event["plan_type"] = plan.plan_type
        event["plan_improvement"] = plan.improvement
        event["plan_steps"] = [
            {
                "vm_id": s.vm_id,
                "src_host": s.src_host,
                "dst_host": s.dst_host,
                "reason": s.reason,
            }
            for s in plan.steps
        ]
        heal = getattr(plan, "soft_aa_heal", None)
        if heal:
            event["mode"] = _config.MODE_SOFT_AA_HEAL
            event["soft_aa_heal"] = dict(heal)
    if step_index is not None:
        event["step_index"] = step_index
    if detail:
        event["detail"] = detail

    if step is None and plan and len(plan.steps) == 1:
        step = plan.steps[0]
    if step is not None:
        vm_detail = (state.all_vm_details or {}).get(step.vm_id, {})
        event.update(
            {
                "vm_id": step.vm_id,
                "vm_name": (state.all_vm_names or {}).get(step.vm_id, step.vm_id),
                "project_id": vm_detail.get("project_id"),
                "project_name": vm_detail.get("project_name"),
                "src_host": step.src_host,
                "dst_host": step.dst_host,
                "step_reason": step.reason,
            }
        )

    return event


def format_migration_event_line(event):
    return json.dumps(event, ensure_ascii=False, sort_keys=True)


def append_migration_event(event, filename=None):
    from compute_maintenance.adapters.protocol import record_event
    record_event(event)
    if filename is None:
        filename = event_path.get() or _config.MIGRATION_EVENTS_FILE

    try:
        directory = os.path.dirname(filename) or "."
        os.makedirs(directory, exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            f.write(format_migration_event_line(event) + "\n")
        logging.debug("Recorded migration event to %s: %s", filename, event.get("result"))
    except Exception as e:  # noqa: BLE001 - the audit log is best effort; the run goes on (logged)
        logging.warning("Failed to record migration event to %s: %s", filename, e)


def record_soft_aa_heal_decision(state):
    """rc11: log the heal rung's decision of a balance run. Groups without a
    heal move are also recorded: ``soft_aa_heal_unresolvable`` for the
    structural case (every host already holds a member) and
    ``soft_aa_heal_deferred`` for temporary causes, with the blocking counts.
    The heal move itself is recorded by the normal dry_run/migration events."""
    heal = state.soft_aa_heal or {}
    broken = heal.get("broken_groups") or []
    if not broken:
        return
    agg = state.aggregate
    deferred = heal.get("deferred")
    if deferred:
        _log.info(
            "Aggregate %s: soft_aa_heal: %d broken soft-anti-affinity group(s), %d violation(s); %s.",
            agg, len(broken), heal.get("violations", 0),
            "heal disabled (HEAL_SOFT_ANTI_AFFINITY=false)" if deferred == "disabled"
            else "heal deferred while a hotspot is handled",
        )
        return
    for label, key in (("unresolvable", "unresolvable"), ("deferred", "deferred_groups")):
        for entry in heal.get(key) or []:
            _log.info(
                "Aggregate %s: soft_aa_heal: group %s (%d violation(s)) %s: %s",
                agg, entry.get("group_name") or entry.get("group_id"), entry.get("violations", 0),
                label, entry.get("reason"),
            )
    plan = heal.get("plan")
    if plan is not None:
        info = getattr(plan, "soft_aa_heal", None) or {}
        step = plan.steps[0]
        _log.info(
            "Aggregate %s: soft_aa_heal: group %s (%s violation(s)) -> move %s %s -> %s "
            "(weighted MAD %.3f -> %.3f)",
            agg, info.get("group_name") or info.get("group_id"), info.get("group_violations_before"),
            (state.all_vm_names or {}).get(step.vm_id, step.vm_id), step.src_host, step.dst_host,
            plan.baseline_weighted_mad, plan.final_weighted_mad,
        )
    for label, key in (("unresolvable", "unresolvable"), ("deferred", "deferred_groups")):
        entries = heal.get(key) or []
        if not entries:
            continue
        append_migration_event({
            "event_time": format_migration_timestamp(datetime.now()),
            "event_ts": time.time(),
            "aggregate": agg,
            "result": "soft_aa_heal_" + label,
            "mode": _config.MODE_SOFT_AA_HEAL,
            "detail_status": state.detail_status,
            "severity": state.severity,
            "detail": "; ".join(
                "{}: {}: {}".format(e.get("group_name") or e.get("group_id"), label, e.get("reason"))
                for e in entries
            ),
            "soft_aa_violations": heal.get("violations", 0),
            "soft_aa_" + label: [dict(e) for e in entries],
        })


def format_drain_vm_label(state, vm_id):
    name = (state.all_vm_names or {}).get(vm_id, vm_id)
    if name == vm_id:
        return vm_id
    return f"{name} ({vm_id})"


def print_drain_report(state, plan, allow_hotspot):
    """Operator-facing report for one drain evaluation."""
    print(f"\n=== DRAIN PLAN: {plan.host} (aggregate {plan.aggregate}) ===")
    cpu_cap, ram_cap = _planning_drain.drain_host_caps(allow_hotspot)
    print(
        "Receiving hosts: {} | caps: CPU <= {:.1f}%, RAM <= {:.1f}%{}".format(
            ", ".join(plan.remaining_hosts) or "none",
            cpu_cap,
            ram_cap,
            " (--drain-allow-hotspot)" if allow_hotspot else "",
        )
    )

    if plan.capacity:
        cap = plan.capacity
        print(
            "Capacity budget: need {:.0f} vCPU / {:.1f} GiB RAM; "
            "safe-free {:.0f} vCPU / {:.1f} GiB RAM".format(
                cap["required_vcpu"],
                cap["required_ram_mb"] / 1024.0,
                cap["available_vcpu"],
                cap["available_ram_mb"] / 1024.0,
            )
        )

    if plan.steps:
        print(
            "\nMigration steps ({} VM(s), packing order {}):".format(
                len(plan.steps), plan.ordering or "n/a"
            )
        )
        for idx, step in enumerate(plan.steps, 1):
            print(
                f"   {idx:>3}. {format_drain_vm_label(state, step.vm_id)} -> {step.dst_host} | flavor {step.vcpus:.0f} vCPU / {step.vm_ram_gb:.1f} GiB | "
                f"dst RAM {state.host_ram_pct[step.dst_host]:.1f}% -> {state.host_ram_pct[step.dst_host] + step.ram_impact_dst:.1f}%"
            )
    else:
        print("\nMigration steps: none")

    if plan.remaining_hosts:
        print("\nProjected end state (drained host excluded from balance):")
        print("   {:<34} {:>12} {:>12}".format("HOST", "CPU%", "RAM%"))
        for host in plan.remaining_hosts:
            print(
                f"   {host:<34} {state.host_cpu_pct[host]:>5.1f} -> {plan.final_host_cpu[host]:>4.1f} {state.host_ram_pct[host]:>5.1f} -> {plan.final_host_ram[host]:>4.1f}"
            )
        print(
            f"   weighted MAD {plan.baseline_weighted_mad:.3f} -> {plan.final_weighted_mad:.3f} (lower is better balanced)"
        )

    if plan.hotspot_hosts:
        print("\nWARNING: host(s) above the balance cap after drain:")
        for entry in plan.hotspot_hosts:
            print(
                "   {} CPU {:.1f}% RAM {:.1f}%".format(
                    entry["host"], entry["cpu_pct"], entry["ram_pct"]
                )
            )

    if plan.affinity_conflicts:
        print(
            f"\nAffinity group(s) co-located on {plan.host} (cannot be evacuated one VM "
            "at a time):"
        )
        for entry in plan.affinity_conflicts:
            print(
                "   {} -> {} member(s) on this host: {}".format(
                    entry["group_name"] or entry["group_id"],
                    len(entry["vm_ids"]),
                    ", ".join(
                        format_drain_vm_label(state, vm_id)
                        for vm_id in entry["vm_ids"]
                    ),
                )
            )

    if plan.soft_policy_warnings:
        print(
            f"\nSoft (anti-)affinity cannot be fully kept ({len(plan.soft_policy_warnings)} VM(s)); least-violating "
            "destination chosen:"
        )
        for entry in plan.soft_policy_warnings:
            print(
                "   {} -> {} ({} soft-policy conflict(s))".format(
                    format_drain_vm_label(state, entry["vm_id"]), entry["dst_host"], entry["penalty"]
                )
            )

    compensation = state.arrival_compensation or {}
    if compensation:
        print("\nLoad added for VMs that arrived within the metric window (not yet in Prometheus):")
        for host, entry in sorted(compensation.items()):
            print("   {} +{:.1f}% CPU +{:.1f}% RAM ({} VM(s), {})".format(
                host, entry["cpu"], entry["ram"], entry["vms"], entry.get("method", "model")))

    if plan.unmovable:
        print(
            f"\nNOT live-migratable ({len(plan.unmovable)} VM(s)) - handle manually "
            "(cold migrate / shut down):"
        )
        for entry in plan.unmovable:
            print(
                "   {} reason={}".format(
                    format_drain_vm_label(state, entry["vm_id"]), entry["reason"]
                )
            )

    if plan.blocked:
        print(f"\nNO DESTINATION for {len(plan.blocked)} VM(s):")
        for entry in plan.blocked:
            print(
                "   {} ({:.0f} vCPU / {:.1f} GiB)".format(
                    format_drain_vm_label(state, entry["vm_id"]),
                    entry["vcpus"],
                    entry["ram_gb"],
                )
            )
            for host, reason in sorted(entry["reasons"].items()):
                print(f"      {host} -> {reason}")

    if plan.infeasible_reasons:
        print("\nDRAIN NOT POSSIBLE:")
        for reason in plan.infeasible_reasons:
            print(f"   - {reason}")
    elif plan.unmovable:
        print(
            f"\nDRAIN FEASIBLE for all live-migratable VMs, but {len(plan.unmovable)} VM(s) will "
            f"remain on {plan.host}."
        )
    else:
        print(f"\nDRAIN FEASIBLE: {plan.host} will be fully evacuated.")
    print()


def build_drain_event(state, plan, result, detail=None, migration_plan=None):
    """Audit record for a drain decision (not for an individual migration)."""
    # Always pass an explicit drain plan: build_migration_event() otherwise
    # falls back to the aggregate's *balancing* plan, which is unrelated to
    # the drain decision being recorded.
    if migration_plan is None:
        migration_plan = _planning_drain.drain_plan_to_migration_plan(plan)
    event = build_migration_event(
        state,
        result,
        detail=detail,
        plan=migration_plan,
    )
    event["drain_host"] = plan.host
    event["drain_feasible"] = plan.feasible
    event["drain_step_count"] = len(plan.steps)
    event["drain_blocked_vm_ids"] = [e["vm_id"] for e in plan.blocked]
    event["drain_unmovable_vm_ids"] = [e["vm_id"] for e in plan.unmovable]
    event["drain_infeasible_reasons"] = list(plan.infeasible_reasons)
    event["drain_capacity"] = plan.capacity
    event["drain_affinity_conflicts"] = plan.affinity_conflicts or []
    event["drain_soft_policy_warnings"] = plan.soft_policy_warnings or []
    event["arrival_compensation"] = state.arrival_compensation or {}
    return event
