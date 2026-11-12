import logging

import compute_maintenance.domain.models as _models
from compute_maintenance.domain.models import get_resource_field

"""planning.constraints — extracted domain responsibilities."""
import math

import compute_maintenance.config as _config
import compute_maintenance.domain.errors as _errors


def build_placement_inventory_record(inv, used, headroom_percent):
    total_raw = _models.get_resource_field(inv, "total")
    reserved_raw = _models.get_resource_field(inv, "reserved")
    ratio_raw = _models.get_resource_field(inv, "allocation_ratio")
    min_unit_raw = _models.get_resource_field(inv, "min_unit")
    max_unit_raw = _models.get_resource_field(inv, "max_unit")
    step_size_raw = _models.get_resource_field(inv, "step_size")

    total = float(0 if total_raw is None else total_raw)
    reserved = float(0 if reserved_raw is None else reserved_raw)
    ratio = float(1.0 if ratio_raw is None else ratio_raw)
    min_unit = int(1 if min_unit_raw is None else min_unit_raw)
    max_unit = int(max(int(total), 1) if max_unit_raw is None else max_unit_raw)
    step_size = int(1 if step_size_raw is None else step_size_raw)
    used = float(used or 0)

    numeric_values = (total, reserved, ratio, used, float(headroom_percent))
    if not all(math.isfinite(value) for value in numeric_values):
        raise _errors.PlacementError("Placement inventory contains a non-finite numeric value")
    if total < 0 or reserved < 0 or ratio <= 0 or used < 0:
        raise _errors.PlacementError(
            "Placement inventory contains negative capacity/usage or a non-positive allocation ratio"
        )
    if min_unit <= 0 or max_unit < min_unit or step_size <= 0:
        raise _errors.PlacementError(
            "Placement inventory has invalid min_unit/max_unit/step_size constraints"
        )

    nominal_capacity = max((total - reserved) * ratio, 0.0)
    policy_capacity = max(
        nominal_capacity * (1.0 - (float(headroom_percent) / 100.0)),
        0.0,
    )
    return {
        "total": total,
        "reserved": reserved,
        "allocation_ratio": ratio,
        "min_unit": min_unit,
        "max_unit": max_unit,
        "step_size": step_size,
        "nominal_capacity": nominal_capacity,
        "policy_capacity": policy_capacity,
        "used": used,
        "free": max(policy_capacity - used, 0.0),
        "nominal_free": max(nominal_capacity - used, 0.0),
        "allocation_pct": (used / nominal_capacity * 100.0) if nominal_capacity > 0 else 100.0,
        "headroom_percent": float(headroom_percent),
    }


def placement_capacity_status(placement):
    v = placement["VCPU"]
    m = placement["MEMORY_MB"]
    v_full = v["free"] <= 0
    m_full = m["free"] <= 0
    v_pressure = v["allocation_pct"] >= _config.runtime().placement_pressure_threshold
    m_pressure = m["allocation_pct"] >= _config.runtime().placement_pressure_threshold

    if v_full and m_full:
        return "ALLOCATION_FULL"
    if v_full:
        return "VCPU_FULL"
    if m_full:
        return "RAM_ALLOCATION_FULL"
    if v_pressure and m_pressure:
        return "ALLOCATION_PRESSURE"
    if v_pressure:
        return "VCPU_PRESSURE"
    if m_pressure:
        return "RAM_ALLOCATION_PRESSURE"
    return "CAPACITY_OK"


def inventory_request_valid(inv, amount):
    amount = int(amount)
    if amount < inv["min_unit"]:
        return False, "below_min_unit"
    if amount > inv["max_unit"]:
        return False, "above_max_unit"
    if inv["step_size"] > 1 and amount % inv["step_size"] != 0:
        return False, "step_size"
    return True, None


def placement_can_fit(placement, vcpus, ram_mb, usage_delta=None):
    """Check a VM flavor against one host's Placement inventory and safe headroom."""
    usage_delta = usage_delta or {}
    requests = {
        "VCPU": int(vcpus),
        "MEMORY_MB": int(ram_mb),
    }

    for rc, amount in requests.items():
        inv = placement[rc]
        valid, constraint_reason = inventory_request_valid(inv, amount)
        if not valid:
            return False, f"placement_{rc.lower()}_{constraint_reason}"

        effective_used = inv["used"] + float(usage_delta.get(rc, 0.0))
        free = inv["policy_capacity"] - effective_used
        if free + 1e-9 < amount:
            if rc == "VCPU":
                return False, "placement_vcpu"
            return False, "placement_memory"

    return True, None


def build_server_group_index(vm_ids, server_groups):
    vm_ids = set(vm_ids)
    vm_to_groups = {vm_id: [] for vm_id in vm_ids}

    if server_groups is None:
        raise _errors.OpenStackDataError(
            "Server-group enforcement is enabled but no server-group snapshot is available."
        )

    for sg in server_groups:
        members = list(get_resource_field(sg, "member_ids", "members") or [])
        if not members:
            continue

        if not any(m in vm_ids for m in members):
            continue

        policies = get_resource_field(sg, "policies")
        if policies and isinstance(policies, (list, tuple, set)):
            policy = next(iter(policies))
        else:
            policy = get_resource_field(sg, "policy")

        rules = get_resource_field(sg, "rules") or {}
        info = {
            "id": get_resource_field(sg, "id"),
            "name": get_resource_field(sg, "name"),
            "policy": policy,
            "members": members,
            "rules": dict(rules) if isinstance(rules, dict) else {},
        }

        for m in members:
            if m in vm_ids:
                vm_to_groups[m].append(info)

    logging.info(
        "Built server group index for %d VMs (some may have 0 groups).",
        len(vm_to_groups),
    )
    return vm_to_groups


def max_server_per_host(group):
    """Nova anti-affinity rule ``max_server_per_host`` (default 1)."""
    raw = (group.get("rules") or {}).get("max_server_per_host", 1)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return max(value, 1)


def _other_member_hosts(vm_id, group, vm_host_map):
    """Hosts of the group's other members (unknown placements ignored)."""
    hosts = []
    for member in group.get("members") or []:
        if member == vm_id:
            continue
        host = vm_host_map.get(member)
        if host is not None:
            hosts.append(host)
    return hosts


def is_server_group_move_compliant(vm_id, src_host, dst_host, vm_host_map, vm_to_groups):
    """Hard server-group check with Nova filter semantics.

    Only the destination is judged, exactly like ServerGroupAffinityFilter /
    ServerGroupAntiAffinityFilter: a group that is already split or already
    over its limit elsewhere does not block a move that does not make the
    destination worse.

    * affinity: the destination must already host another member (any of
      them when the group is already split), or no other member is placed.
    * anti-affinity: other members on the destination must stay below
      ``rules.max_server_per_host`` (default 1).
    """
    groups = vm_to_groups.get(vm_id) or []
    if not groups:
        return True

    for g in groups:
        policy = g.get("policy")
        members = g.get("members") or []
        if not policy or not members:
            continue

        others = _other_member_hosts(vm_id, g, vm_host_map)
        if policy == "affinity":
            if others and dst_host not in set(others):
                logging.debug(
                    "Reject move of %s %s->%s: affinity group %s/%s lives on %s.",
                    vm_id, src_host, dst_host, g.get("name"), g.get("id"), sorted(set(others)),
                )
                return False

        elif policy == "anti-affinity":
            on_destination = sum(1 for host in others if host == dst_host)
            if on_destination >= max_server_per_host(g):
                logging.debug(
                    "Reject move of %s %s->%s: anti-affinity group %s/%s already has %d member(s) on %s "
                    "(max_server_per_host=%d).",
                    vm_id, src_host, dst_host, g.get("name"), g.get("id"), on_destination, dst_host,
                    max_server_per_host(g),
                )
                return False

    return True


def soft_group_penalty(vm_id, host, vm_host_map, vm_to_groups):
    """How much placing ``vm_id`` on ``host`` violates its soft groups.

    Nova only weighs soft policies; with an explicit live-migration target
    they are not enforced at all, so the planner has to respect them itself.

    * soft-anti-affinity: one point per other member already on ``host``.
    * soft-affinity: one point when other members exist but none is on ``host``.
    """
    penalty = 0
    for g in vm_to_groups.get(vm_id) or []:
        policy = g.get("policy")
        if policy not in ("soft-anti-affinity", "soft-affinity"):
            continue
        others = _other_member_hosts(vm_id, g, vm_host_map)
        if policy == "soft-anti-affinity":
            penalty += sum(1 for other in others if other == host)
        elif others and host not in set(others):
            penalty += 1
    return penalty


def soft_group_move_delta(vm_id, src_host, dst_host, vm_host_map, vm_to_groups):
    """Change in soft-group violation if ``vm_id`` moves src -> dst (>0 = worse)."""
    return (soft_group_penalty(vm_id, dst_host, vm_host_map, vm_to_groups)
            - soft_group_penalty(vm_id, src_host, vm_host_map, vm_to_groups))


SOFT_ANTI_AFFINITY = "soft-anti-affinity"


def _group_key(group):
    """Stable identity of a server group (frozen copies are distinct objects)."""
    if group.get("id") is not None:
        return group.get("id")
    if group.get("name") is not None:
        return group.get("name")
    return tuple(group.get("members") or ())


def soft_anti_affinity_members_on(vm_id, host, vm_host_map, vm_to_groups):
    """Other members of ``vm_id``'s soft-anti-affinity group(s) placed on ``host``."""
    count = 0
    for g in vm_to_groups.get(vm_id) or []:
        if g.get("policy") == SOFT_ANTI_AFFINITY:
            count += sum(1 for other in _other_member_hosts(vm_id, g, vm_host_map) if other == host)
    return count


def soft_anti_affinity_move_reduction(vm_id, src_host, dst_host, vm_host_map, vm_to_groups):
    """Soft-anti-affinity violations removed by moving ``vm_id`` src -> dst.

    Violations of a group are ``sum(max(0, members_on_host - 1))`` over hosts,
    so one move removes a violation when another member stays on the source
    and adds one when the destination already holds a member (>0 = heals).
    """
    reduction = 0
    for g in vm_to_groups.get(vm_id) or []:
        if g.get("policy") != SOFT_ANTI_AFFINITY:
            continue
        others = set(_other_member_hosts(vm_id, g, vm_host_map))
        if src_host in others:
            reduction += 1
        if dst_host in others:
            reduction -= 1
    return reduction


def broken_soft_anti_affinity_groups(vm_host_map, vm_to_groups, hosts):
    """Soft-anti-affinity groups with at least one violation on ``hosts``.

    Only VMs present in the snapshot (``vm_host_map``) and placed on one of
    ``hosts`` are counted; violations = sum over hosts of max(0, members - 1).
    Sorted so the group whose worst host holds the most members comes first.
    """
    hosts = set(hosts)
    found = {}
    for vm_id, host in vm_host_map.items():
        if host not in hosts:
            continue
        for g in vm_to_groups.get(vm_id) or []:
            if g.get("policy") != SOFT_ANTI_AFFINITY:
                continue
            entry = found.setdefault(_group_key(g), {"group": g, "members_by_host": {}})
            members = entry["members_by_host"].setdefault(host, [])
            if vm_id not in members:
                members.append(vm_id)

    broken = []
    for key, entry in found.items():
        by_host = {host: sorted(vms) for host, vms in entry["members_by_host"].items()}
        violations = sum(max(0, len(vms) - 1) for vms in by_host.values())
        if violations <= 0:
            continue
        group = entry["group"]
        broken.append({
            "id": group.get("id"),
            "name": group.get("name"),
            "key": key,
            "violations": violations,
            "worst_count": max(len(vms) for vms in by_host.values()),
            "members_by_host": by_host,
        })
    broken.sort(key=lambda e: (-e["worst_count"], -e["violations"], str(e["name"] or e["id"] or e["key"])))
    return broken
