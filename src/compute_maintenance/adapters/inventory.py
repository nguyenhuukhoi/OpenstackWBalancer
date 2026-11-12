"""maint.inventory — extracted domain responsibilities."""
import json
import os
import shutil
import sys

import compute_maintenance.adapters.commands as _maint_remote
import compute_maintenance.domain.maintenance_errors as _maint_errors
import compute_maintenance.profiles.defaults as _maint_settings


def ansible_inventory_bin():
    """ansible-inventory next to this interpreter (activated or not), else PATH."""
    local = os.path.join(os.path.dirname(sys.executable), "ansible-inventory")
    return local if os.access(local, os.X_OK) else (shutil.which("ansible-inventory") or "ansible-inventory")


def group_hosts(data, group, seen=None):
    """Hosts of ``group`` incl. child groups, in inventory order, no duplicates."""
    seen = set() if seen is None else seen
    out = []
    node = data.get(group) or {}
    for h in node.get("hosts", []):
        if h not in out:
            out.append(h)
    for child in node.get("children", []):
        if child in seen:
            continue
        seen.add(child)
        for h in group_hosts(data, child, seen):
            if h not in out:
                out.append(h)
    return out


def load_inventory(path, group=_maint_settings.INVENTORY_GROUP):
    """Return [(host, hostvars)] of ``group`` using ansible-inventory itself,
    so INI/YAML syntax, :children and group_vars resolve like kolla-ansible."""
    try:
        p = _maint_remote.run([ansible_inventory_bin(), "-i", path, "--list"], timeout=120)
    except FileNotFoundError:
        raise _maint_errors.StepFailed("ansible-inventory not found: run from the kolla-ansible venv "
                         "(e.g. /opt/venv-2026.1/bin/python3 rolling_maint.py ...)") from None
    data = json.loads(p.stdout)
    hostvars = data.get("_meta", {}).get("hostvars", {})
    hosts = group_hosts(data, group)
    if not hosts:
        raise _maint_errors.StepFailed(f"inventory {path} has no host in group [{group}]")
    return [(h, hostvars.get(h, {})) for h in hosts]


def resolve_hosts(args, cloud=None):
    """Fill args.hosts and return {host: {address, user, key}}.

    Host selection, each given source narrows the set:
      explicit hosts   -> exactly these, in this order (must be in every source)
      --inventory      -> [compute] group, inventory order
      -a AGGREGATE     -> hosts of the Nova aggregate (also what wbalancer drains into)
    No explicit hosts: hosts in both inventory and aggregate, inventory order if an
    inventory is given, else sorted by name.

    Precedence per connection field: CLI flag > inventory hostvar > default.
    """
    inv = {}
    sources = []  # (label, ordered host list)
    if args.inventory:
        entries = load_inventory(args.inventory)
        inv = dict(entries)
        sources.append((f"[{_maint_settings.INVENTORY_GROUP}] of {args.inventory}", [h for h, _ in entries]))
    if args.aggregate and cloud is not None:
        members = cloud.aggregate_hosts(args.aggregate)
        if not members:
            raise _maint_errors.StepFailed(f"aggregate {args.aggregate} has no host")
        sources.append((f"aggregate {args.aggregate}", sorted(members)))

    if args.hosts:
        for label, members in sources:
            outside = [h for h in args.hosts if h not in members]
            if outside:
                raise _maint_errors.StepFailed("not in {}: {}".format(label, ", ".join(outside)))
    elif sources:
        hosts = sources[0][1]
        for label, members in sources[1:]:
            dropped = [h for h in hosts if h not in members]
            if dropped:
                _maint_settings.log.warning("not in %s, left out: %s", label, ", ".join(dropped))
            extra = [h for h in members if h not in hosts]
            if extra:
                _maint_settings.log.warning("in %s but not in %s, left out: %s", label, sources[0][0], ", ".join(extra))
            hosts = [h for h in hosts if h in members]
        if not hosts:
            raise _maint_errors.StepFailed("no host is in all of: " + "; ".join(label for label, _ in sources))
        args.hosts = hosts
    if not args.hosts:
        raise _maint_errors.StepFailed("no host given (list hosts, or use --inventory / -a)")
    if len(set(args.hosts)) != len(args.hosts):
        raise _maint_errors.StepFailed("duplicate host in list")
    cli_addr = dict(item.split("=", 1) for item in args.host_address)
    conn = {}
    for h in args.hosts:
        hv = inv.get(h, {})
        key = args.ssh_key or hv.get("ansible_ssh_private_key_file") or hv.get("ansible_private_key_file") \
            or _maint_settings.DEFAULT_SSH_KEY
        key = os.path.expanduser(key)
        if not os.path.isfile(key):
            # Explicit --ssh-key must exist; an inventory/default key that is not
            # there just falls back to ssh's own keys (~/.ssh/id_*, agent).
            if args.ssh_key:
                raise _maint_errors.StepFailed(f"--ssh-key {args.ssh_key} does not exist")
            if key != _maint_settings.DEFAULT_SSH_KEY:
                _maint_settings.log.warning("  %s: ssh key %s not found, using ssh's default keys", h, key)
            key = None
        conn[h] = {
            "address": cli_addr.get(h) or hv.get("ansible_host") or h,
            "user": args.ssh_user or hv.get("ansible_user") or "root",
            "key": key,
        }
    return conn
