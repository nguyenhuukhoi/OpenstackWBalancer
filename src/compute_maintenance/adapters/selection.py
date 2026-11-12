"""selection — extracted domain responsibilities."""
import fnmatch
import logging
from collections import defaultdict

import compute_maintenance.domain.errors as _errors


def load_vm_list_file(filename, label="VM list"):
    """Read a VM list file and return its entries (one per line).

    Blank lines and lines starting with '#' are ignored. Each remaining line is
    a VM UUID, an exact VM name, or a glob pattern matched against the VM name.
    A missing file is fatal: silently ignoring the VMs an operator explicitly
    listed would be worse than failing loudly on a wrong path.

    Shared by the exclude list here and by the instance splitter, so both read
    exactly the same format.
    """
    if not filename or not filename.strip():
        return []

    filename = filename.strip()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except OSError as e:
        raise _errors.ConfigError(f"Failed to read {label} file {filename}: {e}")

    patterns = []
    seen = set()
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry not in seen:
            seen.add(entry)
            patterns.append(entry)

    logging.info(
        "Loaded %d %s entr(ies) from %s: %s",
        len(patterns),
        label,
        filename,
        ", ".join(patterns) if patterns else "(none)",
    )
    return patterns


def load_exclude_vm_patterns(filename):
    """Read the exclude file (see load_vm_list_file for the format)."""
    return load_vm_list_file(filename, label="VM exclude")


def match_vm_pattern(vm_id, vm_name, patterns):
    """Return the first list entry matching this VM, or None.

    An entry matches on exact VM id, exact VM name, or a case-sensitive glob
    against the name (OpenStack names are case-sensitive, so matching is too).
    """
    for pattern in patterns:
        if pattern == vm_id or (vm_name and pattern == vm_name):
            return pattern
        if vm_name and fnmatch.fnmatchcase(vm_name, pattern):
            return pattern
    return None


def match_exclude_pattern(vm_id, vm_name, exclude_patterns):
    """Return the first exclude entry matching this VM, or None."""
    return match_vm_pattern(vm_id, vm_name, exclude_patterns)


def build_excluded_vm_map(all_vm_names, exclude_patterns):
    """Resolve the exclude entries against the aggregate VM inventory.

    Returns {vm_id: matching exclude entry} for every VM that must never be
    proposed for migration. Those VMs are still part of the host load used for
    scoring, so the cluster picture stays accurate; only their eligibility as a
    move candidate is removed. Keeping the matching entry (not just the id) lets
    the report show which exclude line pinned which VM.
    """
    if not exclude_patterns:
        return {}

    excluded = {}
    for vm_id, vm_name in all_vm_names.items():
        pattern = match_exclude_pattern(vm_id, vm_name, exclude_patterns)
        if pattern:
            excluded[vm_id] = pattern
            logging.debug(
                "VM %s (%s) excluded from migration by exclude entry %r.",
                vm_name, vm_id, pattern,
            )

    return excluded


def build_excluded_vm_report(excluded_vm_map, all_vm_names):
    """Flatten the exclude map into report rows sorted by entry, then VM name."""
    return [
        {
            "vm_id": vm_id,
            "vm_name": all_vm_names.get(vm_id, vm_id),
            "pattern": pattern,
        }
        for vm_id, pattern in sorted(
            excluded_vm_map.items(),
            key=lambda item: (item[1], all_vm_names.get(item[0], item[0])),
        )
    ]


def group_excluded_vms_by_pattern(excluded_vms):
    """Group report rows into {exclude entry: [vm names]}, preserving order."""
    grouped = defaultdict(list)
    for row in excluded_vms:
        grouped[row["pattern"]].append(row["vm_name"])
    return grouped
