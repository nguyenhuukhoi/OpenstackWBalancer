"""Immutable input values; simulations explicitly copy before modifying load."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass


class FrozenMap(Mapping):
    def __init__(self, values):
        self._values = {key: freeze(value) for key, value in values.items()}

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def copy(self):
        return dict(self._values)

    def __deepcopy__(self, memo):
        # Existing planners request deepcopy specifically for mutable simulation.
        return {key: deepcopy(value, memo) for key, value in self._values.items()}


def freeze(value):
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    return value


@dataclass(frozen=True)
class ClusterSnapshot:
    aggregate: str
    collected_at: float
    host_cpu: Mapping
    host_ram: Mapping
    vm_hosts: Mapping
    raw_metric_ages: Mapping


def freeze_context_inputs(context):
    """Keep diagnostic counters writable; cluster input maps are read-only."""
    for name in ("effective_hosts", "host_cpu_pct", "host_ram_pct", "host_cores", "host_ram",
                 "host_free_cores", "host_free_ram", "host_placement", "vm_host_map", "all_vm_details",
                 "vm_to_groups", "dest_ready_hosts", "host_running_vms", "host_max_instances",
                 "excluded_vm_ids", "vm_status_map", "cooldown_remaining", "host_az"):
        setattr(context, name, freeze(getattr(context, name)))
    return context
