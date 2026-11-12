"""models — extracted domain responsibilities."""
from collections import namedtuple
from dataclasses import dataclass

BestMove = namedtuple(
    "BestMove",
    [
        "vm_id",
        "src_host",
        "dst_host",
        "cpu_impact_src",
        "cpu_impact_dst",
        "ram_impact_src",
        "ram_impact_dst",
        "new_host_cpu",
        "new_host_ram",
        "real_vm_cores",
        "vcpus",
        "real_vm_ram",
        "vm_ram_gb",
        "dst_free_cores",
        "dst_free_ram_gb",
        "new_weighted_mad",
        "mad_improvement",
        "mad_cpu",
        "mad_ram",
    ],
)


@dataclass(frozen=True)
class MigrationStep:
    vm_id: str
    src_host: str
    dst_host: str
    reason: str
    vcpus: float
    vm_ram_gb: float
    real_vm_cores: float
    real_vm_ram: float
    cpu_impact_src: float
    cpu_impact_dst: float
    ram_impact_src: float
    ram_impact_dst: float
    dst_free_vcpu: float
    dst_free_ram_gb: float


@dataclass
class MigrationPlan:
    plan_type: str
    steps: list
    baseline_weighted_mad: float
    final_weighted_mad: float
    improvement: float
    mad_cpu: float
    mad_ram: float
    final_host_cpu: dict
    final_host_ram: dict
    search_scope: str
    # rc11: set on a soft-anti-affinity heal plan only (group and violation
    # counts before/after); carried into the audit events of its step.
    soft_aa_heal: dict = None


@dataclass
class BalanceContext:
    """Per-aggregate state used by direct and depth-2 planning."""
    effective_hosts: list
    host_cpu_pct: dict
    host_ram_pct: dict
    # Real runtime denominators from node exporter.
    host_cores: dict
    host_ram: dict
    # Placement policy-safe free capacity kept for reporting/backward-compatible
    # field names. Feasibility itself uses host_placement below.
    host_free_cores: dict
    host_free_ram: dict
    host_placement: dict
    vm_host_map: dict
    all_vm_details: dict
    vm_to_groups: dict
    enforce_server_groups: bool
    dest_ready_hosts: object = None
    host_running_vms: object = None
    host_max_instances: object = None
    excluded_vm_ids: object = None
    reject_stats: object = None
    dest_reject_stats: object = None
    skip_stats: object = None
    technical_candidate_count: int = 0
    # Drain mode must be able to move every VM off the host under maintenance,
    # so it bypasses the anti-churn cooldown.
    ignore_cooldown: bool = False
    # Nova status is authoritative even when a VM has no Prometheus series.
    # Drain uses this map to distinguish a deliberately stopped VM from an
    # ACTIVE VM whose metrics are unexpectedly incomplete.
    vm_status_map: object = None
    cooldown_remaining: object = None  # captured when the snapshot is collected
    search_budget: int = 100000
    candidates_examined: int = 0
    # {nova-compute host: availability zone}; None disables the AZ check.
    host_az: object = None

    def consume_candidate(self):
        from compute_maintenance.domain.errors import PlanningBudgetExceeded
        if self.candidates_examined >= self.search_budget:
            raise PlanningBudgetExceeded("budget_exhausted: planner candidate limit reached; feasibility unknown")
        self.candidates_examined += 1


@dataclass
class DrainPlan:
    """Result of planning a full evacuation of one compute host."""
    host: str
    aggregate: str
    remaining_hosts: list
    steps: list
    feasible: bool
    # VMs the planner could not find any destination for.
    blocked: list
    # VMs that cannot be live-migrated at all (not ACTIVE, metrics/metadata gaps).
    unmovable: list
    # Aggregate-level capacity arithmetic, used to tell "cluster is simply too
    # small" apart from "capacity exists but is fragmented".
    capacity: dict
    baseline_weighted_mad: float
    final_weighted_mad: float
    final_host_cpu: dict
    final_host_ram: dict
    hotspot_hosts: list
    infeasible_reasons: list
    ordering: str = ""
    # Affinity groups that cannot be evacuated one VM at a time.
    affinity_conflicts: list = None
    # Steps that had to break a soft-(anti-)affinity preference because no
    # destination without a violation was available.
    soft_policy_warnings: list = None


def get_resource_field(resource, *names):
    for name in names:
        if isinstance(resource, dict):
            for key in (name, name.lower(), name.upper(), name.title()):
                value = resource.get(key)
                if value is not None:
                    return value
            # openstacksdk Resource objects also subclass dict, but fields
            # merged onto them after construction (e.g. ResourceProvider.usages
            # via fetch_resource_provider_usages()) live only on the attribute
            # view; the dict view still reflects the object's initial state.
            value = getattr(resource, name, None)
            if value is not None:
                return value
        else:
            value = getattr(resource, name, None)
            if value is not None:
                return value
    return None
