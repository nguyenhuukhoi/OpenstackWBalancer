"""Typed state of one evaluated aggregate (rc12).

``application.snapshots.evaluate_aggregate_state`` builds one
:class:`PlanContext` per aggregate and evaluation: what the snapshot measured,
the balance verdict and the plan it chose. Balancing, migrations, monitoring,
reports and the drain planner read it by attribute. It replaces the rc11
stringly-typed state dict; the field names are the old keys.

Construction takes keyword arguments only; a field that was never set reads
as ``None``. A failed evaluation sets only ``aggregate``, ``error`` and
``hosts_in_agg`` (and ``excluded_hosts`` when the failure is "fewer than two
destination-ready hosts").

Compatibility: item access (``ctx["mode"]``, ``ctx.get("mode")``,
``"mode" in ctx``, ``dict(ctx)``) is a view of exactly the fields that were
set, so third-party readers of the old dict (tests, lab scripts) see the same
keys as before: a failed evaluation still has no ``"best_plan"`` key and an
unset key still raises KeyError. New code uses attributes.
"""
from dataclasses import dataclass, fields

from compute_maintenance.domain.models import BestMove, MigrationPlan


class PlanContextIncomplete(KeyError):
    """A field an operation depends on was never set.

    A KeyError on purpose: the rc11 state dict raised KeyError for a missing
    key and callers rely on that type (e.g. a context without its
    ``exclude_unready_hosts`` planning rule is refused before revalidation).
    """


@dataclass(frozen=True)
class PlanLabels:
    """Classification an audit event takes from the snapshot its plan was MADE
    from, never from a snapshot refreshed for revalidation (that one plans
    nothing and would read "no technical candidate", CRITICAL)."""

    mode: str | None = None
    detail_status: str | None = None
    severity: str | None = None
    best_improvement: float | None = None
    improvement_threshold: float | None = None

    @classmethod
    def of(cls, ctx):
        """The labels of ``ctx``'s own planning verdict."""
        return cls(mode=ctx.mode, detail_status=ctx.detail_status, severity=ctx.severity,
                   best_improvement=ctx.best_improvement, improvement_threshold=ctx.improvement_threshold)


@dataclass(eq=False, init=False)
class PlanContext:
    """One aggregate's snapshot, verdict and plan (see module docstring)."""

    # --- identity / failure
    aggregate: str | None = None
    error: str | None = None                      # set only when the evaluation failed (fail closed)
    hosts_in_agg: list | None = None              # hosts in the algorithm (unready ones left out in balance)
    excluded_hosts: dict | None = None            # {host: reason} left out of the algorithm

    # --- per-host runtime load and Placement capacity (effective hosts only)
    host_cpu_pct: dict | None = None
    host_ram_pct: dict | None = None
    host_cores: dict | None = None
    host_free_cores: dict | None = None
    host_ram: dict | None = None
    host_free_ram: dict | None = None
    host_running_vms: dict | None = None
    host_max_instances: dict | None = None
    host_placement: dict | None = None
    placement_status_by_host: dict | None = None
    placement_summary: dict | None = None
    dest_ready_hosts: list | None = None          # sorted
    dest_not_ready_reasons: dict | None = None

    # --- VM inventory (every host of the aggregate, for server-group checks)
    all_vm_names: dict | None = None
    all_vm_details: dict | None = None
    vm_host_map: dict | None = None
    vm_status_map: dict | None = None
    vm_to_groups: dict | None = None
    excluded_vm_ids: list | None = None           # sorted
    excluded_vms: list | None = None
    unmatched_exclude_patterns: list | None = None

    # --- baseline metrics and mode ladder
    cpu_mean: float | None = None
    ram_mean: float | None = None
    baseline_cpu_mad: float | None = None
    baseline_ram_mad: float | None = None
    baseline_weighted_mad: float | None = None
    mode: str | None = None
    max_cpu: float | None = None
    max_ram: float | None = None
    cpu_weight: float | None = None
    ram_weight: float | None = None
    cpu_signal: float | None = None
    ram_signal: float | None = None

    # --- thresholds the plan must meet
    improvement_threshold: float | None = None
    # Individual executable moves are revalidated against the direct-move
    # threshold; the higher plan threshold is paid once by a whole MAKE_ROOM
    # chain, not again by its final step after room has been made.
    step_improvement_threshold: float | None = None
    make_room_improvement_threshold: float | None = None

    # --- planning result
    best_move: BestMove | None = None             # kept for events; best_plan is authoritative
    best_plan_candidate: MigrationPlan | None = None
    best_plan: MigrationPlan | None = None
    best_improvement: float | None = None
    has_candidate: bool | None = None
    has_valid_move: bool | None = None
    search_scope: str | None = None
    hotspot_source_host: str | None = None
    reject_stats: dict | None = None
    dest_reject_stats: dict | None = None
    skip_stats: dict | None = None
    soft_aa_heal: dict | None = None              # result of balance.plan_soft_aa_heal (rc11)
    load_status: dict | None = None               # load verdict when a heal move replaced it (rc11)

    # --- verdict (Icinga contract)
    detail_status: str | None = None
    reason: str | None = None
    severity: str | None = None
    exit_code: int | None = None
    message: str | None = None                    # monitor text (presentation.reports)

    # --- planning context: every refresh of this context (revalidation before
    # submit, MAKE_ROOM step 2) must rebuild the baseline with the same rule.
    collected_at: float | None = None
    exclude_unready_hosts: bool | None = None
    host_az: dict | None = None
    settle_mode: str | None = None
    arrival_compensation: dict | None = None
    planner_candidate_budget: int | None = None
    balance_candidates_examined: int | None = None
    raw_metric_max_age_seconds: dict | None = None
    snapshot: object = None                       # domain.snapshot.ClusterSnapshot
    collection_seconds: float | None = None

    # --- labels of the plan this context executes (set on refreshed contexts
    # and drain contexts; None = use this context's own verdict)
    event_labels: PlanLabels | None = None

    def __init__(self, **values):
        unknown = sorted(set(values) - _FIELD_SET)
        if unknown:
            raise TypeError("unknown PlanContext field(s): " + ", ".join(unknown))
        self.__dict__["_set_fields"] = set()
        for name, value in values.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in _FIELD_SET:
            self._set_fields.add(name)

    # ------------------------------------- mapping view (rc11 dict compatibility)
    def __getitem__(self, key):
        if key not in self._set_fields:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in _FIELD_SET:
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self._set_fields

    def get(self, key, default=None):
        return getattr(self, key) if key in self._set_fields else default

    def keys(self):
        return [name for name in FIELD_NAMES if name in self._set_fields]


FIELD_NAMES = tuple(f.name for f in fields(PlanContext))
_FIELD_SET = frozenset(FIELD_NAMES)
