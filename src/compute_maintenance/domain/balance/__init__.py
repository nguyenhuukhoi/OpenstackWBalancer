"""Balance planner (domain layer, pure: no I/O).

rc12 split the former single module ``domain/balance.py`` into focused
submodules; every name stays importable from this package, so
``from compute_maintenance.domain import balance`` and ``balance.X`` keep
working. Code that patches a function (tests) must patch the submodule that
calls it; patching the package attribute only affects callers outside the
package (they all call ``balance.X``).

  modes      mode ladder, MAD, adaptive weights, baseline, thresholds
  filters    candidate eligibility + destination filter chain + hotspot guards
  scoring    direct-move search and the soft-anti-affinity tie-break
  makeroom   two-step MAKE_ROOM plan
  heal       soft-anti-affinity heal rung (rc11)
  classify   detail status and reason of a snapshot
The severity / exit-code mapping lives in ``domain.severity`` (rc12 single
source) and is re-exported here under its rc11 names.
"""
from compute_maintenance.domain.balance.classify import (
    classify_cluster_detail_status,
    derive_reason,
)
from compute_maintenance.domain.balance.filters import (
    _inc_counter,
    availability_zone_allows,
    check_direct_destination,
    destination_hot_after,
    is_hotspot_guardrail_satisfied,
    make_step_from_projection,
    max_instances_allows,
    project_runtime_move,
    record_rejection,
    record_skip,
    vm_candidate_details,
    vm_planning_block_reason,
    would_create_any_new_hotspot,
    would_create_new_hotspot,
)
from compute_maintenance.domain.balance.heal import (
    SOFT_AA_DEFERRED,
    SOFT_AA_UNRESOLVABLE,
    _format_counts,
    find_soft_aa_heal_move,
    plan_soft_aa_heal,
    soft_aa_heal_headroom,
    soft_aa_heal_mad_check,
    soft_aa_heal_mad_limit,
    soft_aa_heal_vm_details,
)
from compute_maintenance.domain.balance.makeroom import (
    find_best_make_room_plan,
    placement_with_usage_delta,
    search_best_make_room_for_state,
)
from compute_maintenance.domain.balance.modes import (
    calculate_adaptive_weights,
    calculate_mode_weights,
    calculate_pressure_signals,
    choose_effective_mode,
    compute_baseline_metrics,
    determine_adaptive_mode,
    determine_hotspot_mode,
    determine_improvement_threshold,
    determine_plan_improvement_threshold,
    get_hotspot_source_host,
    mean_absolute_deviation,
)
from compute_maintenance.domain.balance.scoring import (
    choose_with_soft_aa_tiebreak,
    direct_move_to_plan,
    find_best_move,
    search_best_move_for_state,
    soft_aa_tiebreak_enabled,
)
from compute_maintenance.domain.severity import (
    SOFT_AA_HEAL_DETAIL_STATUS,
    map_detail_status_to_severity,
    map_severity_to_exit_code,
    severity_for_exit_code,
)

__all__ = [
    "SOFT_AA_DEFERRED",
    "SOFT_AA_HEAL_DETAIL_STATUS",
    "SOFT_AA_UNRESOLVABLE",
    "_format_counts",
    "_inc_counter",
    "availability_zone_allows",
    "calculate_adaptive_weights",
    "calculate_mode_weights",
    "calculate_pressure_signals",
    "check_direct_destination",
    "choose_effective_mode",
    "choose_with_soft_aa_tiebreak",
    "classify_cluster_detail_status",
    "compute_baseline_metrics",
    "derive_reason",
    "destination_hot_after",
    "determine_adaptive_mode",
    "determine_hotspot_mode",
    "determine_improvement_threshold",
    "determine_plan_improvement_threshold",
    "direct_move_to_plan",
    "find_best_make_room_plan",
    "find_best_move",
    "find_soft_aa_heal_move",
    "get_hotspot_source_host",
    "is_hotspot_guardrail_satisfied",
    "make_step_from_projection",
    "map_detail_status_to_severity",
    "map_severity_to_exit_code",
    "max_instances_allows",
    "mean_absolute_deviation",
    "placement_with_usage_delta",
    "plan_soft_aa_heal",
    "project_runtime_move",
    "record_rejection",
    "record_skip",
    "search_best_make_room_for_state",
    "search_best_move_for_state",
    "severity_for_exit_code",
    "soft_aa_heal_headroom",
    "soft_aa_heal_mad_check",
    "soft_aa_heal_mad_limit",
    "soft_aa_heal_vm_details",
    "soft_aa_tiebreak_enabled",
    "vm_candidate_details",
    "vm_planning_block_reason",
    "would_create_any_new_hotspot",
    "would_create_new_hotspot",
]
