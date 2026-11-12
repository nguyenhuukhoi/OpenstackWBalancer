"""Cluster status classification of a snapshot: detail status and reason.

How severe a status is (severity, Icinga exit code) is defined once, in
``domain.severity``."""
import compute_maintenance.config as _config


def classify_cluster_detail_status(mode, max_cpu, max_ram, baseline_cpu_mad, baseline_ram_mad, has_valid_move):
    is_unbalanced_by_mad = (
        baseline_cpu_mad > _config.runtime().target_avg_mad or
        baseline_ram_mad > _config.runtime().target_avg_mad
    )

    if mode in _config.RAM_HOTSPOT_MODES:
        if has_valid_move:
            return "RAM_HOTSPOT_OPTIMIZABLE"
        return "RAM_HOTSPOT_NO_VALID_MOVE"

    if mode == _config.MODE_CPU_HOTSPOT:
        if has_valid_move:
            return "CPU_HOTSPOT_OPTIMIZABLE"
        return "CPU_HOTSPOT_NO_VALID_MOVE"

    if mode == _config.MODE_PRESSURE:
        if has_valid_move:
            return "RAM_PRESSURE" if max_ram >= max_cpu else "CPU_PRESSURE"
        return "PRESSURE_NO_VALID_MOVE"

    if is_unbalanced_by_mad:
        if has_valid_move:
            return "UNBALANCED"
        return "UNBALANCED_NO_VALID_MOVE"

    if has_valid_move:
        return "BALANCED_OPTIMIZABLE"

    return "BALANCED"


def derive_reason(has_candidate, has_valid_move, mode=None):
    if has_valid_move:
        return None

    if not has_candidate:
        return "no_technical_candidate"

    if mode in _config.HOTSPOT_MODES:
        return "no_hotspot_candidate_above_improvement_threshold"

    if mode == _config.MODE_PRESSURE:
        return "no_pressure_candidate_above_improvement_threshold"

    return "no_candidate_above_improvement_threshold"
