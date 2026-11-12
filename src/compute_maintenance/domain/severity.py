"""Severity and status mapping: the single source of truth (rc12).

The monitor (one line and one Nagios exit code per aggregate, combined into
the plugin's exit code), the balance classification of a snapshot and the
audit events all take their severity, exit code and the special detail
statuses from this module. The numbers themselves are the Icinga contract in
``config`` (ICINGA_OK=0, WARNING=1, CRITICAL=2, UNKNOWN=3); changing any
mapping here changes what operators' Icinga sees (architecture.md §4.7).

The load classification that produces a detail status (mode ladder + MAD) is
``domain.balance.classify``; this module only says how severe a status is.
"""
import compute_maintenance.config as _config

OK = "OK"
WARNING = "WARNING"
CRITICAL = "CRITICAL"
UNKNOWN = "UNKNOWN"

# rc11: a broken soft-anti-affinity group with a feasible heal move.
SOFT_AA_HEAL_DETAIL_STATUS = "SOFT_AA_HEAL"
# An aggregate whose evaluation failed (missing data): monitor UNKNOWN.
UNKNOWN_DETAIL_STATUS = "UNKNOWN"
# Drain events: a drain has no balance severity, only feasible / infeasible.
DRAIN_FEASIBLE_DETAIL_STATUS = "DRAIN_FEASIBLE"
DRAIN_INFEASIBLE_DETAIL_STATUS = "DRAIN_INFEASIBLE"


def map_detail_status_to_severity(detail_status, reason=None):
    # Healthy or near-optimal state
    if detail_status in ("BALANCED", "BALANCED_OPTIMIZABLE"):
        return OK

    # Imbalance or pressure exists but can still be improved (valid move available)
    # -> do not escalate to CRITICAL to avoid alert noise
    if detail_status in (
        "UNBALANCED",
        "RAM_PRESSURE",
        "CPU_PRESSURE",
        "RAM_HOTSPOT_OPTIMIZABLE",
        "CPU_HOTSPOT_OPTIMIZABLE",
        # rc11: a broken soft-anti-affinity group with a feasible heal move.
        SOFT_AA_HEAL_DETAIL_STATUS,
    ):
        return WARNING

    # High pressure or hotspot with no valid move -> real issue
    # -> requires immediate attention
    if detail_status in (
        "PRESSURE_NO_VALID_MOVE",
        "RAM_HOTSPOT_NO_VALID_MOVE",
        "CPU_HOTSPOT_NO_VALID_MOVE",
    ):
        return CRITICAL

    # Special case: unbalanced but no valid move
    if detail_status == "UNBALANCED_NO_VALID_MOVE":
        # Hard block: no VM can be moved at all (resource/scheduler constraints)
        if reason == "no_technical_candidate":
            return CRITICAL

        # Soft block: moves exist but do not meet improvement threshold
        # -> keep as WARNING to reduce alert noise
        return WARNING

    # Fallback
    return UNKNOWN


def map_severity_to_exit_code(severity):
    if severity == OK:
        return _config.ICINGA_OK
    if severity == WARNING:
        return _config.ICINGA_WARNING
    if severity == CRITICAL:
        return _config.ICINGA_CRITICAL
    return _config.ICINGA_UNKNOWN


def severity_for_exit_code(code):
    """Inverse of map_severity_to_exit_code (Icinga contract 0-3)."""
    return {
        _config.ICINGA_OK: OK,
        _config.ICINGA_WARNING: WARNING,
        _config.ICINGA_CRITICAL: CRITICAL,
    }.get(code, UNKNOWN)


# Severity used to combine per-aggregate results into ONE plugin exit code:
# CRITICAL > WARNING > UNKNOWN > OK. A real alert must never be masked by a
# data-error UNKNOWN, so the numeric max() of the codes (UNKNOWN=3 > CRITICAL=2)
# is wrong for this.
EXIT_CODE_SEVERITY = {
    _config.ICINGA_OK: 0,
    _config.ICINGA_UNKNOWN: 1,
    _config.ICINGA_WARNING: 2,
    _config.ICINGA_CRITICAL: 3,
}


def worst_exit_code(codes):
    """Most severe Icinga/Nagios exit code of ``codes`` (OK when empty); a code
    outside 0..3 counts as UNKNOWN."""
    codes = [code if code in EXIT_CODE_SEVERITY else _config.ICINGA_UNKNOWN for code in codes]
    return max(codes, key=EXIT_CODE_SEVERITY.__getitem__, default=_config.ICINGA_OK)


def heal_verdict(load_exit_code):
    """``(severity, exit_code)`` of a snapshot whose plan is a soft-anti-affinity
    heal move: the heal WARNING combined with the load verdict it replaces, in
    the monitor's order, so a CRITICAL load state is never masked."""
    exit_code = worst_exit_code([load_exit_code, map_severity_to_exit_code(
        map_detail_status_to_severity(SOFT_AA_HEAL_DETAIL_STATUS))])
    return severity_for_exit_code(exit_code), exit_code


def drain_detail_status(feasible):
    """Detail status of a drain event."""
    return DRAIN_FEASIBLE_DETAIL_STATUS if feasible else DRAIN_INFEASIBLE_DETAIL_STATUS
