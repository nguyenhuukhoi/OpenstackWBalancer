"""maint.errors — extracted domain responsibilities."""
from compute_maintenance.domain.errors import StateError


class StepFailed(StateError):
    pass


class CommandOutcomeUnknown(StepFailed):
    """Transport/timeout does not prove whether the remote side effect finished."""


class SkipHost(Exception):
    pass


class OperatorStop(Exception):
    """Operator answered no; the host was not touched by this run."""
