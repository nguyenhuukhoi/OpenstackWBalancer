"""errors — extracted domain responsibilities."""



class PrometheusError(Exception):
    pass


class PlacementError(Exception):
    pass


class ConfigError(Exception):
    pass


class OpenStackDataError(Exception):
    pass


class MigrationExecutionError(Exception):
    pass


class StateError(Exception):
    pass


class ProgressTimer:
    """Timeout that extends while an observed signature keeps changing.

    ``expired()`` returns None while waiting is allowed, otherwise a reason:
    "stalled" (no change for ``stall`` seconds after ``minimum``) or
    "max_wait" (``maximum`` reached).
    """

    def __init__(self, minimum, stall, maximum, clock):
        self.minimum, self.stall, self.maximum, self.clock = minimum, stall, maximum, clock
        self.started = self.last_change = clock()
        self.signature = object()

    def observe(self, signature):
        if signature != self.signature:
            self.signature = signature
            self.last_change = self.clock()

    def elapsed(self):
        return self.clock() - self.started

    def expired(self):
        now = self.clock()
        if now - self.started >= self.maximum:
            return "max_wait"
        if now - self.started >= self.minimum and now - self.last_change >= self.stall:
            return "stalled"
        return None


class PlanningBudgetExceeded(OpenStackDataError):
    """A bounded heuristic search stopped; this does not prove infeasibility."""
