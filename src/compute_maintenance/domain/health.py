"""Health facts preserve unknown observations instead of interpreting them as green."""
from dataclasses import dataclass, field
from enum import StrEnum


class HealthStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ComponentHealth:
    status: HealthStatus
    detail: str


@dataclass
class HealthReport:
    components: dict[str, ComponentHealth] = field(default_factory=dict)

    @property
    def ok(self):
        return bool(self.components) and all(c.status == HealthStatus.PASS for c in self.components.values())

    @property
    def detail(self):
        return "; ".join(f"{name} {value.status}: {value.detail}" for name, value in self.components.items()
                         if value.status != HealthStatus.PASS) or "all checks pass"

    def observe(self, name, check):
        try:
            problems = check()
            self.components[name] = ComponentHealth(HealthStatus.FAIL if problems else HealthStatus.PASS,
                                                    "; ".join(problems) or "ok")
        except Exception as exc:  # noqa: BLE001 - a crashing check is UNKNOWN for its component
            self.components[name] = ComponentHealth(HealthStatus.UNKNOWN, str(exc))
