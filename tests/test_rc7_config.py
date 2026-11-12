"""rc7: decision thresholds come from the config file; the cloud name is free.

Thresholds: RAM/CPU hotspot, RAM critical, pressure, target MAD, minimum
improvement and Placement pressure are config keys with the previous constants
as defaults; apply_thresholds() makes them the live values for the process.
Cloud: OS_CLOUD (config or environment) or --os-cloud, default "openstack".
"""
from types import SimpleNamespace as NS

import pytest
from click.testing import CliRunner

import compute_maintenance.config as C
import compute_maintenance.domain.balance as B
import compute_maintenance.domain.drain as DR
from compute_maintenance.domain.errors import ConfigError

LIVE = ("RAM_HOTSPOT_THRESHOLD", "RAM_CRITICAL_THRESHOLD", "CPU_HOTSPOT_THRESHOLD",
        "AUTO_PRESSURE_THRESHOLD", "TARGET_AVG_MAD", "MIN_IMPROVEMENT_THRESHOLD",
        "PLACEMENT_PRESSURE_THRESHOLD", "DRAIN_MAX_HOST_RAM_PCT", "DRAIN_MAX_HOST_CPU_PCT",
        "DRAIN_ABSOLUTE_MAX_HOST_RAM_PCT")


@pytest.fixture(autouse=True)
def restore_live_thresholds(monkeypatch):
    # apply_thresholds() mutates module globals; monkeypatch restores them.
    for name in LIVE:
        monkeypatch.setattr(C, name, getattr(C, name))
    for name in ("OS_CLOUD",) + tuple(k for k, _a, _d in C.THRESHOLD_KEYS) + ("MAKE_ROOM_MIN_IMPROVEMENT",):
        monkeypatch.delenv(name, raising=False)


def write_config(tmp_path, **extra):
    lines = ["PROMETHEUS_QUERY_URL=http://prom.test:9090/api/v1/query",
             "PROMETHEUS_NODE_JOB=node", "PROMETHEUS_LIBVIRT_JOB=libvirt"]
    lines += [f"{k}={v}" for k, v in extra.items()]
    path = tmp_path / "wbalancer.conf"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# --- defaults and parsing -------------------------------------------------------

def test_defaults_equal_the_previous_constants():
    t = C.Thresholds()
    assert t.as_dict() == {
        "RAM_HOTSPOT_THRESHOLD": 80.0, "RAM_CRITICAL_THRESHOLD": 95.0, "CPU_HOTSPOT_THRESHOLD": 70.0,
        "PRESSURE_THRESHOLD": 80.0, "TARGET_AVG_MAD": 3.0, "MIN_IMPROVEMENT_THRESHOLD": 0.15,
        "PLACEMENT_PRESSURE_THRESHOLD": 90.0}
    assert (C.RAM_HOTSPOT_THRESHOLD, C.RAM_CRITICAL_THRESHOLD, C.CPU_HOTSPOT_THRESHOLD,
            C.AUTO_PRESSURE_THRESHOLD, C.TARGET_AVG_MAD, C.MIN_IMPROVEMENT_THRESHOLD,
            C.PLACEMENT_PRESSURE_THRESHOLD) == (80, 95.0, 70.0, 80.0, 3, 0.15, 90.0)


def test_config_without_threshold_keys_keeps_defaults(tmp_path):
    cfg = C.LoadbalancerConfig.load_config(write_config(tmp_path))
    assert cfg.thresholds.as_dict() == C.Thresholds().as_dict()
    assert cfg.make_room_min_improvement == pytest.approx(0.30)
    assert cfg.os_cloud == "openstack"


def test_threshold_keys_are_read_from_the_config_file(tmp_path):
    cfg = C.LoadbalancerConfig.load_config(write_config(
        tmp_path, RAM_HOTSPOT_THRESHOLD=75, RAM_CRITICAL_THRESHOLD=85, CPU_HOTSPOT_THRESHOLD=85,
        PRESSURE_THRESHOLD=70, TARGET_AVG_MAD=2.5, MIN_IMPROVEMENT_THRESHOLD=0.05,
        PLACEMENT_PRESSURE_THRESHOLD=95))
    assert cfg.thresholds.as_dict() == {
        "RAM_HOTSPOT_THRESHOLD": 75.0, "RAM_CRITICAL_THRESHOLD": 85.0, "CPU_HOTSPOT_THRESHOLD": 85.0,
        "PRESSURE_THRESHOLD": 70.0, "TARGET_AVG_MAD": 2.5, "MIN_IMPROVEMENT_THRESHOLD": 0.05,
        "PLACEMENT_PRESSURE_THRESHOLD": 95.0}
    # MAKE_ROOM default follows the configured direct-move threshold (2 steps).
    assert cfg.make_room_min_improvement == pytest.approx(0.10)


def test_environment_overrides_the_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAM_HOTSPOT_THRESHOLD", "77")
    cfg = C.LoadbalancerConfig.load_config(write_config(tmp_path, RAM_HOTSPOT_THRESHOLD=75))
    assert cfg.thresholds.ram_hotspot == 77.0


@pytest.mark.parametrize("extra,message", [
    ({"RAM_CRITICAL_THRESHOLD": 80}, "must be above RAM_HOTSPOT_THRESHOLD"),
    ({"RAM_HOTSPOT_THRESHOLD": 96}, "must be above RAM_HOTSPOT_THRESHOLD"),
    ({"CPU_HOTSPOT_THRESHOLD": 0}, "CPU_HOTSPOT_THRESHOLD must be > 0 and <= 100"),
    ({"PRESSURE_THRESHOLD": 101}, "PRESSURE_THRESHOLD must be > 0 and <= 100"),
    ({"TARGET_AVG_MAD": -1}, "TARGET_AVG_MAD must be > 0"),
    ({"MIN_IMPROVEMENT_THRESHOLD": 0}, "MIN_IMPROVEMENT_THRESHOLD must be > 0"),
    ({"PLACEMENT_PRESSURE_THRESHOLD": "nan"}, "PLACEMENT_PRESSURE_THRESHOLD"),
    ({"RAM_HOTSPOT_THRESHOLD": "high"}, "RAM_HOTSPOT_THRESHOLD"),
    ({"MIN_IMPROVEMENT_THRESHOLD": 0.2, "MAKE_ROOM_MIN_IMPROVEMENT": 0.1}, "MAKE_ROOM_MIN_IMPROVEMENT"),
])
def test_invalid_thresholds_are_refused(tmp_path, extra, message):
    with pytest.raises(ConfigError, match=message):
        C.LoadbalancerConfig.load_config(write_config(tmp_path, **extra))


# --- the configured values drive the planner ------------------------------------

def test_hotspot_mode_follows_applied_thresholds():
    cpu, ram = {"a": 10.0, "b": 10.0}, {"a": 78.0, "b": 40.0}
    assert B.determine_hotspot_mode(cpu, ram)[0] == C.MODE_NORMAL       # 78 < 80 (default)
    C.apply_thresholds(C.Thresholds(ram_hotspot=75, ram_critical=85, cpu_hotspot=85))
    assert B.determine_hotspot_mode(cpu, ram)[0] == C.MODE_RAM_HOTSPOT  # 78 > 75 (old script value)
    assert B.determine_hotspot_mode(cpu, {"a": 86.0, "b": 40.0})[0] == C.MODE_RAM_CRITICAL_HOTSPOT
    assert B.determine_hotspot_mode({"a": 80.0, "b": 10.0}, {"a": 10.0, "b": 10.0})[0] == C.MODE_NORMAL


def test_pressure_mode_becomes_reachable_with_a_lower_pressure_threshold():
    cpu = {"a": 68.0, "b": 69.0, "c": 67.0}   # mean 68, max 69: no CPU hotspot (> 70)
    ram = {"a": 40.0, "b": 40.0, "c": 40.0}
    mean_cpu, mean_ram = sum(cpu.values()) / 3, 40.0
    assert C.Thresholds().pressure_reachable() is False          # defaults: dead mode
    assert B.choose_effective_mode(mean_cpu, mean_ram, cpu, ram)[0] == C.MODE_MAD
    t = C.Thresholds(pressure=65)
    assert t.pressure_reachable() is True
    C.apply_thresholds(t)
    assert B.choose_effective_mode(mean_cpu, mean_ram, cpu, ram)[0] == C.MODE_PRESSURE
    assert B.determine_adaptive_mode(mean_cpu, mean_ram) == C.MODE_PRESSURE   # default arg reads live value


def test_target_mad_min_improvement_and_drain_caps_follow():
    C.apply_thresholds(C.Thresholds(ram_hotspot=70, ram_critical=90, cpu_hotspot=60,
                                    target_mad=5, min_improvement=0.4))
    assert B.determine_improvement_threshold() == 0.4
    # MAD 4 is "balanced" with TARGET_AVG_MAD 5, "unbalanced" with the default 3.
    assert B.classify_cluster_detail_status(C.MODE_MAD, 50, 50, 4, 4, True) == "BALANCED_OPTIMIZABLE"
    assert DR.drain_host_caps(False) == (60.0, 70.0)
    assert DR.drain_host_caps(True) == (100.0, 90.0)


def test_unknown_threshold_name_is_a_programming_error():
    with pytest.raises(TypeError):
        C.Thresholds(ram=1)


# --- cloud name -------------------------------------------------------------------

def test_cloud_name_from_config_and_environment(tmp_path, monkeypatch):
    assert C.LoadbalancerConfig.load_config(write_config(tmp_path, OS_CLOUD="prod")).os_cloud == "prod"
    monkeypatch.setenv("OS_CLOUD", "dr-site")
    assert C.LoadbalancerConfig.load_config(write_config(tmp_path, OS_CLOUD="prod")).os_cloud == "dr-site"


def test_connection_uses_the_configured_cloud(monkeypatch):
    import compute_maintenance.application.monitoring as M
    from compute_maintenance.adapters import openstack as adapter
    calls = []
    monkeypatch.setattr(adapter.openstack, "connect", lambda **kw: calls.append(kw))
    adapter.get_openstack_connection("prod")
    adapter.get_openstack_connection()
    assert calls == [{"cloud": "prod", "api_timeout": 30}, {"cloud": "openstack", "api_timeout": 30}]

    seen = []

    def stop(cloud):
        seen.append(cloud)
        raise RuntimeError("stop")
    monkeypatch.setattr(M._sources_openstack, "get_openstack_connection", stop)
    with pytest.raises(RuntimeError):
        M.monitor_by_aggregate(NS(os_cloud="envvars"))
    assert seen == ["envvars"]


def test_wbalancer_cli_applies_config_thresholds_and_os_cloud(tmp_path, monkeypatch):
    import compute_maintenance.cli.balance as CLI
    seen = {}

    def fake_monitor(cfg, **kw):
        seen.update(cloud=cfg.os_cloud, ram=C.RAM_HOTSPOT_THRESHOLD, drain_ram=C.DRAIN_MAX_HOST_RAM_PCT)
        return 0
    monkeypatch.setattr(CLI._execution_monitor, "monitor_by_aggregate", fake_monitor)
    conf = write_config(tmp_path, RAM_HOTSPOT_THRESHOLD=75, OS_CLOUD="from-config")
    result = CliRunner().invoke(CLI.main, ["--config", conf, "--monitor-only", "--os-cloud", "prod",
                                           "--cooldown-file", str(tmp_path / "cool.json")])
    assert result.exit_code == 0, result.output
    assert seen == {"cloud": "prod", "ram": 75.0, "drain_ram": 75.0}


def test_rolling_maint_passes_its_cloud_to_the_drain_child():
    from compute_maintenance.application.stages import StageActions

    def cmd(os_cloud, wbalancer=None):
        args = NS(wbalancer=wbalancer, wlb_config="/etc/w.conf", wlb_events_file="/tmp/e.jsonl",
                  drain_allow_hotspot=False, notify="off", wlb_extra=None, os_cloud=os_cloud)
        return StageActions.wbalancer_cmd(NS(args=args, aggregate_for=lambda h: "agg"), "h1", True)

    prod = cmd("prod")
    assert prod[prod.index("--os-cloud") + 1] == "prod"
    assert "--os-cloud" in cmd("openstack")                              # installed module: always
    # rc10: a custom child always gets it too (review M1), even the default cloud.
    custom = cmd("openstack", wbalancer="/opt/custom/wbalancer.py")
    assert custom[custom.index("--os-cloud") + 1] == "openstack"
    assert "--os-cloud" in cmd("prod", wbalancer="/opt/old/wbalancer.py")


def test_placement_line_shows_the_configured_threshold():
    from compute_maintenance.domain.plan_context import PlanContext
    from compute_maintenance.presentation.reports import build_detail_lines
    state = PlanContext(mode=C.MODE_MAD, max_cpu=50.0, max_ram=60.0, cpu_mean=40.0, ram_mean=50.0,
                        baseline_cpu_mad=1.0, baseline_ram_mad=2.0, baseline_weighted_mad=1.5,
                        cpu_weight=0.5, ram_weight=0.5, has_valid_move=False, reason="no_technical_candidate",
                        placement_summary={"vcpu_pressure_hosts": 1, "memory_pressure_hosts": 0})
    assert "VCPU>=90% hosts=1" in "\n".join(build_detail_lines(state))
    C.apply_thresholds(C.Thresholds(placement_pressure=95))
    assert "VCPU>=95% hosts=1 | MEM>=95% hosts=0" in "\n".join(build_detail_lines(state))
