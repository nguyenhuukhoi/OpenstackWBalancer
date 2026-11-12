"""rc8/rc9: migration wait -- one no-progress window (MIGRATION_TIMEOUT, 600 s)
plus a 30-minute cap (MIGRATION_MAX_WAIT), both configurable;
wbalancer.conf.example stays valid and only lists keys the code reads."""
import logging
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

import compute_maintenance.config as C
from compute_maintenance.application.recovery import MigrationJournal
from compute_maintenance.domain.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "wbalancer.conf.example"
LIVE = ("MIGRATION_TIMEOUT", "MIGRATION_MAX_WAIT",
        "RAM_HOTSPOT_THRESHOLD", "RAM_CRITICAL_THRESHOLD", "CPU_HOTSPOT_THRESHOLD",
        "AUTO_PRESSURE_THRESHOLD", "TARGET_AVG_MAD", "MIN_IMPROVEMENT_THRESHOLD",
        "PLACEMENT_PRESSURE_THRESHOLD", "DRAIN_MAX_HOST_RAM_PCT", "DRAIN_MAX_HOST_CPU_PCT",
        "DRAIN_ABSOLUTE_MAX_HOST_RAM_PCT")


def example_keys():
    return re.findall(r"^#?([A-Z][A-Z0-9_]+)=", EXAMPLE.read_text(encoding="utf-8"), re.MULTILINE)


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    for name in LIVE:
        monkeypatch.setattr(C, name, getattr(C, name))
    for key in set(example_keys()) | {"MAKE_ROOM_MIN_IMPROVEMENT", "MIGRATION_STALL_TIMEOUT"}:
        monkeypatch.delenv(key, raising=False)


def write_config(tmp_path, **extra):
    lines = ["PROMETHEUS_QUERY_URL=http://prom.test:9090/api/v1/query",
             "PROMETHEUS_NODE_JOB=node", "PROMETHEUS_LIBVIRT_JOB=libvirt"]
    lines += [f"{k}={v}" for k, v in extra.items()]
    path = tmp_path / "wbalancer.conf"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def wait_until_given_up(tmp_path, progress_until):
    """Seconds after which the real journal timer gives up, with Nova reporting
    progress every 6 s until ``progress_until`` and nothing afterwards."""
    now = [0.0]
    timer = MigrationJournal(tmp_path / "journal", clock=lambda: now[0])._timer()
    step = 0
    while True:
        # After progress_until the counters stop: the signature keeps its last value.
        timer.observe(("running", min(now[0], max(progress_until, 0))))
        reason = timer.expired()
        if reason:
            return now[0], reason
        step += 1
        now[0] = step * 6.0


# --- one no-progress window + 30-minute cap ------------------------------------------

def test_defaults_600_without_progress_and_30_minute_cap():
    assert (C.MIGRATION_TIMEOUT, C.MIGRATION_MAX_WAIT) == (600, 1800)
    assert not hasattr(C, "MIGRATION_STALL_TIMEOUT")
    assert C.MigrationTimeouts().as_dict() == {"MIGRATION_TIMEOUT": 600.0, "MIGRATION_MAX_WAIT": 1800.0}


@pytest.mark.parametrize("progress_until,stops_at,reason", [
    (-1, 600, "stalled"),         # frozen from the start: minute 10 (as before)
    (120, 720, "stalled"),        # progress until minute 2: minute 12
    (720, 1320, "stalled"),       # progress until minute 12: minute 22
    (1620, 1800, "max_wait"),     # progress until minute 27: the 30-minute cap
    (10_000, 1800, "max_wait"),   # still progressing: the 30-minute cap
])
def test_wait_scenarios_from_the_design_table(tmp_path, progress_until, stops_at, reason):
    assert wait_until_given_up(tmp_path, progress_until) == (stops_at, reason)


def test_journal_and_configured_values(tmp_path):
    journal = MigrationJournal(tmp_path / "a")
    assert (journal.active_timeout, journal.stall_timeout, journal.max_wait) == (600, 600, 1800)
    C.apply_migration_timeouts(C.MigrationTimeouts(no_progress=900, maximum=3600))
    journal = MigrationJournal(tmp_path / "b")
    assert (journal.active_timeout, journal.stall_timeout, journal.max_wait) == (900.0, 900.0, 3600.0)


def test_timeouts_are_read_from_the_config(tmp_path):
    cfg = C.LoadbalancerConfig.load_config(write_config(tmp_path, MIGRATION_TIMEOUT=300, MIGRATION_MAX_WAIT=2700))
    assert cfg.migration_timeouts.as_dict() == {"MIGRATION_TIMEOUT": 300.0, "MIGRATION_MAX_WAIT": 2700.0}


def test_obsolete_stall_key_only_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = C.LoadbalancerConfig.load_config(write_config(tmp_path, MIGRATION_STALL_TIMEOUT=120))
    assert cfg.migration_timeouts.as_dict() == {"MIGRATION_TIMEOUT": 600.0, "MIGRATION_MAX_WAIT": 1800.0}
    assert "Ignoring obsolete key(s) MIGRATION_STALL_TIMEOUT" in caplog.text


@pytest.mark.parametrize("extra,message", [
    ({"MIGRATION_MAX_WAIT": 500}, "MIGRATION_MAX_WAIT .* must be >= MIGRATION_TIMEOUT"),
    ({"MIGRATION_TIMEOUT": 0}, "MIGRATION_TIMEOUT must be a positive number"),
    ({"MIGRATION_TIMEOUT": -1}, "MIGRATION_TIMEOUT must be a positive number"),
    ({"MIGRATION_MAX_WAIT": "inf"}, "MIGRATION_MAX_WAIT"),
])
def test_invalid_timeouts_are_refused(tmp_path, extra, message):
    with pytest.raises(ConfigError, match=message):
        C.LoadbalancerConfig.load_config(write_config(tmp_path, **extra))


def test_cli_applies_and_logs_the_migration_wait(tmp_path, monkeypatch):
    import compute_maintenance.cli.balance as CLI
    seen = {}

    def fake_monitor(cfg, **kw):
        seen.update(timeout=C.MIGRATION_TIMEOUT, max_wait=C.MIGRATION_MAX_WAIT)
        return 0
    monkeypatch.setattr(CLI._execution_monitor, "monitor_by_aggregate", fake_monitor)
    conf = write_config(tmp_path, MIGRATION_TIMEOUT=300, MIGRATION_MAX_WAIT=900)
    result = CliRunner().invoke(CLI.main, ["--config", conf, "--monitor-only",
                                           "--cooldown-file", str(tmp_path / "cool.json")])
    assert result.exit_code == 0, result.output
    assert seen == {"timeout": 300.0, "max_wait": 900.0}


# --- wbalancer.conf.example -------------------------------------------------------------

def test_example_lists_only_keys_the_code_reads():
    source = "\n".join(p.read_text(encoding="utf-8")
                       for p in (ROOT / "src" / "compute_maintenance").rglob("*.py"))
    keys = example_keys()
    assert len(keys) == len(set(keys)), "duplicate key in the example"
    unknown = [k for k in keys if f'"{k}"' not in source and f"'{k}'" not in source]
    assert not unknown, unknown


def test_example_has_no_obsolete_keys():
    obsolete = set(C.OBSOLETE_TIMEOUT_KEYS) | set(C.OBSOLETE_CONFIG_KEYS)
    assert not obsolete & set(example_keys())


def test_example_covers_every_threshold_and_timeout_key():
    keys = set(example_keys())
    expected = {k for k, _a, _d in C.THRESHOLD_KEYS} | {k for k, _a, _d in C.TIMEOUT_KEYS} | {"OS_CLOUD"}
    assert expected <= keys


def test_example_as_shipped_loads_with_defaults(tmp_path):
    path = tmp_path / "wbalancer.conf"
    path.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = C.LoadbalancerConfig.load_config(str(path))
    assert cfg.thresholds.as_dict() == C.Thresholds().as_dict()
    assert cfg.migration_timeouts.as_dict() == C.MigrationTimeouts().as_dict()
    assert cfg.os_cloud == "openstack"


def test_example_with_every_key_uncommented_is_valid_and_matches_defaults(tmp_path):
    text = re.sub(r"^#([A-Z][A-Z0-9_]+=)", r"\1", EXAMPLE.read_text(encoding="utf-8"), flags=re.MULTILINE)
    path = tmp_path / "wbalancer.conf"
    path.write_text(text, encoding="utf-8")
    cfg = C.LoadbalancerConfig.load_config(str(path))
    # The commented values are the documented defaults.
    assert cfg.thresholds.as_dict() == C.Thresholds().as_dict()
    assert cfg.migration_timeouts.as_dict() == C.MigrationTimeouts().as_dict()
    assert cfg.make_room_min_improvement == pytest.approx(0.30)
    assert (cfg.placement_vcpu_headroom_percent, cfg.placement_ram_headroom_percent) == (5.0, 5.0)
    assert (cfg.make_room_max_vm_cpu_percent, cfg.make_room_max_vm_ram_percent) == (10.0, 50.0)
    assert cfg.planner_candidate_budget == 100000
    assert cfg.max_instances_per_host == {"compute-01": 40}
    # The RSS example is exactly the query --use-vm-ram-rss would build for job "libvirt".
    assert cfg.prometheus_query_vm_ram_rss == C.build_default_queries("node", "libvirt")["vm_ram_rss"]
