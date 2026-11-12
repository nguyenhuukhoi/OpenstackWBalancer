"""rc9: default paths drop the "-v1" suffix (same FHS layout), and state left
in /var/lib/wbalancer-v1 by an earlier version is never silently ignored."""
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

import compute_maintenance.config as C
import compute_maintenance.paths as P

ROOT = Path(__file__).resolve().parents[1]


def test_default_layout_without_v1_suffix():
    assert (C.COOLDOWN_STATE_FILE, C.MIGRATION_EVENTS_FILE, C.LOCK_FILE) == (
        "/var/lib/wbalancer/vm-cooldown.json", "/var/log/wbalancer/migrations.jsonl",
        "/var/lock/wbalancer-balancer.lock")
    from compute_maintenance.cli.maintain import parse_args
    args = parse_args(["--profile", "unused"])
    assert (args.state_dir, args.log_dir, args.lock_file, args.wlb_events_file) == (
        "/var/lib/wbalancer/maintenance", "/var/log/wbalancer/maintenance",
        "/var/lock/wbalancer-maint.lock", "/var/log/wbalancer/migrations.jsonl")
    assert P.COORDINATOR_LOCK == "/var/lock/wbalancer-maint.lock"


def test_no_other_v1_path_left_in_the_code():
    hits = []
    for path in (ROOT / "src" / "compute_maintenance").rglob("*.py"):
        if path.name == "paths.py":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"""["']/var/[^"']*wbalancer-v1""", line):
                hits.append(f"{path.name}: {line.strip()}")
    assert not hits, hits


# --- legacy state detection ---------------------------------------------------------

def make(base, *files):
    for rel in files:
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    base.mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("legacy_files,new_files,warns", [
    (["vm-cooldown.json"], [], True),
    (["operations/migration-journal.json"], [], True),
    (["maintenance/h1.json"], [], True),
    ([], [], False),                                               # fresh install
    (["vm-cooldown.json"], ["vm-cooldown.json"], False),           # already moved / in use
    (["maintenance/"], [], False),                                 # empty checkpoint dir is not state
])
def test_legacy_state_message(tmp_path, legacy_files, new_files, warns):
    old, new = tmp_path / "wbalancer-v1", tmp_path / "wbalancer"
    make(old, *[f for f in legacy_files if not f.endswith("/")])
    for f in legacy_files:
        if f.endswith("/"):
            (old / f).mkdir(parents=True, exist_ok=True)
    make(new, *new_files)
    message = P.legacy_state_message(str(new), str(old))
    assert bool(message) is warns
    if warns:
        assert f"mv {old}/* {new}/" in message


@pytest.fixture
def legacy_layout(tmp_path, monkeypatch):
    old, new = tmp_path / "wbalancer-v1", tmp_path / "wbalancer"
    make(old, "operations/migration-journal.json")
    monkeypatch.setattr(P, "LEGACY_STATE_DIR", str(old))
    monkeypatch.setattr(P, "STATE_DIR", str(new))
    monkeypatch.setattr(P, "COOLDOWN_FILE", str(new / "vm-cooldown.json"))
    monkeypatch.setattr(P, "MAINT_STATE_DIR", str(new / "maintenance"))
    conf = tmp_path / "wbalancer.conf"
    conf.write_text("PROMETHEUS_QUERY_URL=http://prom.test:9090/api/v1/query\n"
                    "PROMETHEUS_NODE_JOB=node\nPROMETHEUS_LIBVIRT_JOB=libvirt\n", encoding="utf-8")
    for name in ("RAM_HOTSPOT_THRESHOLD", "RAM_CRITICAL_THRESHOLD", "CPU_HOTSPOT_THRESHOLD",
                 "AUTO_PRESSURE_THRESHOLD", "TARGET_AVG_MAD", "MIN_IMPROVEMENT_THRESHOLD",
                 "PLACEMENT_PRESSURE_THRESHOLD", "DRAIN_MAX_HOST_RAM_PCT", "DRAIN_MAX_HOST_CPU_PCT",
                 "DRAIN_ABSOLUTE_MAX_HOST_RAM_PCT", "MIGRATION_TIMEOUT", "MIGRATION_MAX_WAIT"):
        monkeypatch.setattr(C, name, getattr(C, name))
    return {"old": old, "new": new, "conf": str(conf), "cooldown": str(new / "vm-cooldown.json")}


def test_wbalancer_apply_refuses_before_taking_any_lock(legacy_layout, monkeypatch, caplog):
    import compute_maintenance.cli.balance as CLI

    def no_lock(*a, **k):
        raise AssertionError("must stop before locking")
    monkeypatch.setattr(CLI._locking, "shared_apply_lock", no_lock)
    monkeypatch.setattr(CLI._locking, "acquire_single_instance_lock", no_lock)
    result = CliRunner().invoke(CLI.main, ["--config", legacy_layout["conf"], "-y",
                                           "--cooldown-file", legacy_layout["cooldown"]])
    assert result.exit_code == 1
    assert "Not running: state from an earlier version was found" in caplog.text


def test_wbalancer_dry_run_only_warns(legacy_layout, monkeypatch, caplog):
    import compute_maintenance.cli.balance as CLI
    ran = []
    monkeypatch.setattr(CLI._execution_workflow, "balance_by_aggregate", lambda *a, **k: ran.append(1))
    result = CliRunner().invoke(CLI.main, ["--config", legacy_layout["conf"], "--dry-run",
                                           "--cooldown-file", legacy_layout["cooldown"]])
    assert result.exit_code == 0, result.output
    assert ran == [1]


def test_rolling_maint_stops_before_any_host_work(legacy_layout, monkeypatch, tmp_path):
    import compute_maintenance.cli.maintain as M
    calls = []
    monkeypatch.setattr(M._deps, "check_cli", lambda: calls.append("cli") or (None, None, ["stop"], []))
    common = ["--profile", str(ROOT / "profiles" / "dummy.yaml"), "--notify", "off",
              "--log-dir", str(tmp_path / "log"), "--state-dir", str(legacy_layout["new"] / "maintenance"),
              "--wlb-config", legacy_layout["conf"], "h1"]
    assert M.main(common) == 1 and calls == []          # apply: refused before anything else
    assert M.main(common + ["--dry-run"]) == 1 and calls == ["cli"]   # dry-run: warned, went on
