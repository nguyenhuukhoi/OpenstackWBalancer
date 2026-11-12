"""rc6: openstacksdk / python-openstackclient are installed by the operator.

The package no longer pins them; OpenStack 2026.1 upper-constraints is the
tested baseline. These tests pin down the start-up checks, the CLI lookup
(venv first, then PATH) and the packaging contract.
"""
import ast
import io
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import compute_maintenance.cli.entry as E
import compute_maintenance.dependencies as D

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ("openstacksdk", "python-openstackclient")


# --- version baseline ---------------------------------------------------------

@pytest.mark.parametrize("dist,version,older", [
    ("openstacksdk", "4.10.0", False),
    ("openstacksdk", "4.10", False),
    ("openstacksdk", "4.11.0rc1", False),
    ("openstacksdk", "5.0.0", False),
    ("openstacksdk", "4.9.2", True),
    ("openstacksdk", "4.10.0.dev3", False),
    ("python-openstackclient", "9.0.1", False),
    ("python-openstackclient", "9.0.0", True),
    ("python-openstackclient", "6.6.0", True),
    ("python-openstackclient", "10.0.0", False),
])
def test_older_than_2026_1_baseline(dist, version, older):
    assert D.older_than_baseline(dist, version) is older


def test_unparsable_version_is_not_reported_as_older():
    assert D.older_than_baseline("openstacksdk", "unknown") is False


# --- openstacksdk check at start-up -------------------------------------------

def test_missing_sdk_exits_1_with_install_instructions(monkeypatch):
    monkeypatch.setattr(D, "installed_version", lambda dist: None)
    err = io.StringIO()
    with pytest.raises(SystemExit) as exc:
        E.require_sdk(1, stderr=err)
    assert exc.value.code == 1
    text = err.getvalue()
    assert "openstacksdk is not installed" in text
    assert D.CONSTRAINTS_URL in text and "python-openstackclient" in text


def test_missing_sdk_in_monitor_mode_is_icinga_unknown(monkeypatch):
    monkeypatch.setattr(D, "installed_version", lambda dist: None)
    monkeypatch.setattr(sys, "argv", ["wbalancer", "--monitor-only", "-a", "agg"])
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(SystemExit) as exc:
        E.balance()
    assert exc.value.code == 3


def test_old_sdk_only_warns(monkeypatch):
    monkeypatch.setattr(D, "installed_version", lambda dist: "4.1.0")
    err = io.StringIO()
    E.require_sdk(1, stderr=err)  # no SystemExit
    assert "older than the OpenStack 2026.1 baseline 4.10.0" in err.getvalue()


def test_baseline_sdk_is_silent(monkeypatch):
    monkeypatch.setattr(D, "installed_version", lambda dist: "4.10.0")
    err = io.StringIO()
    E.require_sdk(1, stderr=err)
    assert err.getvalue() == ""


# --- openstack CLI lookup ------------------------------------------------------

def _executable(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


def test_cli_next_to_interpreter_wins_over_path(tmp_path, monkeypatch):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    local = _executable(venv_bin / "openstack")
    monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))
    monkeypatch.setattr(D.shutil, "which", lambda name: "/usr/bin/openstack")
    assert D.openstack_cli_path() == local


def test_cli_falls_back_to_path(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(D.shutil, "which", lambda name: "/usr/bin/openstack")
    assert D.openstack_cli_path() == "/usr/bin/openstack"


@pytest.mark.parametrize("stdout,stderr,expected", [
    ("openstack 9.0.1\n", "", "9.0.1"),
    ("", "openstack 6.6.0\n", "6.6.0"),       # some releases print to stderr
    ("openstack 10.0.0rc1\n", "", "10.0.0rc1"),
    ("garbage", "", None),
])
def test_cli_version_parsing(stdout, stderr, expected):
    run = lambda *a, **k: NS(stdout=stdout, stderr=stderr, returncode=0)
    assert D.cli_version("/x/openstack", run=run) == expected


def test_cli_version_failure_is_none():
    def run(*a, **k):
        raise subprocess.TimeoutExpired("openstack", 60)
    assert D.cli_version("/x/openstack", run=run) is None


def test_check_cli_reports_missing_old_and_ok(monkeypatch):
    monkeypatch.setattr(D, "openstack_cli_path", lambda: None)
    path, version, errors, warnings = D.check_cli()
    assert path is None and errors and not warnings

    monkeypatch.setattr(D, "openstack_cli_path", lambda: "/usr/bin/openstack")
    old = lambda *a, **k: NS(stdout="openstack 6.6.0", stderr="", returncode=0)
    path, version, errors, warnings = D.check_cli(run=old)
    assert (path, version, errors) == ("/usr/bin/openstack", "6.6.0", [])
    assert "older than the OpenStack 2026.1 baseline 9.0.1" in warnings[0]

    ok = lambda *a, **k: NS(stdout="openstack 9.0.1", stderr="", returncode=0)
    assert D.check_cli(run=ok) == ("/usr/bin/openstack", "9.0.1", [], [])


def test_cloud_runs_the_resolved_cli(monkeypatch):
    import compute_maintenance.adapters.compute_cli as C
    seen = []
    monkeypatch.setattr(C._maint_remote, "run", lambda argv, **k: seen.append(argv) or NS(stdout="[]"))
    monkeypatch.setattr(C._deps, "openstack_cli_path", lambda: "/opt/venv/bin/openstack")
    cloud = C.Cloud("openstack")
    cloud.env = {}
    cloud.osc("server", "list")
    explicit = C.Cloud("openstack", cli="/other/openstack")
    explicit.env = {}
    explicit.osc("server", "list")
    assert [argv[0] for argv in seen] == ["/opt/venv/bin/openstack", "/other/openstack"]


def test_rolling_maint_stops_before_any_host_work_without_cli(tmp_path, monkeypatch):
    import compute_maintenance.cli.maintain as M
    monkeypatch.setattr(M._deps, "check_cli", lambda: (None, None, ["no openstack CLI"], []))
    touched = []
    monkeypatch.setattr(M._maint_inventory, "resolve_hosts", lambda *a: touched.append(a))
    monkeypatch.setattr(M._sources_compute, "Cloud", lambda *a, **k: touched.append(a))
    rc = M.main(["--profile", str(ROOT / "profiles" / "dummy.yaml"), "--notify", "off",
                 "--log-dir", str(tmp_path / "log"), "--state-dir", str(tmp_path / "state"),
                 "--wlb-config", str(tmp_path / "missing.conf"), "h1"])
    assert rc == 1 and touched == []


# --- packaging contract ----------------------------------------------------------

def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _name(requirement):
    return re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].strip().lower()


def test_openstack_clients_are_not_package_dependencies():
    project = _pyproject()["project"]
    declared = {_name(r) for r in project["dependencies"]}
    for extra in project.get("optional-dependencies", {}).values():
        declared |= {_name(r) for r in extra}
    assert declared.isdisjoint(CLIENTS)
    lock = (ROOT / "requirements.lock.txt").read_text(encoding="utf-8").lower()
    assert not re.search(r"^(openstacksdk|python-openstackclient)==", lock, re.MULTILINE)


def test_console_scripts_go_through_the_dependency_check():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts == {"wbalancer": "compute_maintenance.cli.entry:balance",
                       "rolling-maint": "compute_maintenance.cli.entry:maintain"}


def _constraints():
    out = {}
    for line in (ROOT / "constraints" / "openstack-2026.1.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)===([^;\s]+)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def test_lock_matches_the_2026_1_baseline():
    uc = _constraints()
    assert uc["openstacksdk"] == D.BASELINE_VERSIONS["openstacksdk"]
    assert uc["python-openstackclient"] == D.BASELINE_VERSIONS["python-openstackclient"]
    lock = (ROOT / "requirements.lock.txt").read_text(encoding="utf-8")
    pinned = {m.group(1).lower(): m.group(2)
              for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", lock, re.MULTILINE)}
    shared = {name: (version, uc[name]) for name, version in pinned.items() if name in uc}
    assert shared, "lock and constraints should overlap (requests, PyYAML, ...)"
    mismatched = {n: v for n, v in shared.items() if v[0] != v[1]}
    assert not mismatched, mismatched


def test_dependency_check_modules_use_only_the_standard_library():
    allowed_local = {"compute_maintenance.config", "compute_maintenance.dependencies",
                     "compute_maintenance.cli.balance", "compute_maintenance.cli.maintain"}
    for rel in ("src/compute_maintenance/dependencies.py", "src/compute_maintenance/cli/entry.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert name in allowed_local or name.split(".")[0] in sys.stdlib_module_names, (rel, name)
