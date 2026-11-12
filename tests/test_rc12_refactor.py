"""rc12 technical-debt cleanup (no behaviour change): typed PlanContext,
balance package, RuntimeSettings, single severity source, hermetic config."""
import ast
import dataclasses
from pathlib import Path

import pytest

import compute_maintenance.config as C
from compute_maintenance.domain import balance
from compute_maintenance.domain.plan_context import (
    PlanContext,
    PlanContextIncomplete,
    PlanLabels,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "compute_maintenance"


@pytest.fixture(autouse=True)
def restore_runtime():
    saved = C.runtime()
    yield
    C.set_runtime(saved)


# --- STEP 1: typed PlanContext -------------------------------------------------------

def test_plan_context_attributes_and_the_rc11_dict_view_agree():
    ctx = PlanContext(aggregate="agg", mode="mad", reason=None)
    assert (ctx.aggregate, ctx.mode, ctx.reason, ctx.best_plan) == ("agg", "mad", None, None)
    # The mapping view is exactly the set fields (the rc11 dict keys).
    assert dict(ctx) == {"aggregate": "agg", "mode": "mad", "reason": None}
    assert "reason" in ctx and "best_plan" not in ctx
    assert ctx.get("best_plan", "absent") == "absent" and ctx["reason"] is None
    with pytest.raises(KeyError):
        ctx["best_plan"]
    ctx["event_labels"] = PlanLabels(mode="drain")
    ctx.severity = "OK"
    assert ctx.event_labels.mode == "drain" and ctx["severity"] == "OK"
    assert list(ctx.keys()) == ["aggregate", "mode", "reason", "severity", "event_labels"]   # field order


def test_plan_context_rejects_unknown_fields():
    with pytest.raises(TypeError, match="unknown PlanContext field"):
        PlanContext(attempt=1)
    with pytest.raises(KeyError):
        PlanContext()["attempt"] = 1


def test_plan_labels_of_a_context_are_its_own_verdict():
    ctx = PlanContext(mode="mad", detail_status="UNBALANCED", severity="WARNING",
                      best_improvement=1.5, improvement_threshold=0.15)
    assert PlanLabels.of(ctx) == PlanLabels("mad", "UNBALANCED", "WARNING", 1.5, 0.15)


def test_a_context_without_its_planning_rule_is_refused_as_a_key_error():
    assert issubclass(PlanContextIncomplete, KeyError)


# --- STEP 2: domain/balance package keeps the public API -----------------------------

def test_balance_package_re_exports_every_rc11_name():
    import json
    owners = json.loads((Path(__file__).with_name("api_owners.json")).read_text())["balance"]
    for name, owner in owners.items():
        if owner == "domain.balance":
            assert callable(getattr(balance, name)), name
    for name in ("check_direct_destination", "plan_soft_aa_heal", "find_soft_aa_heal_move",
                 "soft_aa_heal_mad_check", "severity_for_exit_code", "SOFT_AA_HEAL_DETAIL_STATUS",
                 "SOFT_AA_UNRESOLVABLE", "SOFT_AA_DEFERRED"):
        assert hasattr(balance, name), name
    assert sorted(balance.__all__) == sorted(set(balance.__all__))


# --- STEP 3: RuntimeSettings is canonical, module constants mirror it ----------------

def test_runtime_defaults_are_the_rc11_module_defaults_with_their_types():
    rc11 = {"ram_hotspot_threshold": 80, "ram_critical_threshold": 95.0, "cpu_hotspot_threshold": 70.0,
            "auto_pressure_threshold": 80.0, "target_avg_mad": 3, "min_improvement_threshold": 0.15,
            "placement_pressure_threshold": 90.0, "drain_max_host_ram_pct": 80, "drain_max_host_cpu_pct": 70.0,
            "drain_absolute_max_host_ram_pct": 95.0, "heal_soft_anti_affinity": True,
            "soft_aa_heal_mad_tolerance": 0.05, "soft_aa_tiebreak_margin": 0.15, "migration_timeout": 600,
            "migration_max_wait": 1800}
    defaults = dataclasses.asdict(C.RuntimeSettings())
    assert defaults == rc11 and set(C.RUNTIME_CONSTANTS.values()) == set(rc11)
    # Types matter for output text: 600 prints "600s", 600.0 would print "600.0s".
    assert {k: type(v) for k, v in defaults.items()} == {k: type(v) for k, v in rc11.items()}


def test_apply_functions_replace_the_runtime_settings_and_the_mirror():
    before = C.runtime()
    C.apply_thresholds(C.Thresholds(ram_hotspot=75, ram_critical=85, cpu_hotspot=65, pressure=60))
    C.apply_soft_aa_heal(C.SoftAntiAffinityHeal(enabled=False, mad_tolerance=0.2, tiebreak_margin=0.3))
    C.apply_migration_timeouts(C.MigrationTimeouts(no_progress=300, maximum=900))
    live = C.runtime()
    assert live is not before and before.ram_hotspot_threshold == 80        # frozen, replaced
    assert (live.ram_hotspot_threshold, live.drain_max_host_ram_pct, live.drain_absolute_max_host_ram_pct,
            live.auto_pressure_threshold) == (75.0, 75.0, 85.0, 60.0)
    assert (live.heal_soft_anti_affinity, live.soft_aa_heal_mad_tolerance, live.migration_max_wait) == (
        False, 0.2, 900.0)
    for name, field in C.RUNTIME_CONSTANTS.items():
        assert getattr(C, name) == getattr(live, field), name
    with pytest.raises(dataclasses.FrozenInstanceError):
        live.ram_hotspot_threshold = 1


def test_assigning_a_mirrored_constant_updates_runtime(monkeypatch):
    monkeypatch.setattr(C, "SOFT_AA_HEAL_MAD_TOLERANCE", 0.5)
    assert C.runtime().soft_aa_heal_mad_tolerance == 0.5
    assert balance.soft_aa_heal_mad_limit(3.5) == 4.0                       # readers see it
    C.CPU_HOTSPOT_THRESHOLD = 50.0
    assert C.runtime().cpu_hotspot_threshold == 50.0
    C.SOME_OTHER_NAME = 1                                                  # plain attributes untouched
    assert C.SOME_OTHER_NAME == 1 and "some_other_name" not in C.RUNTIME_CONSTANTS.values()
    del C.SOME_OTHER_NAME
    with pytest.raises(TypeError):
        C.set_runtime({"ram_hotspot_threshold": 1})


def test_no_source_module_imports_a_config_value_by_name():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "compute_maintenance.config":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno} {[a.name for a in node.names]}")
    assert offenders == []


# --- STEP 4: one severity / status mapping ---------------------------------------------

def test_severity_mapping_is_defined_once_and_shared():
    import compute_maintenance.application.monitoring as MON
    from compute_maintenance.domain import severity
    defined = {}
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name in (
                    "map_detail_status_to_severity", "map_severity_to_exit_code", "severity_for_exit_code",
                    "worst_exit_code", "heal_verdict", "drain_detail_status"):
                defined.setdefault(node.name, []).append(path.relative_to(SRC).as_posix())
    assert {name: paths for name, paths in defined.items()} == {
        name: ["domain/severity.py"] for name in defined} and len(defined) == 6
    assert balance.map_detail_status_to_severity is severity.map_detail_status_to_severity
    assert balance.map_severity_to_exit_code is severity.map_severity_to_exit_code
    assert balance.severity_for_exit_code is severity.severity_for_exit_code
    assert MON.worst_exit_code is severity.worst_exit_code
    assert MON.EXIT_CODE_SEVERITY is severity.EXIT_CODE_SEVERITY


# rc11 contract table: detail status (+ reason) -> severity -> Icinga exit code.
RC11_SEVERITY = {
    ("BALANCED", None): ("OK", 0), ("BALANCED_OPTIMIZABLE", None): ("OK", 0),
    ("UNBALANCED", None): ("WARNING", 1), ("RAM_PRESSURE", None): ("WARNING", 1),
    ("CPU_PRESSURE", None): ("WARNING", 1), ("RAM_HOTSPOT_OPTIMIZABLE", None): ("WARNING", 1),
    ("CPU_HOTSPOT_OPTIMIZABLE", None): ("WARNING", 1), ("SOFT_AA_HEAL", None): ("WARNING", 1),
    ("PRESSURE_NO_VALID_MOVE", None): ("CRITICAL", 2), ("RAM_HOTSPOT_NO_VALID_MOVE", None): ("CRITICAL", 2),
    ("CPU_HOTSPOT_NO_VALID_MOVE", None): ("CRITICAL", 2),
    ("UNBALANCED_NO_VALID_MOVE", "no_technical_candidate"): ("CRITICAL", 2),
    ("UNBALANCED_NO_VALID_MOVE", "no_candidate_above_improvement_threshold"): ("WARNING", 1),
    ("UNBALANCED_NO_VALID_MOVE", None): ("WARNING", 1),
    ("UNKNOWN", None): ("UNKNOWN", 3), ("DRAIN_FEASIBLE", None): ("UNKNOWN", 3), (None, None): ("UNKNOWN", 3),
}


@pytest.mark.parametrize("status,reason", list(RC11_SEVERITY))
def test_severity_table_is_the_rc11_contract(status, reason):
    from compute_maintenance.domain import severity
    sev = severity.map_detail_status_to_severity(status, reason)
    assert (sev, severity.map_severity_to_exit_code(sev)) == RC11_SEVERITY[(status, reason)]
    assert severity.severity_for_exit_code(severity.map_severity_to_exit_code(sev)) == sev


@pytest.mark.parametrize("load_code,expected", [(0, ("WARNING", 1)), (1, ("WARNING", 1)),
                                                (2, ("CRITICAL", 2)), (3, ("WARNING", 1))])
def test_heal_verdict_never_masks_a_critical_load_state(load_code, expected):
    from compute_maintenance.domain import severity
    assert severity.heal_verdict(load_code) == expected


def test_monitor_unknown_line_is_the_rc11_text():
    import compute_maintenance.application.monitoring as MON
    assert MON.unknown_line("agg", "less_than_2_hosts") == (
        "UNKNOWN - aggregate=agg, detail=UNKNOWN, reason=less_than_2_hosts")


# --- STEP 5: hermetic default config (WLB_CONFIG_PATH) ---------------------------------

@pytest.fixture
def unreadable_etc_config(monkeypatch):
    """Simulate a host where /etc/wbalancer/wbalancer.conf exists but is root-only
    (mode 600) and the suite runs unprivileged: reading it raises EACCES."""
    import compute_maintenance.adapters.settings as SETTINGS
    real = SETTINGS.dotenv_values

    def dotenv_values(path, *args, **kwargs):
        if str(path).replace("\\", "/") == "/etc/wbalancer/wbalancer.conf":
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(SETTINGS, "dotenv_values", dotenv_values)
    monkeypatch.delenv("OS_CLOUD", raising=False)       # else the file is never read


def test_reading_the_root_only_default_really_fails_without_the_override(unreadable_etc_config, monkeypatch):
    # Negative control: this is how the three CLI tests failed before rc12.
    from compute_maintenance.cli.maintain import parse_args
    monkeypatch.delenv("WLB_CONFIG_PATH")
    with pytest.raises(SystemExit) as exc:
        parse_args(["--profile", "unused", "h1"])
    assert exc.value.code == 2


def test_the_three_cli_tests_pass_with_an_unreadable_etc_config(unreadable_etc_config, capsys):
    import tests.test_rc9_paths as rc9
    import tests.test_rc10_review_fixes as rc10
    rc9.test_default_layout_without_v1_suffix()
    for flag in ("--ssh-strict-host-key", "--no-ssh-strict-host-key"):
        rc10.test_rolling_maint_rejects_the_removed_flag(capsys, flag)


def test_wlb_config_path_is_the_default_config_of_both_clis(tmp_path, monkeypatch):
    from click.testing import CliRunner

    import compute_maintenance.cli.balance as CLI
    from compute_maintenance.cli.maintain import parse_args
    from compute_maintenance.paths import default_config_file
    conf = tmp_path / "w.conf"
    conf.write_text("OS_CLOUD=prod\n", encoding="utf-8")
    other = tmp_path / "other.conf"
    other.write_text("OS_CLOUD=lab\n", encoding="utf-8")
    monkeypatch.delenv("OS_CLOUD", raising=False)
    monkeypatch.setenv("WLB_CONFIG_PATH", str(conf))
    assert default_config_file() == str(conf)
    args = parse_args(["--profile", "unused"])
    assert (args.wlb_config, args.os_cloud) == (str(conf), "prod")
    assert parse_args(["--profile", "unused", "--wlb-config", str(other)]).os_cloud == "lab"   # option wins

    seen = []
    monkeypatch.setattr(CLI._config, "resolve_os_cloud", lambda cli, path: seen.append(path) or "x")
    monkeypatch.setattr(CLI._config.LoadbalancerConfig, "load_config",
                        classmethod(lambda cls, path: (_ for _ in ()).throw(SystemExit(7))))
    CliRunner().invoke(CLI.main, ["--monitor-only"])
    CliRunner().invoke(CLI.main, ["--monitor-only", "--config", str(other)])
    assert seen == [str(conf), str(other)]
    monkeypatch.setenv("WLB_CONFIG_PATH", "")                 # empty = unset
    assert default_config_file() == "/etc/wbalancer/wbalancer.conf"
