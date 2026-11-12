"""rc10: fixes and test gaps from the independent review of 2026-11-11
(wbalancer-v1/REVIEW-INDEPENDENT-2026-11-11.md).

M1  one cloud resolution (--os-cloud > env OS_CLOUD > OS_CLOUD in the config
    file > "openstack") for the OpenStack connection AND the drain-child result
    identity; rolling-maint always passes its cloud to the child.
M2  --monitor-only combines aggregates by severity CRITICAL > WARNING > UNKNOWN > OK.
M3  the pre-submit refresh rebuilds the baseline with the planner's own
    exclude_unready_hosts.
M4  no StrictHostKeyChecking injected: the coordinator's OpenSSH config decides;
    --ssh-strict-host-key removed.
Gaps 1-4: journal-before-submit, snapshot fail-closed, freshness gate wiring
    in apply mode, Placement allocation_ratio math.
"""
import json
from types import SimpleNamespace as NS

import pytest
from click.testing import CliRunner

import compute_maintenance.application.migrations as MIG
import compute_maintenance.config as C
from compute_maintenance.domain.errors import ConfigError
from compute_maintenance.domain.plan_context import PlanContext

LIVE = ("RAM_HOTSPOT_THRESHOLD", "RAM_CRITICAL_THRESHOLD", "CPU_HOTSPOT_THRESHOLD",
        "AUTO_PRESSURE_THRESHOLD", "TARGET_AVG_MAD", "MIN_IMPROVEMENT_THRESHOLD",
        "PLACEMENT_PRESSURE_THRESHOLD", "DRAIN_MAX_HOST_RAM_PCT", "DRAIN_MAX_HOST_CPU_PCT",
        "DRAIN_ABSOLUTE_MAX_HOST_RAM_PCT", "MIGRATION_TIMEOUT", "MIGRATION_MAX_WAIT")


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    # The CLI applies config thresholds to module globals; monkeypatch restores them.
    for name in LIVE:
        monkeypatch.setattr(C, name, getattr(C, name))
    monkeypatch.delenv("OS_CLOUD", raising=False)


def write_config(tmp_path, **extra):
    lines = ["PROMETHEUS_QUERY_URL=http://prom.test:9090/api/v1/query",
             "PROMETHEUS_NODE_JOB=node", "PROMETHEUS_LIBVIRT_JOB=libvirt"]
    lines += [f"{k}={v}" for k, v in extra.items()]
    path = tmp_path / "wbalancer.conf"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# --- M1: cloud resolution -------------------------------------------------------------

def test_resolve_os_cloud_precedence_cli_env_config_default(tmp_path, monkeypatch):
    conf = write_config(tmp_path, OS_CLOUD="from-config")
    assert C.resolve_os_cloud(None, None) == "openstack"
    assert C.resolve_os_cloud(None, str(tmp_path / "missing.conf")) == "openstack"
    assert C.resolve_os_cloud(None, conf) == "from-config"
    monkeypatch.setenv("OS_CLOUD", "from-env")
    assert C.resolve_os_cloud(None, conf) == "from-env"
    assert C.resolve_os_cloud(" from-cli ", conf) == "from-cli"
    # An empty OS_CLOUD in the environment counts as unset: the file decides.
    monkeypatch.setenv("OS_CLOUD", "")
    assert C.resolve_os_cloud(None, conf) == "from-config"


@pytest.mark.parametrize("cli,env,conf_value,message", [
    ("  ", None, None, "--os-cloud must not be empty"),
    ("", None, None, "--os-cloud must not be empty"),
    (None, "   ", None, "OS_CLOUD in the environment must not be empty"),
    (None, None, '"   "', "must not be empty"),
])
def test_resolve_os_cloud_rejects_a_blank_value_that_is_set(tmp_path, monkeypatch, cli, env, conf_value, message):
    if env is not None:
        monkeypatch.setenv("OS_CLOUD", env)
    conf = write_config(tmp_path, **({"OS_CLOUD": conf_value} if conf_value else {}))
    with pytest.raises(ConfigError, match=message):
        C.resolve_os_cloud(cli, conf)


def test_load_config_uses_the_same_resolution(tmp_path, monkeypatch):
    conf = write_config(tmp_path, OS_CLOUD="prod")
    assert C.LoadbalancerConfig.load_config(conf).os_cloud == C.resolve_os_cloud(None, conf) == "prod"
    monkeypatch.setenv("OS_CLOUD", "")
    assert C.LoadbalancerConfig.load_config(conf).os_cloud == "prod"


def run_drain_child(tmp_path, monkeypatch, conf, extra=()):
    """Run the wbalancer CLI as rolling-maint's drain child (dry-run) and
    return (cloud the run really used, cloud in the result identity)."""
    import compute_maintenance.cli.balance as CLI
    used = {}

    def fake_drain(cfg, **kw):
        used["cloud"] = cfg.os_cloud   # what get_openstack_connection() receives
        return 0
    monkeypatch.setattr(CLI._execution_workflow, "drain_by_aggregate", fake_drain)
    result_file = tmp_path / "result.json"
    args = ["--config", conf, "--drain-host", "h1", "-a", "agg", "--dry-run",
            "--cooldown-file", str(tmp_path / "cool.json"), "--events-file", str(tmp_path / "e.jsonl"),
            "--run-id", "run-1", "--result-file", str(result_file), *extra]
    result = CliRunner().invoke(CLI.main, args)
    assert result.exit_code == 0, result.output
    return used["cloud"], json.loads(result_file.read_text(encoding="utf-8"))["cloud"]


def test_child_identity_is_the_config_file_cloud_without_cli_or_env(tmp_path, monkeypatch):
    # Review M1 scenario: env empty, no --os-cloud, OS_CLOUD=prod in the config.
    # Before rc10 the run used "prod" but reported identity "openstack".
    conf = write_config(tmp_path, OS_CLOUD="prod")
    assert run_drain_child(tmp_path, monkeypatch, conf) == ("prod", "prod")


def test_child_identity_follows_cli_over_env_over_config(tmp_path, monkeypatch):
    conf = write_config(tmp_path, OS_CLOUD="prod")
    monkeypatch.setenv("OS_CLOUD", "dr-site")
    assert run_drain_child(tmp_path, monkeypatch, conf) == ("dr-site", "dr-site")
    assert run_drain_child(tmp_path, monkeypatch, conf, ["--os-cloud", "lab"]) == ("lab", "lab")


def test_child_with_blank_config_cloud_writes_no_result(tmp_path, monkeypatch):
    # Unresolvable cloud: the child stops before writing any result, so the
    # parent rejects the run (missing result = outcome UNKNOWN), fail-closed.
    import compute_maintenance.cli.balance as CLI
    conf = write_config(tmp_path, OS_CLOUD='"  "')
    result = CliRunner().invoke(CLI.main, [
        "--config", conf, "--drain-host", "h1", "-a", "agg", "--dry-run",
        "--cooldown-file", str(tmp_path / "cool.json"),
        "--run-id", "run-1", "--result-file", str(tmp_path / "result.json")])
    assert result.exit_code == 1
    assert not (tmp_path / "result.json").exists()


def test_rolling_maint_resolves_its_cloud_like_wbalancer(tmp_path, monkeypatch):
    from compute_maintenance.cli.maintain import parse_args
    conf = write_config(tmp_path, OS_CLOUD="prod")
    base = ["--profile", "unused", "--wlb-config", conf]
    assert parse_args(base).os_cloud == "prod"                        # config file
    assert parse_args(["--profile", "unused", "--wlb-config", str(tmp_path / "none")]).os_cloud == "openstack"
    monkeypatch.setenv("OS_CLOUD", "dr-site")
    assert parse_args(base).os_cloud == "dr-site"                     # env beats config
    assert parse_args(base + ["--os-cloud", "lab"]).os_cloud == "lab"  # CLI beats env


def test_custom_child_always_gets_the_parent_cloud(tmp_path):
    from compute_maintenance.application.stages import StageActions

    def cmd(os_cloud, wbalancer):
        args = NS(wbalancer=wbalancer, wlb_config="/etc/w.conf", wlb_events_file="/tmp/e.jsonl",
                  drain_allow_hotspot=False, notify="off", wlb_extra=None, os_cloud=os_cloud)
        return StageActions.wbalancer_cmd(NS(args=args, aggregate_for=lambda h: "agg"), "h1", True)

    for wbalancer in (None, "/opt/custom/wbalancer.py"):
        for cloud in ("openstack", "prod"):
            line = cmd(cloud, wbalancer)
            assert line.count("--os-cloud") == 1
            assert line[line.index("--os-cloud") + 1] == cloud


def test_parent_and_child_agree_end_to_end_on_the_config_cloud(tmp_path, monkeypatch):
    """rolling-maint resolves "prod" from --wlb-config, hands it to a custom
    child, the child reports "prod": read_result() with the parent's expected
    identity accepts it."""
    from compute_maintenance.adapters.protocol import read_result
    from compute_maintenance.application.stages import StageActions
    from compute_maintenance.cli.maintain import parse_args
    conf = write_config(tmp_path, OS_CLOUD="prod")
    parent = parse_args(["--profile", "unused", "--wlb-config", conf])
    parent.wbalancer, parent.notify, parent.wlb_extra, parent.drain_allow_hotspot = (
        "/opt/custom/wbalancer.py", "off", None, False)
    line = StageActions.wbalancer_cmd(NS(args=parent, aggregate_for=lambda h: "agg"), "h1", True)
    child_cloud = line[line.index("--os-cloud") + 1]
    used, identity = run_drain_child(tmp_path, monkeypatch, conf, ["--os-cloud", child_cloud])
    assert used == identity == parent.os_cloud == "prod"
    expected = {"run_id": "run-1", "cloud": parent.os_cloud, "aggregate": "agg", "host": "h1",
                "mode": "dry-run"}
    data = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert all(data[k] == v for k, v in expected.items())
    # A child that reported another cloud is refused (exit 1: no summary checks).
    (tmp_path / "result.json").write_text(json.dumps(dict(data, cloud="openstack", exit_code=1)),
                                          encoding="utf-8")
    from compute_maintenance.adapters.persistence import StateError
    with pytest.raises(StateError, match="mismatch"):
        read_result(tmp_path / "result.json", expected, 1)


# --- M2: monitor exit code severity -------------------------------------------------------

OK, WARNING, CRITICAL, UNKNOWN = C.ICINGA_OK, C.ICINGA_WARNING, C.ICINGA_CRITICAL, C.ICINGA_UNKNOWN


@pytest.mark.parametrize("codes,expected", [
    ([CRITICAL, UNKNOWN], CRITICAL),
    ([UNKNOWN, CRITICAL], CRITICAL),
    ([WARNING, UNKNOWN], WARNING),
    ([OK, UNKNOWN], UNKNOWN),
    ([OK], OK),
    ([OK, WARNING, CRITICAL, UNKNOWN], CRITICAL),
    ([], OK),
    ([OK, 7], UNKNOWN),          # an out-of-range code is treated as UNKNOWN
])
def test_worst_exit_code_ranks_severity_not_number(codes, expected):
    from compute_maintenance.application.monitoring import worst_exit_code
    assert worst_exit_code(codes) == expected


def monitor(monkeypatch, aggregates, outcomes):
    """monitor_by_aggregate() over fake aggregates; ``outcomes`` maps an
    aggregate to its exit code, or to an exception its evaluation raises."""
    import compute_maintenance.application.monitoring as M
    monkeypatch.setattr(M._sources_openstack, "get_openstack_connection", lambda cloud: object())
    monkeypatch.setattr(M._sources_openstack, "get_aggregate_map", lambda conn, **kw: dict(aggregates))
    monkeypatch.setattr(M._sources_openstack, "build_selected_host_list",
                        lambda amap: sorted({h for hosts in amap.values() for h in hosts}))
    monkeypatch.setattr(M._snapshot, "build_cluster_cache", lambda *a, **k: {})
    names = {OK: "OK", WARNING: "WARNING", CRITICAL: "CRITICAL", UNKNOWN: "UNKNOWN"}

    def evaluate(**kw):
        outcome = outcomes[kw["agg_name"]]
        if isinstance(outcome, Exception):
            raise outcome
        return PlanContext(detail_status="X", severity=names[outcome], exit_code=outcome,
                           message=f"{names[outcome]} - aggregate={kw['agg_name']}")
    monkeypatch.setattr(M._snapshot, "evaluate_aggregate_state", evaluate)
    return M.monitor_by_aggregate(NS(os_cloud="openstack"))


def test_monitor_critical_aggregate_is_not_masked_by_a_data_error(monkeypatch, capsys):
    from compute_maintenance.domain.errors import PrometheusError
    code = monitor(monkeypatch, {"a": ["h1", "h2"], "b": ["h3", "h4"], "c": ["h5"]},
                   {"a": CRITICAL, "b": PrometheusError("series missing")})
    out = capsys.readouterr().out
    assert code == CRITICAL
    # Per-aggregate detail is kept: the UNKNOWN aggregates still say so.
    assert "CRITICAL - aggregate=a" in out
    assert "UNKNOWN - aggregate=b" in out and "series missing" in out
    assert "UNKNOWN - aggregate=c" in out and "less_than_2_hosts" in out


def test_monitor_warning_beats_unknown_and_unknown_beats_ok(monkeypatch, capsys):
    two = {"a": ["h1", "h2"], "b": ["h3", "h4"]}
    assert monitor(monkeypatch, two, {"a": WARNING, "b": UNKNOWN}) == WARNING
    assert monitor(monkeypatch, two, {"a": OK, "b": UNKNOWN}) == UNKNOWN
    assert monitor(monkeypatch, two, {"a": OK, "b": OK}) == OK


@pytest.mark.parametrize("code", [OK, WARNING, CRITICAL, UNKNOWN])
def test_monitor_single_aggregate_passes_its_code_through(monkeypatch, capsys, code):
    assert monitor(monkeypatch, {"a": ["h1", "h2"]}, {"a": code}) == code


# --- shared: a real evaluate_aggregate_state over fake Prometheus/Nova/Placement -----------

def host_resource(cores=32, ram_gb=256):
    from compute_maintenance.domain.policies import (
        build_placement_inventory_record as record,
    )
    placement = {"resource_provider_uuid": "rp",
                 "VCPU": record({"total": cores, "reserved": 0, "allocation_ratio": 1.0}, 0, 0),
                 "MEMORY_MB": record({"total": ram_gb * 1024, "reserved": 0, "allocation_ratio": 1.0}, 0, 0)}
    return {"total_cores": cores, "free_cores": cores, "total_ram_gb": ram_gb, "free_ram_gb": ram_gb,
            "running_vms": 0, "placement": placement}


@pytest.fixture
def snapshot_env(monkeypatch, tmp_path):
    """Snapshot inputs a test can break: Prometheus series per query (parsed by
    the real get_metrics), Nova service readiness, Placement/runtime capacity."""
    import compute_maintenance.adapters.prometheus as P
    import compute_maintenance.application.snapshots as S
    from compute_maintenance.adapters.cooldown import CooldownStore
    env = NS(hosts=["h1", "h2", "h3"], series={"CPU": {}, "MEM": {}}, ready={"h1", "h2", "h3"},
             resources={"h1", "h2", "h3"}, resource_errors={}, queries=[])
    env.series["CPU"].update(h1=60.0, h2=20.0, h3=40.0)
    env.series["MEM"].update(h1=50.0, h2=50.0, h3=50.0)

    def do_query(url, query, at=None):
        env.queries.append(query)
        rows = [{"metric": {"alias": host}, "value": [1_700_000_000, str(value)]}
                for host, value in env.series[query].items()]
        return {"status": "success", "data": {"result": rows}}
    monkeypatch.setattr(P, "do_query", do_query)
    monkeypatch.setattr(S._sources_openstack, "is_nova_compute_service_ready",
                        lambda conn, host: (True, "") if host in env.ready else (False, "service disabled"))
    monkeypatch.setattr(S._sources_prometheus, "get_all_vm_metrics", lambda *a, **k: {})
    monkeypatch.setattr(S, "build_aggregate_vm_inventory", lambda *a, **k: ({}, {}, {}))
    env.cfg = NS(prometheus_query_url="http://prom.test/api/v1/query", prometheus_query_cpu_used="CPU",
                 prometheus_query_mem_used="MEM", cooldown_store=CooldownStore(str(tmp_path / "cool.json")),
                 max_instances_per_host={}, max_instances_per_host_default=0,
                 placement_final_allocation_candidate_check=False, planner_candidate_budget=1000,
                 make_room_min_improvement=0.3, enable_make_room=False)

    def cache():
        return {"host_resources": {h: host_resource() for h in env.resources},
                "host_resource_errors": dict(env.resource_errors), "host_az": None,
                "server_cache": {}, "server_groups": None, "project_cache": {}}
    env.cache = cache

    def evaluate(hosts=None, **kw):
        return S.evaluate_aggregate_state(env.cfg, object(), "agg", list(hosts or env.hosts),
                                          enforce_server_groups=False, cluster_cache=cache(), **kw)
    env.evaluate = evaluate
    return env


# --- M3: revalidation refresh follows the planner's exclude_unready_hosts -----------------

def test_snapshot_records_its_planning_context(snapshot_env):
    assert snapshot_env.evaluate(exclude_unready_hosts=True)["exclude_unready_hosts"] is True
    assert snapshot_env.evaluate()["exclude_unready_hosts"] is False


def test_unready_host_changes_the_baseline_only_with_the_wrong_flag(snapshot_env):
    """Why M3 matters: h3 turned unready after planning. With the planner's
    rule (True) it stays out of the baseline; with the old refresh (False)
    its load leaks into the mean/MAD the improvement gate compares against."""
    plan = snapshot_env.evaluate(exclude_unready_hosts=True, plan_balance=False)
    assert plan["hosts_in_agg"] == ["h1", "h2", "h3"]
    snapshot_env.ready.discard("h3")
    good = snapshot_env.evaluate(plan["hosts_in_agg"], exclude_unready_hosts=True, plan_balance=False)
    leaked = snapshot_env.evaluate(plan["hosts_in_agg"], exclude_unready_hosts=False, plan_balance=False)
    assert good["hosts_in_agg"] == ["h1", "h2"] and "h3" in good["excluded_hosts"]
    assert leaked["hosts_in_agg"] == ["h1", "h2", "h3"]
    assert good["cpu_mean"] == pytest.approx(40.0)          # (60 + 20) / 2
    assert good["baseline_cpu_mad"] == pytest.approx(20.0)  # |60-40| and |20-40|
    assert leaked["baseline_cpu_mad"] != pytest.approx(good["baseline_cpu_mad"])


@pytest.fixture
def refresh_spy(monkeypatch, tmp_path):
    """execute_migration_step with a journal; records the refresh call."""
    from compute_maintenance.adapters.cooldown import CooldownStore
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return PlanContext(error="stop after the refresh call")
    monkeypatch.setattr(MIG._snapshot, "evaluate_aggregate_state", evaluate)
    cfg = NS(cooldown_store=CooldownStore(str(tmp_path / "cool.json")), migration_journal=object())
    step = NS(vm_id="vm", src_host="h1", dst_host="h2", vcpus=1, vm_ram_gb=1)
    return NS(calls=calls, cfg=cfg, step=step)


@pytest.mark.parametrize("plan_type,flag", [("direct", True), ("make_room", True), ("drain", False)])
def test_refresh_before_submit_uses_the_plan_flag(refresh_spy, plan_type, flag):
    from compute_maintenance.domain.errors import MigrationExecutionError
    state = PlanContext(aggregate="agg", hosts_in_agg=["h1", "h2"], exclude_unready_hosts=flag)
    with pytest.raises(MigrationExecutionError, match="fresh snapshot failed"):
        MIG.execute_migration_step(refresh_spy.cfg, object(), state, refresh_spy.step,
                                   NS(plan_type=plan_type, steps=[refresh_spy.step]), 1)
    assert len(refresh_spy.calls) == 1
    assert refresh_spy.calls[0]["exclude_unready_hosts"] is flag
    assert refresh_spy.calls[0]["plan_balance"] is False


def test_refresh_without_a_planning_context_is_refused(refresh_spy):
    # A state that does not say how it was planned cannot be revalidated.
    with pytest.raises(KeyError, match="exclude_unready_hosts"):
        MIG.execute_migration_step(refresh_spy.cfg, object(), PlanContext(aggregate="agg", hosts_in_agg=["h1"]),
                                   refresh_spy.step, NS(plan_type="direct", steps=[refresh_spy.step]), 1)
    assert refresh_spy.calls == []


@pytest.mark.parametrize("flag", [True, False])
def test_make_room_step_two_refresh_takes_the_flag_it_is_given(monkeypatch, flag):
    seen = []
    monkeypatch.setattr(MIG._snapshot, "build_cluster_cache", lambda *a, **k: {})
    monkeypatch.setattr(MIG._snapshot, "evaluate_aggregate_state",
                        lambda **kw: seen.append(kw["exclude_unready_hosts"]) or PlanContext(error="stop"))
    cfg = NS(placement_sync_timeout=0, placement_sync_interval=1)
    _, ok, _why = MIG.wait_for_make_room_step(cfg, object(), "agg", ["h1", "h2"], object(),
                                              exclude_unready_hosts=flag)
    assert ok is False and seen == [flag]


def test_balance_passes_the_plan_flag_to_make_room_step_two(monkeypatch):
    import compute_maintenance.application.balancing as BAL
    from compute_maintenance.domain.errors import MigrationExecutionError
    first, second = NS(vm_id="a", src_host="h1", dst_host="h3"), NS(vm_id="b", src_host="h2", dst_host="h1")
    plan = NS(plan_type="make_room", steps=[first, second])
    state = PlanContext(has_valid_move=True, best_plan=plan, exclude_unready_hosts=True, all_vm_names={})
    planned_with, step_two = [], []
    monkeypatch.setattr(BAL._snapshot, "evaluate_aggregate_state",
                        lambda **kw: planned_with.append(kw["exclude_unready_hosts"]) or state)
    for name in ("print_baseline_table", "print_state_summary", "print_move_details",
                 "append_migration_event", "build_migration_event"):
        monkeypatch.setattr(BAL._report, name, lambda *a, **k: None)
    monkeypatch.setattr(BAL._execution_migrate, "execute_migration_step", lambda *a, **k: True)
    monkeypatch.setattr(BAL._execution_migrate, "wait_for_make_room_step",
                        lambda *a, **kw: step_two.append(kw["exclude_unready_hosts"]) or (state, False, "stop"))
    with pytest.raises(MigrationExecutionError, match="step 2 was aborted"):
        BAL.auto_balance_aggregate(NS(), object(), "agg", ["h1", "h2", "h3"], assume_yes=True)
    assert planned_with == [True] and step_two == [True]


def test_fresh_projection_names_a_host_that_left_the_snapshot():
    from compute_maintenance.domain.errors import MigrationExecutionError
    state = PlanContext(all_vm_details={"vm": {"vcpus": 1}}, host_cpu_pct={"h1": 50.0},
                        excluded_hosts={"h2": "service disabled"})
    step = NS(vm_id="vm", src_host="h1", dst_host="h2")
    with pytest.raises(MigrationExecutionError, match="destination host h2 .*service disabled"):
        MIG.validate_fresh_projection(state, step, NS(plan_type="direct"), True)


# --- M4: SSH host-key policy comes from the coordinator's OpenSSH configuration ----------

@pytest.mark.parametrize("key", [None, "/root/.ssh/id_ed25519"])
def test_ssh_command_sets_no_host_key_policy(monkeypatch, key):
    import compute_maintenance.adapters.commands as CMD
    seen = []
    monkeypatch.setattr(CMD, "run", lambda cmd, **kw: seen.append(cmd) or NS(returncode=0, stdout="", stderr=""))
    node = CMD.Node("h1", "10.0.0.11", "root", key, "docker")
    node.ssh("uptime")
    node.reboot()
    node.containers()
    assert len(seen) == 3
    for cmd in seen:
        line = " ".join(cmd)
        assert cmd[0] == "ssh"
        assert "StrictHostKeyChecking" not in line
        assert "UserKnownHostsFile" not in line
        # The other options are unchanged.
        for option in ("BatchMode=yes", "ConnectTimeout=8", "ServerAliveInterval=5", "ServerAliveCountMax=2"):
            assert option in cmd
        assert ("-i" in cmd) is (key is not None)
        assert "root@10.0.0.11" in cmd


def test_no_host_key_option_left_in_any_remote_command_builder():
    # Guards a later re-introduction in another ssh/scp call site.
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "src" / "compute_maintenance"
    hits = [str(p.relative_to(root)) for p in root.rglob("*.py")
            if "StrictHostKeyChecking=" in p.read_text(encoding="utf-8")
            or "UserKnownHostsFile=" in p.read_text(encoding="utf-8")]
    assert hits == []


def test_node_no_longer_takes_a_strict_flag():
    import compute_maintenance.adapters.commands as CMD
    with pytest.raises(TypeError):
        CMD.Node("h1", "10.0.0.11", "root", None, "docker", True)


@pytest.mark.parametrize("flag", ["--ssh-strict-host-key", "--no-ssh-strict-host-key"])
def test_rolling_maint_rejects_the_removed_flag(capsys, flag):
    from compute_maintenance.cli.maintain import parse_args
    with pytest.raises(SystemExit) as exc:
        parse_args(["--profile", "unused", flag, "h1"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not hasattr(parse_args(["--profile", "unused", "h1"]), "ssh_strict_host_key")


# --- Test gap 1: the journal intent is durable BEFORE the live-migration submit ----------

class Rejected(Exception):
    status_code = 409   # a definitive Nova rejection: no monitoring needed


class FakeNova:
    """Minimal Nova; live_migrate_server() looks at what is on disk at submit time."""

    def __init__(self, journal_dir, order):
        self.compute = self
        self.journal_dir, self.order = journal_dir, order
        self.pending_on_disk_at_submit = "never submitted"

    def server_migrations(self, _vm):
        return []

    def migrations(self, **_query):
        return [{"id": 41, "status": "completed", "migration_type": "live-migration"}]

    def server_actions(self, _vm):
        return []

    def live_migrate_server(self, **_kw):
        from compute_maintenance.application.recovery import MigrationJournal
        self.order.append("live_migrate_server")
        # What a crash at this very moment would leave behind for reconcile():
        self.pending_on_disk_at_submit = MigrationJournal(self.journal_dir).state.data.get("pending")
        raise Rejected("409 Conflict")


def fresh_state(collected_at=None, **extra):
    import time
    state = {"aggregate": "agg", "hosts_in_agg": ["src", "dst"], "exclude_unready_hosts": True,
             "collected_at": time.time() if collected_at is None else collected_at,
             "all_vm_names": {"vm": "web-1"},
             "all_vm_details": {"vm": {"vcpus": 2, "cpu_pct": 10.0, "ram_gb": 4.0, "ram_pct": 50.0}},
             "host_cpu_pct": {"src": 60.0, "dst": 20.0}, "host_ram_pct": {"src": 50.0, "dst": 40.0},
             "host_cores": {"src": 32, "dst": 32}, "host_ram": {"src": 256.0, "dst": 256.0}}
    state.update(extra)
    return PlanContext(**state)


@pytest.fixture
def submit_harness(monkeypatch, tmp_path):
    from compute_maintenance.adapters.cooldown import CooldownStore
    from compute_maintenance.application.recovery import MigrationJournal
    order, events = [], []
    journal = MigrationJournal(str(tmp_path / "operations"), 360)
    begin = journal.begin

    def recording_begin(*args, **kwargs):
        order.append("journal.begin")
        return begin(*args, **kwargs)
    journal.begin = recording_begin
    cfg = NS(cooldown_store=CooldownStore(str(tmp_path / "cool.json")), migration_journal=journal)
    monkeypatch.setattr(MIG._snapshot, "evaluate_aggregate_state", lambda *a, **k: fresh_state())
    monkeypatch.setattr(MIG, "validate_migration_step", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(MIG._report, "append_migration_event", events.append)
    step = NS(vm_id="vm", src_host="src", dst_host="dst", vcpus=2, vm_ram_gb=4.0, reason="test")
    plan = NS(plan_type="direct", steps=[step], improvement=1.0)
    return NS(cfg=cfg, journal=journal, order=order, events=events, step=step, plan=plan,
              nova=FakeNova(str(tmp_path / "operations"), order))


def test_journal_intent_is_on_disk_before_the_submit(submit_harness):
    h = submit_harness
    assert MIG.execute_migration_step(h.cfg, h.nova, fresh_state(), h.step, h.plan, 1) is False
    assert h.order == ["journal.begin", "live_migrate_server"]
    pending = h.nova.pending_on_disk_at_submit
    assert isinstance(pending, dict), "no durable intent existed when Nova received the submit"
    assert (pending["vm_id"], pending["src_host"], pending["dst_host"]) == ("vm", "src", "dst")
    assert pending["global_max_id"] == 41          # reconcile only trusts rows above this id
    # The definitive 409 then closes the intent.
    assert h.journal.state.data["pending"] is None
    assert h.journal.state.data["last_outcome"] == "rejected"


def test_no_submit_when_the_intent_cannot_be_persisted(submit_harness):
    h = submit_harness

    def disk_full(*args, **kwargs):
        h.order.append("journal.begin")
        raise OSError("No space left on device")
    h.cfg.migration_journal.begin = disk_full
    with pytest.raises(OSError, match="No space left"):
        MIG.execute_migration_step(h.cfg, h.nova, fresh_state(), h.step, h.plan, 1)
    assert h.order == ["journal.begin"]            # Nova never saw a request


def test_no_new_submit_while_an_earlier_intent_is_pending(submit_harness):
    from compute_maintenance.domain.errors import MigrationExecutionError
    h = submit_harness
    h.journal.state.save(pending={"vm_id": "other", "src_host": "a", "dst_host": "b"})
    with pytest.raises(MigrationExecutionError, match="pending migration must be reconciled"):
        MIG.execute_migration_step(h.cfg, h.nova, fresh_state(), h.step, h.plan, 1)
    assert "live_migrate_server" not in h.order


# --- Test gap 2: snapshot-level fail-closed on missing/invalid Prometheus or Placement ----

def drop(series, host):
    series.pop(host, None)


@pytest.mark.parametrize("breakage,error", [
    (lambda env: drop(env.series["CPU"], "h3"), "missing_or_invalid_host_metrics:h3"),
    (lambda env: drop(env.series["MEM"], "h2"), "missing_or_invalid_host_metrics:h2"),
    (lambda env: (env.series["CPU"].clear(), env.series["MEM"].clear()),
     "missing_or_invalid_host_metrics:h1,h2,h3"),                       # empty result sets
    (lambda env: env.series["CPU"].update(h1=250.0), "missing_or_invalid_host_metrics:h1"),
    (lambda env: (env.resources.discard("h2"),
                  env.resource_errors.update(h2="h2: no Placement resource provider")),
     "missing_host_resources:h2: no Placement resource provider"),
    (lambda env: env.ready.intersection_update({"h1"}), "less_than_2_destination_ready_hosts:h2,h3"),
])
def test_snapshot_fails_closed_without_a_plan(snapshot_env, breakage, error):
    breakage(snapshot_env)
    state = snapshot_env.evaluate(exclude_unready_hosts=True)
    assert state["error"] == error
    assert "best_plan" not in state and "has_valid_move" not in state


def test_snapshot_with_a_single_measured_host_fails_closed(snapshot_env):
    state = snapshot_env.evaluate(["h1"], exclude_unready_hosts=True)
    assert state["error"] == "less_than_2_hosts_with_metrics"


@pytest.mark.parametrize("value", ["NaN", "+Inf", "not-a-number"])
def test_non_finite_host_sample_raises_instead_of_planning(snapshot_env, value):
    from compute_maintenance.domain.errors import PrometheusError
    snapshot_env.series["MEM"]["h2"] = value
    with pytest.raises(PrometheusError, match="h2"):
        snapshot_env.evaluate(exclude_unready_hosts=True)


@pytest.mark.parametrize("breakage", [
    lambda env: drop(env.series["CPU"], "h3"),
    lambda env: env.series["MEM"].update(h2="NaN"),
])
def test_monitor_reports_unknown_for_a_broken_snapshot(snapshot_env, breakage):
    import compute_maintenance.application.monitoring as MON
    breakage(snapshot_env)
    result = MON.monitor_aggregate(snapshot_env.cfg, object(), "agg", snapshot_env.hosts,
                                   enforce_server_groups=False, cluster_cache=snapshot_env.cache())
    assert result["exit_code"] == C.ICINGA_UNKNOWN
    assert result["message"].startswith("UNKNOWN - aggregate=agg")


def test_balance_and_drain_refuse_to_plan_on_a_broken_snapshot(snapshot_env, monkeypatch):
    import compute_maintenance.application.balancing as BAL
    from compute_maintenance.domain.errors import OpenStackDataError
    drop(snapshot_env.series["CPU"], "h3")
    monkeypatch.setattr(BAL._execution_migrate, "execute_migration_step",
                        lambda *a, **k: pytest.fail("nothing may be submitted"))
    with pytest.raises(OpenStackDataError, match="missing_or_invalid_host_metrics:h3"):
        BAL.auto_balance_aggregate(snapshot_env.cfg, object(), "agg", snapshot_env.hosts,
                                   enforce_server_groups=False, cluster_cache=snapshot_env.cache(),
                                   assume_yes=True)
    with pytest.raises(OpenStackDataError, match="missing_or_invalid_host_metrics:h3"):
        BAL.evaluate_drain(snapshot_env.cfg, object(), "agg", snapshot_env.hosts, "h1",
                           enforce_server_groups=False, cluster_cache=snapshot_env.cache())


# --- Test gap 3: the freshness gate is wired in apply mode (CLI -> journal -> gate) -------

@pytest.fixture
def apply_run(monkeypatch, tmp_path):
    """wbalancer -y through the real CLI wiring and balance workflow; only the
    cloud edges (Nova connection, snapshot collection, Nova re-reads) are faked."""
    import contextlib
    import time

    import compute_maintenance.application.balancing as BAL
    import compute_maintenance.cli.balance as CLI
    run = NS(submitted=[], refresh_age=0.0, cooldown=tmp_path / "state" / "vm-cooldown.json",
             events=tmp_path / "events.jsonl", journal_dir=tmp_path / "state" / "operations")
    run.cooldown.parent.mkdir(parents=True)
    step = NS(vm_id="vm", src_host="src", dst_host="dst", vcpus=2, vm_ram_gb=4.0, reason="balance")
    plan = NS(plan_type="direct", steps=[step], improvement=1.0)

    class Nova:
        compute = None

        def server_migrations(self, _vm):
            return []

        def migrations(self, **_q):
            return []

        def server_actions(self, _vm):
            return []

        def live_migrate_server(self, server, host, **_kw):
            run.submitted.append((server, host))
            raise Rejected("409 Conflict")   # keep the positive control short
    nova = Nova()
    nova.compute = nova

    def evaluate(*args, **kwargs):
        if kwargs.get("plan_balance", True):      # the planning snapshot
            return fresh_state(has_valid_move=True, best_plan=plan)
        return fresh_state(collected_at=time.time() - run.refresh_age)   # pre-submit refresh

    monkeypatch.setattr(CLI._locking, "shared_apply_lock", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(CLI._locking, "acquire_single_instance_lock", lambda path: (True, None))
    monkeypatch.setattr(BAL._sources_openstack, "get_openstack_connection", lambda cloud: nova)
    monkeypatch.setattr(BAL._sources_openstack, "get_aggregate_map", lambda conn, **kw: {"agg": ["src", "dst"]})
    monkeypatch.setattr(BAL._snapshot, "build_cluster_cache", lambda *a, **k: {})
    monkeypatch.setattr(BAL._snapshot, "evaluate_aggregate_state", evaluate)
    for name in ("print_baseline_table", "print_state_summary", "print_move_details"):
        monkeypatch.setattr(BAL._report, name, lambda *a, **k: None)
    monkeypatch.setattr(MIG, "validate_migration_step", lambda *a, **k: (True, "ok"))
    monkeypatch.delenv("RUNTIME_SETTLE_SECONDS", raising=False)

    def invoke(*extra):
        conf = write_config(tmp_path)
        return CliRunner().invoke(CLI.main, ["--config", conf, "--cooldown-file", str(run.cooldown),
                                             "--events-file", str(run.events),
                                             "--lock-file", str(tmp_path / "w.lock"), *extra])
    run.invoke = invoke

    def events():
        if not run.events.exists():
            return []
        return [json.loads(line) for line in run.events.read_text(encoding="utf-8").splitlines()]
    run.read_events = events
    return run


def test_apply_mode_refuses_a_stale_snapshot_before_submit(apply_run):
    from compute_maintenance.application.recovery import MigrationJournal
    apply_run.refresh_age = 121.0                   # the refresh took longer than the 120 s budget
    result = apply_run.invoke("-y")
    assert result.exit_code == 0, result.output     # refused before submit: nothing failed
    assert apply_run.submitted == []
    refused = [e for e in apply_run.read_events() if e["result"] == "step_revalidation_failed"]
    assert len(refused) == 1 and "snapshot expired before submit" in refused[0]["detail"]
    # The wiring created the journal next to the cooldown file; no intent was written.
    assert MigrationJournal(str(apply_run.journal_dir)).state.data.get("pending") is None
    rejected = json.loads(apply_run.cooldown.read_text(encoding="utf-8"))
    assert "vm" in json.dumps(rejected)             # 30-min rejected-move cooldown persisted


def test_apply_mode_submits_a_fresh_snapshot(apply_run):
    # Positive control: the same run with a fresh refresh reaches Nova, so the
    # refusal above comes from the freshness gate and not from the harness.
    apply_run.refresh_age = 5.0
    result = apply_run.invoke("-y")
    assert apply_run.submitted == [("vm", "dst")]
    assert result.exit_code == 1                    # the fake Nova rejected it (409)


def test_dry_run_never_wires_the_journal(apply_run):
    apply_run.refresh_age = 121.0
    result = apply_run.invoke("--dry-run")
    assert result.exit_code == 0, result.output
    assert apply_run.submitted == [] and not apply_run.journal_dir.exists()


# --- Test gap 4: Placement allocation_ratio math ------------------------------------------

def test_nominal_capacity_applies_the_allocation_ratio():
    from compute_maintenance.domain.policies import build_placement_inventory_record
    gib = 1024
    memory = build_placement_inventory_record(
        {"total": 256 * gib, "reserved": 16 * gib, "allocation_ratio": 1.5}, used=180 * gib,
        headroom_percent=5)
    assert memory["nominal_capacity"] == 360 * gib            # (256 - 16) x 1.5
    assert memory["policy_capacity"] == pytest.approx(342 * gib)   # 360 x 0.95
    assert memory["nominal_free"] == 180 * gib
    assert memory["free"] == pytest.approx(162 * gib)
    assert memory["allocation_pct"] == pytest.approx(50.0)    # 180 / 360 (75.0 without the ratio)
    vcpu = build_placement_inventory_record(
        {"total": 64, "reserved": 4, "allocation_ratio": 4.0}, used=0, headroom_percent=0)
    assert vcpu["nominal_capacity"] == 240                    # (64 - 4) x 4.0


def test_a_vm_fits_only_thanks_to_the_ratio():
    from compute_maintenance.domain.policies import (
        build_placement_inventory_record,
        placement_can_fit,
    )
    gib = 1024
    placement = {
        "VCPU": build_placement_inventory_record({"total": 64, "reserved": 4, "allocation_ratio": 4.0}, 200, 0),
        "MEMORY_MB": build_placement_inventory_record(
            {"total": 256 * gib, "reserved": 16 * gib, "allocation_ratio": 1.5}, 300 * gib, 0),
    }
    # 240 - 200 = 40 VCPU and 360 - 300 = 60 GiB free; without the ratios both would be full.
    assert placement_can_fit(placement, 32, 48 * gib) == (True, None)
    assert placement_can_fit(placement, 41, 1 * gib) == (False, "placement_vcpu")
    assert placement_can_fit(placement, 1, 61 * gib) == (False, "placement_memory")
