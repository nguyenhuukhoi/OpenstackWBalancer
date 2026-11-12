"""Independent safety and race-condition tests for wlbalancer."""

import itertools
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compute_maintenance.domain.plan_context import PlanContext  # noqa: E402
from tests.legacy_api import balance as wlb  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_module_state():
    moved = dict(wlb.VM_LAST_MOVED_AT)
    failed = dict(wlb.VM_LAST_FAILED_MOVE_AT)
    cooldown_file = wlb.COOLDOWN_STATE_FILE
    events_file = wlb.MIGRATION_EVENTS_FILE
    wlb.VM_LAST_MOVED_AT.clear()
    wlb.VM_LAST_FAILED_MOVE_AT.clear()
    yield
    wlb.VM_LAST_MOVED_AT.clear()
    wlb.VM_LAST_MOVED_AT.update(moved)
    wlb.VM_LAST_FAILED_MOVE_AT.clear()
    wlb.VM_LAST_FAILED_MOVE_AT.update(failed)
    wlb.COOLDOWN_STATE_FILE = cooldown_file
    wlb.MIGRATION_EVENTS_FILE = events_file


def prometheus_payload(rows):
    return {"status": "success", "data": {"result": rows}}


def sample(labels, value):
    return {"metric": labels, "value": [1_700_000_000, str(value)]}


def inventory(total, used=0, headroom=0, max_unit=None):
    raw = {
        "total": total,
        "reserved": 0,
        "allocation_ratio": 1.0,
        "min_unit": 1,
        "max_unit": total if max_unit is None else max_unit,
        "step_size": 1,
    }
    return wlb.build_placement_inventory_record(raw, used, headroom)


def placement(vcpu_total=32, vcpu_used=0, ram_total_gb=128, ram_used_gb=0):
    return {
        "resource_provider_uuid": "rp",
        "VCPU": inventory(vcpu_total, vcpu_used),
        "MEMORY_MB": inventory(ram_total_gb * 1024, ram_used_gb * 1024),
    }


def vm_details(vcpus=4, cpu=50, ram_gb=8, ram_pct=50, status="ACTIVE"):
    return {
        "vcpus": vcpus,
        "cpu_pct": cpu,
        "ram_gb": ram_gb,
        "ram_pct": ram_pct,
        "ram_rss_gb": None,
        "status": status,
        "metadata_complete": True,
        "metrics_complete": True,
    }


def context(**changes):
    values = {
        "effective_hosts": ["h1", "h2"],
        "host_cpu_pct": {"h1": 60.0, "h2": 10.0},
        "host_ram_pct": {"h1": 40.0, "h2": 20.0},
        "host_cores": {"h1": 16.0, "h2": 16.0},
        "host_ram": {"h1": 128.0, "h2": 128.0},
        "host_free_cores": {"h1": 8.0, "h2": 24.0},
        "host_free_ram": {"h1": 64.0, "h2": 120.0},
        "host_placement": {"h1": placement(), "h2": placement()},
        "vm_host_map": {"vm1": "h1"},
        "all_vm_details": {"vm1": vm_details()},
        "vm_to_groups": {"vm1": []},
        "enforce_server_groups": False,
        "dest_ready_hosts": {"h1", "h2"},
        "host_running_vms": {"h1": 1, "h2": 0},
        "host_max_instances": {"h1": 0, "h2": 0},
        "excluded_vm_ids": set(),
        "reject_stats": {},
        "dest_reject_stats": {},
        "skip_stats": {},
    }
    values.update(changes)
    return wlb.BalanceContext(**values)


class TestConfiguration:
    @pytest.fixture(autouse=True)
    def configured_test_jobs(self, monkeypatch):
        for key, value in zip(self.JOB_KEYS[:3], ('node-test', 'libvirt-test', 'os-test')):
            monkeypatch.setenv(key, value)
        # rc5: no built-in Prometheus URL any more (it pointed at a public name).
        monkeypatch.setenv("PROMETHEUS_QUERY_URL", "http://prometheus.test:9090/api/v1/query")

    def test_missing_prometheus_url_is_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PROMETHEUS_QUERY_URL", "x")
        monkeypatch.delenv("PROMETHEUS_QUERY_URL")
        with pytest.raises(wlb.ConfigError, match="PROMETHEUS_QUERY_URL is required"):
            wlb.LoadbalancerConfig.load_config(str(tmp_path / "missing.conf"))

    @pytest.mark.parametrize("name", ["PLACEMENT_SYNC_TIMEOUT", "MAKE_ROOM_MAX_VM_CPU_PERCENT",
                                      "PLACEMENT_VCPU_HEADROOM_PERCENT"])
    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
    def test_non_finite_numbers_are_rejected(self, monkeypatch, tmp_path, name, raw):
        monkeypatch.setenv(name, raw)
        with pytest.raises(wlb.ConfigError, match="finite"):
            wlb.LoadbalancerConfig.load_config(str(tmp_path / "missing.conf"))

    @pytest.mark.parametrize("raw,expected", [("yes", True), ("ON", True), ("0", False), ("off", False)])
    def test_boolean_environment_is_parsed_strictly(self, monkeypatch, raw, expected):
        monkeypatch.setenv("WLB_TEST_BOOL", raw)
        assert wlb.env_bool("WLB_TEST_BOOL") is expected

    def test_boolean_environment_typo_is_rejected(self, monkeypatch):
        monkeypatch.setenv("WLB_TEST_BOOL", "flase")
        with pytest.raises(wlb.ConfigError):
            wlb.env_bool("WLB_TEST_BOOL")

    def test_max_instance_map_accepts_integer_strings(self):
        assert wlb.parse_max_instances_map('{"h1": "3", "h2": 0}') == {"h1": 3, "h2": 0}

    @pytest.mark.parametrize(
        "raw",
        ['{"h1": -1}', '{"h1": 1.5}', '{"h1": true}', '{"": 2}', '[1, 2]'],
    )
    def test_max_instance_map_rejects_unsafe_values(self, raw):
        with pytest.raises(wlb.ConfigError):
            wlb.parse_max_instances_map(raw)

    def test_make_room_limits_and_sync_interval_are_validated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAKE_ROOM_MAX_VM_CPU_PERCENT", "101")
        with pytest.raises(wlb.ConfigError):
            wlb.LoadbalancerConfig.load_config(str(tmp_path / "missing.conf"))

    def test_make_room_uses_higher_churn_aware_threshold_by_default(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("MAKE_ROOM_MIN_IMPROVEMENT", raising=False)
        cfg = wlb.LoadbalancerConfig.load_config(str(tmp_path / "missing.conf"))
        assert cfg.make_room_min_improvement == pytest.approx(0.3)

    @pytest.mark.parametrize("raw", ["nan", "inf", "-1", "0.099"])
    def test_make_room_threshold_rejects_unsafe_values(
        self, monkeypatch, tmp_path, raw
    ):
        monkeypatch.setenv("MAKE_ROOM_MIN_IMPROVEMENT", raw)
        with pytest.raises(wlb.ConfigError, match="MAKE_ROOM_MIN_IMPROVEMENT"):
            wlb.LoadbalancerConfig.load_config(str(tmp_path / "missing.conf"))

    JOB_KEYS = ("PROMETHEUS_NODE_JOB", "PROMETHEUS_LIBVIRT_JOB", "PROMETHEUS_CPU_USED",
                "PROMETHEUS_OPENSTACK_EXPORTER_JOB", "PROMETHEUS_CPU_RATIO", "PROMETHEUS_MEM_RATIO")

    def _clean_job_env(self, monkeypatch):
        # setenv+delenv: monkeypatch restores "unset" even for keys that
        # load_dotenv() writes into os.environ during the test.
        for key in self.JOB_KEYS:
            monkeypatch.setenv(key, "x")
            monkeypatch.delenv(key)

    def test_prometheus_jobs_come_from_config_file(self, monkeypatch, tmp_path):
        self._clean_job_env(monkeypatch)
        conf = tmp_path / "wbalancer.conf"
        conf.write_text("PROMETHEUS_NODE_JOB=node-x\nPROMETHEUS_LIBVIRT_JOB=libvirt-x\n")
        cfg = wlb.LoadbalancerConfig.load_config(str(conf))
        assert cfg.prometheus_node_job == "node-x"
        for query in (cfg.prometheus_query_cpu_used, cfg.prometheus_query_mem_used,
                      cfg.prometheus_query_logical_cpus, cfg.prometheus_query_total_ram_bytes):
            assert 'job="node-x"' in query and "Prod-Openstack" not in query
        assert 'job="libvirt-x"' in wlb.cfg_vm_ram_rss_query(cfg)
        mem, cpu, _ = wlb.build_vm_metric_queries(["h1"], libvirt_job=wlb.cfg_libvirt_job(cfg))
        assert 'job="libvirt-x"' in mem and 'job="libvirt-x"' in cpu

    def test_explicit_query_wins_over_job(self, monkeypatch, tmp_path):
        self._clean_job_env(monkeypatch)
        conf = tmp_path / "wbalancer.conf"
        conf.write_text("PROMETHEUS_NODE_JOB=node-x\nPROMETHEUS_LIBVIRT_JOB=libvirt-x\nPROMETHEUS_CPU_USED=my_cpu_query\n")
        cfg = wlb.LoadbalancerConfig.load_config(str(conf))
        assert cfg.prometheus_query_cpu_used == "my_cpu_query"
        assert 'job="node-x"' in cfg.prometheus_query_mem_used

    @pytest.mark.parametrize('missing', JOB_KEYS[:2])
    def test_jobs_required_when_not_configured(self, monkeypatch, tmp_path, missing):
        self._clean_job_env(monkeypatch)
        conf = tmp_path / 'missing.conf'
        conf.write_text(''.join(f'{key}=test-job\n' for key in self.JOB_KEYS[:2] if key != missing))
        with pytest.raises(wlb.ConfigError, match=missing + ' is required'):
            wlb.LoadbalancerConfig.load_config(str(conf))

    def test_obsolete_ratio_keys_are_ignored_with_warning(self, monkeypatch, tmp_path, caplog):
        # Old configs still carry the openstack-exporter job / ratio queries.
        # Placement supplies allocation ratios, so loading must not fail.
        self._clean_job_env(monkeypatch)
        conf = tmp_path / 'old.conf'
        conf.write_text("PROMETHEUS_NODE_JOB=node-x\nPROMETHEUS_LIBVIRT_JOB=libvirt-x\n"
                        "PROMETHEUS_OPENSTACK_EXPORTER_JOB=does-not-exist\nPROMETHEUS_CPU_RATIO=q\n")
        with caplog.at_level(logging.WARNING):
            cfg = wlb.LoadbalancerConfig.load_config(str(conf))
        assert not hasattr(cfg, "prometheus_openstack_exporter_job")
        assert not hasattr(cfg, "prometheus_query_cpu_ratio")
        assert "PROMETHEUS_OPENSTACK_EXPORTER_JOB" in caplog.text and "PROMETHEUS_CPU_RATIO" in caplog.text

    def test_runtime_capacity_queries_retain_last_good_sample_for_five_minutes(self):
        queries = wlb.build_default_queries("node-x", "libvirt-x")
        assert queries["logical_cpus"] == (
            'count by (alias) (last_over_time('
            'node_cpu_seconds_total{job="node-x",mode="idle"}[5m]))'
        )
        assert queries["total_ram_bytes"] == (
            'avg by (alias) (last_over_time('
            'node_memory_MemTotal_bytes{job="node-x"}[5m]))'
        )

    @pytest.mark.parametrize("bad", ['a"b', "a\\b", "a}b"])
    def test_job_name_cannot_break_promql(self, monkeypatch, tmp_path, bad):
        self._clean_job_env(monkeypatch)
        monkeypatch.setenv("PROMETHEUS_NODE_JOB", bad)
        with pytest.raises(wlb.ConfigError, match="PROMETHEUS_NODE_JOB"):
            wlb.LoadbalancerConfig.load_config(str(tmp_path / "missing.conf"))

    def test_plan_threshold_charges_make_room_for_both_migrations(self):
        cfg = SimpleNamespace(make_room_min_improvement=0.2)
        direct = SimpleNamespace(plan_type="direct", improvement=0.186)
        make_room = SimpleNamespace(plan_type="make_room", improvement=0.186)

        assert wlb.determine_plan_improvement_threshold(direct, cfg) == pytest.approx(0.15)
        threshold = wlb.determine_plan_improvement_threshold(make_room, cfg)
        assert threshold == pytest.approx(0.2)
        assert make_room.improvement < threshold

    def test_programmatic_config_cannot_weaken_make_room_below_direct(self):
        cfg = SimpleNamespace(make_room_min_improvement=0.01)
        plan = SimpleNamespace(plan_type="make_room")
        assert wlb.determine_plan_improvement_threshold(plan, cfg) == pytest.approx(0.15)


class TestPrometheusBoundaries:
    def test_candidate_route_uses_locale_safe_ascii_arrow(self):
        plan = SimpleNamespace(
            plan_type="direct",
            steps=[SimpleNamespace(vm_id="vm1", src_host="compute13", dst_host="compute18")],
        )
        line = wlb.format_candidate_line(
            PlanContext(
                best_plan_candidate=plan,
                all_vm_names={"vm1": "example-vm"},
                has_valid_move=True,
            )
        )

        assert line == "Proposed plan: DIRECT | example-vm (compute13 -> compute18)"
        assert "â" not in line

    def test_re2_literal_escape_does_not_escape_hyphen(self):
        escaped = wlb.escape_prometheus_regex_literal("node-1.example(blue)")
        assert escaped == r"node-1\.example\(blue\)"
        mem, cpu, _ = wlb.build_vm_metric_queries(["node-1.example(blue)"], libvirt_job='libvirt-test')
        assert r"^(node-1\.example\(blue\))$" in mem
        assert r"\-" not in cpu

    def test_host_metric_rejects_duplicate_alias(self, monkeypatch):
        data = prometheus_payload(
            [sample({"alias": "h1"}, 10), sample({"alias": "h1"}, 11)]
        )
        monkeypatch.setattr(wlb, "do_query", lambda *_: data)
        with pytest.raises(wlb.PrometheusError, match="duplicate"):
            wlb.get_metrics("url", "query")

    @pytest.mark.parametrize("raw", ["NaN", "Inf", "-Inf"])
    def test_host_metric_rejects_non_finite_values(self, monkeypatch, raw):
        monkeypatch.setattr(
            wlb,
            "do_query",
            lambda *_: prometheus_payload([sample({"alias": "h1"}, raw)]),
        )
        with pytest.raises(wlb.PrometheusError, match="not finite"):
            wlb.get_metrics("url", "query")

    def _vm_metrics(self, monkeypatch, mem_rows, cpu_rows, rss_rows=None):
        monkeypatch.setattr(
            wlb,
            "build_vm_metric_queries",
            lambda *_args, **_kwargs: ("mem", "cpu", "rss" if rss_rows is not None else None),
        )
        payloads = {
            "mem": prometheus_payload(mem_rows),
            "cpu": prometheus_payload(cpu_rows),
            "rss": prometheus_payload(rss_rows or []),
        }
        monkeypatch.setattr(wlb, "do_query", lambda _url, query: payloads[query])
        cfg = SimpleNamespace(prometheus_query_url="url", prometheus_query_vm_ram_rss="", prometheus_libvirt_job='libvirt-test')
        return wlb.get_all_vm_metrics(cfg, ["h1", "h2"])

    def test_complete_vm_snapshot_is_returned(self, monkeypatch):
        metrics = self._vm_metrics(
            monkeypatch,
            [sample({"instanceId": "vm1", "host": "h1"}, 40)],
            [sample({"instanceId": "vm1", "host": "h1"}, 25)],
        )
        assert metrics["vm1"]["host"] == "h1"
        assert metrics["vm1"]["metrics_complete"] is True

    def test_two_host_vm_snapshot_is_skipped_during_migration(self, monkeypatch):
        metrics = self._vm_metrics(
            monkeypatch,
            [
                sample({"instanceId": "vm1", "host": "h1"}, 40),
                sample({"instanceId": "vm1", "host": "h2"}, 41),
            ],
            [sample({"instanceId": "vm1", "host": "h2"}, 25)],
        )
        assert "vm1" not in metrics

    def test_incomplete_vm_snapshot_is_skipped(self, monkeypatch):
        metrics = self._vm_metrics(
            monkeypatch,
            [sample({"instanceId": "vm1", "host": "h1"}, 40)],
            [],
        )
        assert metrics == {}

    def test_small_irate_overshoot_is_clamped(self, monkeypatch):
        metrics = self._vm_metrics(
            monkeypatch,
            [sample({"instanceId": "vm1", "host": "h1"}, 20)],
            [sample({"instanceId": "vm1", "host": "h1"}, 100.09)],
        )
        assert metrics["vm1"]["cpu_pct"] == 100.0

    def _vm_metrics_rss_fallback(self, monkeypatch, mem_rows, cpu_rows, fallback_rows):
        monkeypatch.setattr(
            wlb, "build_vm_metric_queries", lambda *_a, **_k: ("mem", "cpu", None)
        )
        calls = []

        def fake_query(_url, query):
            calls.append(query)
            if query == "mem":
                return prometheus_payload(mem_rows)
            if query == "cpu":
                return prometheus_payload(cpu_rows)
            assert "libvirt_domain_stat_memory_rss_bytes" in query
            if fallback_rows is None:
                raise wlb.PrometheusError("boom")
            return prometheus_payload(fallback_rows)

        monkeypatch.setattr(wlb, "do_query", fake_query)
        cfg = SimpleNamespace(prometheus_query_url="url", prometheus_query_vm_ram_rss="", prometheus_libvirt_job='libvirt-test')
        return wlb.get_all_vm_metrics(cfg, ["h1", "h2"]), calls

    def test_vm_without_balloon_stats_uses_rss(self, monkeypatch):
        gib = 1024.0 ** 3
        metrics, calls = self._vm_metrics_rss_fallback(
            monkeypatch,
            [sample({"instanceId": "vm1", "host": "h1"}, 40)],
            [
                sample({"instanceId": "vm1", "host": "h1"}, 25),
                sample({"instanceId": "opn", "host": "h1"}, 5),
            ],
            [
                sample({"instanceId": "vm1", "host": "h1"}, 9 * gib),
                sample({"instanceId": "opn", "host": "h1"}, 3 * gib),
            ],
        )
        assert len(calls) == 3
        assert metrics["opn"]["ram_rss_gb"] == pytest.approx(3.0)
        assert metrics["opn"]["ram_pct"] is None
        assert metrics["opn"]["metrics_complete"] is True
        # a VM with balloon stats keeps its own RAM% and gets no RSS
        assert metrics["vm1"]["ram_pct"] == 40
        assert metrics["vm1"]["ram_rss_gb"] is None

    def test_no_fallback_query_when_all_vms_have_ram(self, monkeypatch):
        _, calls = self._vm_metrics_rss_fallback(
            monkeypatch,
            [sample({"instanceId": "vm1", "host": "h1"}, 40)],
            [sample({"instanceId": "vm1", "host": "h1"}, 25)],
            [],
        )
        assert calls == ["mem", "cpu"]

    def test_failed_fallback_keeps_vm_skipped(self, monkeypatch):
        metrics, _ = self._vm_metrics_rss_fallback(
            monkeypatch,
            [],
            [sample({"instanceId": "opn", "host": "h1"}, 5)],
            None,
        )
        assert metrics == {}

    def test_rss_fallback_ram_pct_from_flavor(self):
        conn = SimpleNamespace()
        metrics = {"opn": {"host": "h1", "cpu_pct": 5.0, "ram_pct": None,
                           "ram_rss_gb": 3.0, "metrics_complete": True}}
        orig = wlb.get_vm_metadata
        try:
            wlb.get_vm_metadata = lambda *_a, **_k: {
                "name": "opn", "vcpus": 2, "ram_gb": 4.0, "project_id": None,
                "project_name": None, "status": "ACTIVE", "metadata_complete": True}
            _, details = wlb.calculate_vm_scores(conn, "h1", vm_metrics=metrics)
        finally:
            wlb.get_vm_metadata = orig
        assert details["opn"]["ram_pct"] == pytest.approx(75.0)
        assert details["opn"]["ram_rss_gb"] == 3.0

    def test_busy_vm_above_its_vcpus_is_clamped_not_dropped(self, monkeypatch):
        # libvirt cpu_time includes emulator/vhost threads: a guest at 100%
        # reads 101-106% on the lab (wlb-cpu-test). It must stay plannable.
        metrics = self._vm_metrics(
            monkeypatch,
            [sample({"instanceId": "vm1", "host": "h1"}, 20)],
            [sample({"instanceId": "vm1", "host": "h1"}, 105)],
        )
        assert metrics["vm1"]["cpu_pct"] == 100.0

    def test_absurd_vm_cpu_percentage_is_skipped(self, monkeypatch):
        metrics = self._vm_metrics(
            monkeypatch,
            [sample({"instanceId": "vm1", "host": "h1"}, 20)],
            [sample({"instanceId": "vm1", "host": "h1"}, 1500)],
        )
        assert metrics == {}


class TestPersistentStateAndLock:
    def test_cooldown_filter_keeps_future_drops_expired_and_bad_values(self):
        # A future stamp means the clock stepped back: keep the VM cooling
        # down (from now) instead of letting it bounce straight back.
        now = 1_000.0
        assert wlb.load_active_cooldown_entries(
            {"fresh": 990, "expired": 800, "future": 1001, "bad": "x", "nan": "nan"},
            max_age=100,
            label="test",
            now=now,
        ) == {"fresh": 990.0, "future": 1000.0}

    def test_remaining_cooldown_uses_the_longer_failure_window(self):
        now = 10_000.0
        wlb.VM_LAST_MOVED_AT["vm1"] = now - wlb.VM_COOLDOWN_SECONDS + 1
        wlb.VM_LAST_FAILED_MOVE_AT["vm1"] = now - 10
        assert wlb.vm_cooldown_remaining("vm1", now) == pytest.approx(
            wlb.VM_FAILED_MOVE_COOLDOWN_SECONDS - 10
        )

    def test_runtime_cooldown_path_is_honored_and_atomic(self, tmp_path):
        path = tmp_path / "state.json"
        wlb.COOLDOWN_STATE_FILE = str(path)
        recorded_at = time.time()
        wlb.VM_LAST_MOVED_AT["vm1"] = recorded_at
        assert wlb.save_cooldown_state() is True
        assert json.loads(path.read_text())["moved"] == {"vm1": recorded_at}
        assert list(tmp_path.glob("*.tmp")) == []

    @pytest.mark.parametrize("content", ["not-json", "[]", '{"moved": []}'])
    def test_corrupt_cooldown_state_fails_closed(self, tmp_path, content):
        path = tmp_path / "state.json"
        path.write_text(content)
        with pytest.raises(wlb.ConfigError):
            wlb.load_cooldown_state(str(path))

    def test_lock_is_exclusive(self, tmp_path):
        if wlb.fcntl is None:
            pytest.skip("POSIX lock test")
        path = str(tmp_path / "apply.lock")
        ok1, handle1 = wlb.acquire_single_instance_lock(path)
        ok2, handle2 = wlb.acquire_single_instance_lock(path)
        try:
            assert ok1 is True and handle1 is not None
            assert (ok2, handle2) == (False, None)
            assert "pid=" in Path(path).read_text()
        finally:
            if handle1 is not None:
                handle1.close()

    def test_apply_lock_fails_closed_without_fcntl(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wlb, "fcntl", None)
        assert wlb.acquire_single_instance_lock(str(tmp_path / "x")) == (False, None)


class TestExclusionRules:
    def test_exact_uuid_exact_name_and_glob_are_supported(self):
        assert wlb.match_vm_pattern("id-1", "web-01", ["id-1"]) == "id-1"
        assert wlb.match_vm_pattern("id-1", "web-01", ["web-01"]) == "web-01"
        assert wlb.match_vm_pattern("id-1", "web-01", ["web-*"]) == "web-*"
        assert wlb.match_vm_pattern("id-1", "web-01", ["WEB-*"]) is None

    def test_exclusion_report_is_grouped_and_truncated(self):
        names = {f"id{i}": f"db-{i}" for i in range(7)}
        excluded = wlb.build_excluded_vm_map(names, ["db-*"])
        report = wlb.build_excluded_vm_report(excluded, names)
        lines = wlb.format_excluded_vm_lines(report, total_vms=7, limit=3)
        assert "7 of 7" in lines[0]
        assert "+4 more" in lines[1]

    def test_exclude_file_deduplicates_and_missing_file_is_fatal(self, tmp_path):
        path = tmp_path / "exclude.txt"
        path.write_text("# comment\nweb-*\n\nweb-*\nid-1\n")
        assert wlb.load_exclude_vm_patterns(str(path)) == ["web-*", "id-1"]
        with pytest.raises(wlb.ConfigError):
            wlb.load_exclude_vm_patterns(str(tmp_path / "missing"))


class TestOpenStackSafetyData:
    def test_server_group_fetch_failure_is_fatal(self):
        class Compute:
            def server_groups(self, **_kwargs):
                raise RuntimeError("keystone unavailable")

        with pytest.raises(wlb.OpenStackDataError, match="refusing"):
            wlb.fetch_server_groups(SimpleNamespace(compute=Compute()))

    def test_non_metric_server_remains_in_group_topology(self, monkeypatch):
        servers = {
            "vm1": SimpleNamespace(
                id="vm1", name="candidate", hypervisor_hostname="h1",
                status="ACTIVE", flavor={"vcpus": 2, "ram": 2048}, project_id=None,
            ),
            "vm2": SimpleNamespace(
                id="vm2", name="group-peer", hypervisor_hostname="h1",
                status="ACTIVE", flavor={"vcpus": 2, "ram": 2048}, project_id=None,
            ),
        }
        monkeypatch.setattr(wlb, "build_nova_host_to_hypervisor_map", lambda *_: {"h1": "h1"})
        conn = SimpleNamespace(compute=SimpleNamespace())
        metrics = {
            "vm1": {
                "host": "h1", "cpu_pct": 10, "ram_pct": 10,
                "ram_rss_gb": None, "metrics_complete": True,
            }
        }
        names, details, host_map = wlb.build_aggregate_vm_inventory(
            conn, ["h1"], servers, metrics, {}
        )
        assert names == {"vm1": "candidate", "vm2": "group-peer"}
        assert set(details) == {"vm1"}
        assert host_map == {"vm1": "h1", "vm2": "h1"}

    def test_missing_flavor_is_not_replaced_with_fake_small_capacity(self):
        server = SimpleNamespace(
            id="vm1", name="unknown-flavor", flavor={}, status="ACTIVE", project_id=None
        )
        conn = SimpleNamespace(compute=SimpleNamespace())
        metadata = wlb.get_vm_metadata(conn, "vm1", server_cache={"vm1": server})
        assert metadata["metadata_complete"] is False
        assert metadata["vcpus"] is None
        assert metadata["ram_gb"] is None

    def test_flavor_id_is_resolved_authoritatively(self):
        server = SimpleNamespace(
            id="vm1", name="vm", flavor={"id": "f1"}, status="ACTIVE", project_id=None
        )
        compute = SimpleNamespace(
            get_flavor=lambda _id: {"id": "f1", "vcpus": 8, "ram": 16384}
        )
        metadata = wlb.get_vm_metadata(
            SimpleNamespace(compute=compute), "vm1", server_cache={"vm1": server}
        )
        assert metadata["metadata_complete"] is True
        assert (metadata["vcpus"], metadata["ram_gb"]) == (8, 16.0)

    @pytest.mark.parametrize(
        "status,state,forced,expected",
        [
            ("enabled", "up", False, True),
            ("disabled", "up", False, False),
            ("enabled", "down", False, False),
            ("enabled", "up", True, False),
            ("enabled", "up", "False", True),
        ],
    )
    def test_compute_service_readiness_matrix(self, status, state, forced, expected):
        service = SimpleNamespace(
            host="h1", binary="nova-compute", status=status, state=state,
            is_forced_down=forced, disabled_reason="maintenance",
        )
        compute = SimpleNamespace(services=lambda **_kwargs: [service])
        ready, _reason = wlb.is_nova_compute_service_ready(
            SimpleNamespace(compute=compute), "h1"
        )
        assert ready is expected

    def test_late_sdk_attribute_wins_when_dict_view_is_empty(self):
        class SdkResource(dict):
            pass

        resource = SdkResource({"id": "rp"})
        object.__setattr__(resource, "usages", {"VCPU": 7})
        assert wlb.get_resource_field(resource, "usages") == {"VCPU": 7}

    @pytest.mark.parametrize(
        "policy,hosts,destination,expected",
        [
            ("affinity", {"vm1": "h1", "vm2": "h1"}, "h2", False),
            ("affinity", {"vm1": "h1"}, "h2", True),
            ("anti-affinity", {"vm1": "h1", "vm2": "h2"}, "h2", False),
            ("anti-affinity", {"vm1": "h1", "vm2": "h2"}, "h3", True),
        ],
    )
    def test_server_group_policy_matrix(self, policy, hosts, destination, expected):
        groups = {
            "vm1": [{"policy": policy, "members": list(hosts), "id": "g", "name": "g"}]
        }
        assert wlb.is_server_group_move_compliant(
            "vm1", "h1", destination, hosts, groups
        ) is expected


class TestCapacityAndDecisionLogic:
    def test_adaptive_weight_floor_remains_a_real_floor(self):
        assert wlb.calculate_adaptive_weights(90, 10) == pytest.approx((0.8, 0.2))
        assert wlb.calculate_adaptive_weights(10, 90) == pytest.approx((0.2, 0.8))

    def test_invalid_adaptive_weight_floors_are_rejected(self):
        with pytest.raises(ValueError):
            wlb.calculate_adaptive_weights(1, 1, 0.6, 0.5)

    def test_mad_and_pressure_signal_math(self):
        assert wlb.mean_absolute_deviation([10, 20, 30]) == pytest.approx(20 / 3)
        assert wlb.mean_absolute_deviation([]) == 0
        assert wlb.calculate_pressure_signals(10, 5, 90, 50) == pytest.approx((34, 18.5))

    @pytest.mark.parametrize(
        "cpu,ram,expected",
        [
            ({}, {}, wlb.MODE_NORMAL),
            ({"h": 70}, {"h": 80}, wlb.MODE_NORMAL),
            ({"h": 70.1}, {"h": 1}, wlb.MODE_CPU_HOTSPOT),
            ({"h": 1}, {"h": 80.1}, wlb.MODE_RAM_HOTSPOT),
            ({"h": 99}, {"h": 95.1}, wlb.MODE_RAM_CRITICAL_HOTSPOT),
        ],
    )
    def test_hotspot_mode_boundaries(self, cpu, ram, expected):
        assert wlb.determine_hotspot_mode(cpu, ram)[0] == expected

    @pytest.mark.parametrize(
        "mode,cpu_mad,ram_mad,valid,expected",
        [
            (wlb.MODE_RAM_HOTSPOT, 0, 0, True, "RAM_HOTSPOT_OPTIMIZABLE"),
            (wlb.MODE_RAM_HOTSPOT, 0, 0, False, "RAM_HOTSPOT_NO_VALID_MOVE"),
            (wlb.MODE_CPU_HOTSPOT, 0, 0, True, "CPU_HOTSPOT_OPTIMIZABLE"),
            (wlb.MODE_CPU_HOTSPOT, 0, 0, False, "CPU_HOTSPOT_NO_VALID_MOVE"),
            (wlb.MODE_PRESSURE, 0, 0, True, "RAM_PRESSURE"),
            (wlb.MODE_PRESSURE, 0, 0, False, "PRESSURE_NO_VALID_MOVE"),
            (wlb.MODE_MAD, 4, 0, True, "UNBALANCED"),
            (wlb.MODE_MAD, 4, 0, False, "UNBALANCED_NO_VALID_MOVE"),
            (wlb.MODE_MAD, 1, 1, True, "BALANCED_OPTIMIZABLE"),
            (wlb.MODE_MAD, 1, 1, False, "BALANCED"),
        ],
    )
    def test_cluster_classification_matrix(self, mode, cpu_mad, ram_mad, valid, expected):
        assert wlb.classify_cluster_detail_status(
            mode, 50, 90, cpu_mad, ram_mad, valid
        ) == expected

    @pytest.mark.parametrize(
        "status,reason,severity,exit_code",
        [
            ("BALANCED", None, "OK", 0),
            ("UNBALANCED", None, "WARNING", 1),
            ("RAM_HOTSPOT_NO_VALID_MOVE", None, "CRITICAL", 2),
            ("UNBALANCED_NO_VALID_MOVE", "no_technical_candidate", "CRITICAL", 2),
            ("UNBALANCED_NO_VALID_MOVE", "below_threshold", "WARNING", 1),
            ("INVALID", None, "UNKNOWN", 3),
        ],
    )
    def test_severity_and_exit_code_matrix(self, status, reason, severity, exit_code):
        actual = wlb.map_detail_status_to_severity(status, reason)
        assert actual == severity
        assert wlb.map_severity_to_exit_code(actual) == exit_code

    @pytest.mark.parametrize(
        "raw,used",
        [
            ({"total": -1, "allocation_ratio": 1}, 0),
            ({"total": 10, "allocation_ratio": 0}, 0),
            ({"total": 10, "allocation_ratio": 1, "step_size": 0}, 0),
            ({"total": 10, "allocation_ratio": 1}, math.nan),
        ],
    )
    def test_invalid_placement_inventory_fails_closed(self, raw, used):
        with pytest.raises(wlb.PlacementError):
            wlb.build_placement_inventory_record(raw, used, 0)

    def test_placement_headroom_and_usage_delta_are_enforced(self):
        p = {
            "VCPU": inventory(100, 75, headroom=20),
            "MEMORY_MB": inventory(100 * 1024, 50 * 1024, headroom=10),
        }
        assert wlb.placement_can_fit(p, 5, 1024) == (True, None)
        assert wlb.placement_can_fit(p, 5, 1024, {"VCPU": 1}) == (
            False,
            "placement_vcpu",
        )

    @pytest.mark.parametrize(
        "vcpu_pct,mem_pct,vcpu_free,mem_free,expected",
        [
            (10, 10, 1, 1, "CAPACITY_OK"),
            (95, 10, 0, 1, "VCPU_FULL"),
            (10, 95, 1, 0, "RAM_ALLOCATION_FULL"),
            (95, 95, 0, 0, "ALLOCATION_FULL"),
            (91, 10, 1, 1, "VCPU_PRESSURE"),
            (10, 91, 1, 1, "RAM_ALLOCATION_PRESSURE"),
            (91, 91, 1, 1, "ALLOCATION_PRESSURE"),
        ],
    )
    def test_placement_status_matrix(self, vcpu_pct, mem_pct, vcpu_free, mem_free, expected):
        p = {
            "VCPU": {"allocation_pct": vcpu_pct, "free": vcpu_free},
            "MEMORY_MB": {"allocation_pct": mem_pct, "free": mem_free},
        }
        assert wlb.placement_capacity_status(p) == expected

    def test_missing_runtime_denominator_is_fatal(self, monkeypatch):
        cfg = SimpleNamespace(
            prometheus_query_url="url",
            prometheus_query_logical_cpus="cpu",
            prometheus_query_total_ram_bytes="ram",
        )
        monkeypatch.setattr(
            wlb,
            "get_metrics",
            lambda _url, query, *_hosts: ({"h1": {"value": 16}} if query == "cpu" else {}),
        )
        with pytest.raises(wlb.PrometheusError):
            wlb.get_runtime_capacity_map(cfg, ["h1"])
        # Collecting mode: only the broken host is reported, the others load.
        monkeypatch.setattr(
            wlb,
            "get_metrics",
            lambda _url, query, *_hosts: {"h1": {"value": 16}, "h2": {"value": 8}}
            if query == "cpu" else {"h2": {"value": 8 * 1024 ** 3}},
        )
        missing = {}
        result = wlb.get_runtime_capacity_map(cfg, ["h1", "h2"], missing=missing)
        assert list(result) == ["h2"] and list(missing) == ["h1"]

    def test_exact_threshold_then_crossing_is_a_new_hotspot(self):
        assert wlb.would_create_new_hotspot(
            {"h1": wlb.CPU_HOTSPOT_THRESHOLD},
            {"h1": 10},
            {"h1": wlb.CPU_HOTSPOT_THRESHOLD + 0.1},
            {"h1": 10},
            "h1",
        ) is True

    def test_direct_search_rejects_a_new_destination_hotspot(self):
        ctx = context(
            host_cpu_pct={"h1": 60.0, "h2": 64.0},
            all_vm_details={"vm1": vm_details(vcpus=4, cpu=100)},
        )
        move, _ = wlb.find_best_move(ctx, wlb.MODE_MAD, ["h1"], 20, 1, 0)
        assert move is None
        assert ctx.reject_stats["would_create_new_hotspot"] == 1

    def test_non_improving_safe_pair_is_still_a_technical_candidate(self):
        ctx = context(
            host_cpu_pct={"h1": 10.0, "h2": 10.0},
            host_ram_pct={"h1": 10.0, "h2": 10.0},
            all_vm_details={"vm1": vm_details(cpu=0, ram_pct=0)},
        )
        move, improvement = wlb.find_best_move(ctx, wlb.MODE_MAD, ["h1"], 0, 0.5, 0.5)
        assert move is None and improvement == 0
        assert ctx.technical_candidate_count == 1

    def test_make_room_planner_builds_two_safe_steps(self):
        ctx = context(
            effective_hosts=["a", "b", "c"],
            host_cpu_pct={"a": 80.0, "b": 10.0, "c": 10.0},
            host_ram_pct={"a": 40.0, "b": 10.0, "c": 10.0},
            host_cores={"a": 10.0, "b": 10.0, "c": 10.0},
            host_ram={"a": 100.0, "b": 100.0, "c": 100.0},
            host_free_cores={"a": 20, "b": 1, "c": 20},
            host_free_ram={"a": 100, "b": 5, "c": 100},
            host_placement={
                "a": placement(20, 4, 100, 20),
                "b": placement(20, 17, 100, 95),
                "c": placement(20, 0, 100, 0),
            },
            vm_host_map={"primary": "a", "room": "b"},
            all_vm_details={
                "primary": vm_details(vcpus=4, cpu=50, ram_gb=8, ram_pct=50),
                "room": vm_details(vcpus=4, cpu=0, ram_gb=20, ram_pct=10),
            },
            vm_to_groups={"primary": [], "room": []},
            dest_ready_hosts={"a", "b", "c"},
            host_running_vms={"a": 1, "b": 1, "c": 0},
            host_max_instances={"a": 0, "b": 0, "c": 0},
        )
        cfg = SimpleNamespace(
            enable_make_room=True,
            make_room_max_vm_cpu_percent=10,
            make_room_max_vm_ram_percent=50,
        )
        plan = wlb.find_best_make_room_plan(
            ctx, cfg, wlb.MODE_CPU_HOTSPOT, ["a"], 20, 1, 0, "test"
        )
        assert plan is not None
        assert plan.steps[0].vm_id == "room"
        assert plan.steps[0].src_host == "b"
        assert plan.steps[0].dst_host in {"a", "c"}
        assert (plan.steps[1].vm_id, plan.steps[1].src_host, plan.steps[1].dst_host) == (
            "primary", "a", "b"
        )


class TestMakeRoomPlannerSafetyMatrix:
    @staticmethod
    def _cfg(**changes):
        values = {
            "enable_make_room": True,
            "make_room_max_vm_cpu_percent": 10,
            "make_room_max_vm_ram_percent": 50,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    @staticmethod
    def _ctx(**changes):
        values = {
            "effective_hosts": ["a", "b", "c"],
            "host_cpu_pct": {"a": 80.0, "b": 10.0, "c": 10.0},
            "host_ram_pct": {"a": 40.0, "b": 10.0, "c": 10.0},
            "host_cores": {"a": 10.0, "b": 10.0, "c": 10.0},
            "host_ram": {"a": 100.0, "b": 100.0, "c": 100.0},
            "host_free_cores": {"a": 2, "b": 1, "c": 20},
            "host_free_ram": {"a": 10, "b": 5, "c": 100},
            "host_placement": {
                "a": placement(20, 18, 100, 90),
                "b": placement(20, 17, 100, 95),
                "c": placement(20, 0, 100, 0),
            },
            "vm_host_map": {"primary": "a", "room": "b"},
            "all_vm_details": {
                "primary": vm_details(vcpus=4, cpu=50, ram_gb=8, ram_pct=50),
                "room": vm_details(vcpus=4, cpu=0, ram_gb=20, ram_pct=10),
            },
            "vm_to_groups": {"primary": [], "room": []},
            "dest_ready_hosts": {"a", "b", "c"},
            "host_running_vms": {"a": 1, "b": 1, "c": 0},
            "host_max_instances": {"a": 0, "b": 0, "c": 0},
        }
        values.update(changes)
        return context(**values)

    def _plan(self, ctx=None, cfg=None):
        return wlb.find_best_make_room_plan(
            ctx or self._ctx(),
            cfg or self._cfg(),
            wlb.MODE_CPU_HOTSPOT,
            ["a"],
            30,
            1,
            0,
            "safety-matrix",
        )

    def test_disabled_feature_never_builds_a_chain(self):
        assert self._plan(cfg=self._cfg(enable_make_room=False)) is None

    def test_make_room_is_not_used_when_primary_already_fits(self):
        p = {
            "a": placement(20, 18, 100, 90),
            "b": placement(20, 0, 100, 0),
            "c": placement(20, 0, 100, 0),
        }
        assert self._plan(self._ctx(host_placement=p)) is None

    @pytest.mark.parametrize(
        "detail_change",
        [
            {"cpu_pct": 10.01},
            {"ram_pct": 50.01},
        ],
    )
    def test_room_vm_above_impact_limit_is_rejected(self, detail_change):
        room = vm_details(vcpus=4, cpu=0, ram_gb=20, ram_pct=10)
        room.update(detail_change)
        details = {
            "primary": vm_details(vcpus=4, cpu=50, ram_gb=8, ram_pct=50),
            "room": room,
        }
        assert self._plan(self._ctx(all_vm_details=details)) is None

    def test_release_that_is_too_small_for_primary_is_rejected(self):
        details = {
            "primary": vm_details(vcpus=4, cpu=50, ram_gb=8, ram_pct=50),
            "room": vm_details(vcpus=2, cpu=0, ram_gb=4, ram_pct=10),
        }
        p = {
            "a": placement(20, 18, 100, 90),
            "b": placement(20, 19, 100, 98),
            "c": placement(20, 0, 100, 0),
        }
        assert self._plan(self._ctx(all_vm_details=details, host_placement=p)) is None

    def test_buffer_host_at_instance_limit_is_rejected(self):
        ctx = self._ctx(
            host_running_vms={"a": 1, "b": 1, "c": 1},
            host_max_instances={"a": 0, "b": 0, "c": 1},
        )
        assert self._plan(ctx) is None

    def test_final_target_instance_limit_is_simulated_after_release(self):
        ctx = self._ctx(
            host_running_vms={"a": 1, "b": 2, "c": 0},
            host_max_instances={"a": 0, "b": 1, "c": 0},
        )
        assert self._plan(ctx) is None

    def test_step_one_crossing_buffer_hotspot_is_rejected(self):
        ctx = self._ctx(host_cpu_pct={"a": 80.0, "b": 10.0, "c": 69.0})
        details = dict(ctx.all_vm_details)
        details["room"] = vm_details(vcpus=4, cpu=10, ram_gb=20, ram_pct=10)
        ctx.all_vm_details = details
        assert self._plan(ctx) is None

    def test_step_two_crossing_target_hotspot_is_rejected(self):
        ctx = self._ctx(host_cpu_pct={"a": 80.0, "b": 60.0, "c": 10.0})
        assert self._plan(ctx) is None

    def test_planning_does_not_mutate_placement_snapshot(self):
        ctx = self._ctx()
        before = json.loads(json.dumps(ctx.host_placement))
        assert self._plan(ctx) is not None
        assert ctx.host_placement == before


class TestMigrationTrackingRaces:
    @pytest.mark.parametrize(
        "status,expected",
        [
            ("accepted", "IN PROGRESS"),
            ("running (post-copy)", "IN PROGRESS"),
            ("completed", "SUCCESS"),
            ("cancelled", "FAILED"),
            ("failed", "FAILED"),
            ("vendor-state", "UNKNOWN (vendor-state)"),
        ],
    )
    def test_migration_status_mapping(self, status, expected):
        assert wlb.map_migration_state(status) == expected

    def test_server_result_requires_the_expected_destination(self):
        assert wlb.get_server_migration_result(
            "vm", "wrong", "ACTIVE", src_host="src", dst_host="dst", verbose=False
        ) is None
        assert wlb.get_server_migration_result(
            "vm", "dst", "ACTIVE", src_host="src", dst_host="dst", verbose=False
        ) == wlb.MIGRATION_RESULT_SUCCESS
        assert wlb.get_server_migration_result(
            "vm", "src", "ERROR", src_host="src", dst_host="dst", verbose=False
        ) == wlb.MIGRATION_RESULT_FAILED

    def test_tracked_lookup_never_falls_back_to_old_row(self):
        old = {"id": "old", "status": "completed"}
        assert wlb.find_tracked_migration([old], None) is None
        assert wlb.find_tracked_migration([old], "new") is None
        assert wlb.find_tracked_migration([old], "old") is old

    def test_monitor_ignores_old_success_and_waits_for_destination(self, monkeypatch):
        old = {"id": "old", "status": "completed"}
        new_running = {"id": "new", "status": "running"}
        new_done = {"id": "new", "status": "completed"}
        migration_snapshots = [
            [old],
            [old],
            [old, new_running],
            [old, new_done],
            [old, new_done],
        ]
        server_snapshots = [
            {"status": "ACTIVE", "hypervisor_hostname": "src"},
            {"status": "ACTIVE", "hypervisor_hostname": "src"},
            {"status": "MIGRATING", "hypervisor_hostname": "src"},
            {"status": "ACTIVE", "hypervisor_hostname": "src"},
            {"status": "ACTIVE", "hypervisor_hostname": "dst"},
        ]

        class Compute:
            calls = 0

            def server_migrations(self, _server):
                return migration_snapshots.pop(0)

            def get_server(self, _server):
                self.calls += 1
                return server_snapshots.pop(0)

            def migrations(self):
                return []

        compute = Compute()
        monkeypatch.setattr(wlb.time, "sleep", lambda _seconds: None)
        result = wlb.monitor_migration(
            SimpleNamespace(compute=compute),
            "vm1",
            "vm1",
            src_host="src",
            dst_host="dst",
            timeout=10,
            verbose=False,
            preexisting_migration_ids={"old"},
        )
        assert result == wlb.MIGRATION_RESULT_SUCCESS
        assert compute.calls == 5

    def test_completed_record_on_unexpected_host_is_not_success(self, monkeypatch):
        class Compute:
            def server_migrations(self, _server):
                return [{"id": "new", "status": "completed"}]

            def get_server(self, _server):
                return {"status": "ACTIVE", "hypervisor_hostname": "other"}

            def migrations(self, **_query):
                return []

        clock = {"t": 0.0}

        def monotonic():
            clock["t"] += 1.0
            return clock["t"]
        monkeypatch.setattr(wlb.time, "monotonic", monotonic)
        monkeypatch.setattr(wlb.time, "sleep", lambda _seconds: None)
        # rc5: the timeout is progress-aware; an unchanging record is a stall.
        result = wlb.monitor_migration(
            SimpleNamespace(compute=Compute()),
            "vm1",
            "vm1",
            src_host="src",
            dst_host="dst",
            timeout=1,
            stall_timeout=0,
            verbose=False,
            preexisting_migration_ids=set(),
        )
        assert result == wlb.MIGRATION_RESULT_TIMEOUT

    def test_failed_record_is_terminal(self):
        class Compute:
            def server_migrations(self, _server):
                return [{
                    "id": 17686,
                    "uuid": "migration-uuid",
                    "status": "failed",
                    "source_compute": "src",
                    "dest_compute": "dst",
                }]

            def get_server(self, _server):
                return {
                    "status": "MIGRATING",
                    "hypervisor_hostname": "src",
                    "task_state": "migrating",
                    "power_state": 1,
                }

            def migrations(self):
                return []

        result = wlb.monitor_migration(
            SimpleNamespace(compute=Compute()), "vm1", "vm1",
            src_host="src", dst_host="dst", timeout=1, verbose=False,
        )
        assert result == wlb.MIGRATION_RESULT_FAILED
    def test_failed_result_includes_nova_instance_action_root_cause(self):
        old = {"request_id": "req-old", "action": "reboot", "start_time": "01"}
        new = {
            "request_id": "req-new",
            "action": "os-migrateLive",
            "start_time": "02",
        }
        detailed = {
            **new,
            "message": "No valid host was found",
            "events": [
                {
                    "event": "compute_check_can_live_migrate_destination",
                    "result": "Error",
                    "host": "compute07",
                    "details": "NoValidHost: destination pre-check failed",
                    "traceback": "Traceback ... nova.exception.NoValidHost",
                }
            ],
        }

        class Compute:
            def server_actions(self, _server):
                return [old, new]

            def get_server_action(self, request_id, server_id, ignore_missing=True):
                assert (request_id, server_id, ignore_missing) == ("req-new", "vm1", True)
                return detailed

        conn = SimpleNamespace(compute=Compute())
        result = wlb.build_migration_monitor_result(
            wlb.MIGRATION_RESULT_FAILED,
            conn=conn,
            server_id="vm1",
            migration={"id": 17686, "uuid": "mig-uuid", "status": "failed"},
            server={"status": "ACTIVE", "hypervisor_hostname": "compute01"},
            src_host="compute01",
            dst_host="compute07",
            reason="migration_record_entered_failed_state",
            preexisting_action_request_ids={"req-old"},
        )
        detail = wlb.format_migration_monitor_detail(result)

        assert "action_request_id=req-new" in detail
        assert "NoValidHost: destination pre-check failed" in detail
        assert "compute_check_can_live_migrate_destination" in detail

    def test_action_diagnostics_never_reuse_old_action_without_snapshot(self):
        compute = SimpleNamespace(server_actions=lambda _server: [
            {"request_id": "req-old", "action": "os-migrateLive"}
        ])
        diagnostics = wlb.collect_new_server_action_diagnostics(
            SimpleNamespace(compute=compute), "vm1", None
        )
        assert diagnostics == {}

    def test_definitive_submit_rejection_reports_actual_nova_error(self):
        result = wlb.MigrationMonitorResult(
            wlb.MIGRATION_RESULT_FAILED,
            {"reason": "nova_rejected_migration_submission"},
        )
        detail = wlb.format_migration_monitor_detail(
            result,
            submit_error="NoValidHost: compute07 rejected the request",
            monitor_called=False,
        )

        assert detail.startswith("migration submission rejected")
        assert "NoValidHost" in detail
        assert "monitor_result" not in detail

    def test_queued_record_disables_false_active_on_source_timeout(self, monkeypatch):
        migrations = [
            [{"id": "new", "status": "queued"}],
            [{"id": "new", "status": "preparing"}],
            [{"id": "new", "status": "completed"}],
        ]
        servers = [
            {"status": "ACTIVE", "hypervisor_hostname": "src"},
            {"status": "ACTIVE", "hypervisor_hostname": "src"},
            {"status": "ACTIVE", "hypervisor_hostname": "dst"},
        ]

        class Compute:
            def server_migrations(self, _server):
                return migrations.pop(0)

            def get_server(self, _server):
                return servers.pop(0)

            def migrations(self):
                return []

        ticks = itertools.count(0, 200)
        monkeypatch.setattr(wlb.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(wlb.time, "sleep", lambda _seconds: None)
        result = wlb.monitor_migration(
            SimpleNamespace(compute=Compute()), "vm1", "vm1",
            src_host="src", dst_host="dst", timeout=1000, verbose=False,
        )
        assert result == wlb.MIGRATION_RESULT_SUCCESS

    @pytest.mark.parametrize(
        "host,task_state,power_state,expected_reset",
        [
            ("src", None, 1, True),
            ("other", None, 1, False),
            ("src", "migrating", 1, False),
            ("src", None, 4, False),
        ],
    )
    def test_error_reset_requires_source_idle_and_running(
        self, host, task_state, power_state, expected_reset
    ):
        server = {
            "status": "ERROR", "hypervisor_hostname": host,
            "task_state": task_state, "power_state": power_state,
        }

        class Compute:
            reset_called = False

            def get_server(self, _server):
                return server

            def reset_server_state(self, _server, state):
                assert state == "active"
                self.reset_called = True

        compute = Compute()
        result = wlb.reset_error_server_to_active(
            SimpleNamespace(compute=compute), "vm1", "vm1", expected_source_host="src"
        )
        assert result is expected_reset
        assert compute.reset_called is expected_reset

    def _step_and_plan(self):
        step = wlb.MigrationStep(
            vm_id="vm1", src_host="src", dst_host="dst", reason="direct",
            vcpus=2, vm_ram_gb=2, real_vm_cores=1, real_vm_ram=1,
            cpu_impact_src=1, cpu_impact_dst=1, ram_impact_src=1,
            ram_impact_dst=1, dst_free_vcpu=10, dst_free_ram_gb=10,
        )
        plan = wlb.MigrationPlan(
            plan_type="direct", steps=[step], baseline_weighted_mad=2,
            final_weighted_mad=1, improvement=1, mad_cpu=1, mad_ram=1,
            final_host_cpu={}, final_host_ram={}, search_scope="test",
        )
        return step, plan

    def _state(self):
        return PlanContext(
            aggregate="agg", all_vm_names={"vm1": "vm1"},
            all_vm_details={"vm1": {"project_id": None, "project_name": None}},
            detail_status="UNBALANCED", severity="WARNING",
            best_improvement=1, improvement_threshold=0.1,
        )

    def test_uncertain_submit_error_is_monitored_and_can_succeed(self, monkeypatch):
        step, plan = self._step_and_plan()

        class Compute:
            def live_migrate_server(self, **_kwargs):
                raise RuntimeError("socket closed after write")

        monkeypatch.setattr(wlb, "validate_migration_step", lambda *_a, **_k: (True, "ok"))
        monkeypatch.setattr(wlb, "get_server_migration_ids", lambda *_a: {"old"})
        monkeypatch.setattr(wlb, "monitor_migration", lambda *_a, **_k: wlb.MIGRATION_RESULT_SUCCESS)
        from compute_maintenance.adapters.cooldown import default_store
        monkeypatch.setattr(default_store, "save", lambda *_a, **_k: True)
        monkeypatch.setattr(wlb, "append_migration_event", lambda *_a, **_k: None)
        ok = wlb.execute_migration_step(
            SimpleNamespace(), SimpleNamespace(compute=Compute()), self._state(),
            step, plan, 1, enforce_server_groups=False,
        )
        assert ok is True

    def test_definitive_4xx_rejection_is_not_monitored(self, monkeypatch):
        step, plan = self._step_and_plan()
        events = []

        class Rejected(Exception):
            status_code = 409

        class Compute:
            def live_migrate_server(self, **_kwargs):
                raise Rejected("conflict")

        monkeypatch.setattr(wlb, "validate_migration_step", lambda *_a, **_k: (True, "ok"))
        monkeypatch.setattr(wlb, "get_server_migration_ids", lambda *_a: set())
        monkeypatch.setattr(
            wlb, "monitor_migration",
            lambda *_a, **_k: pytest.fail("definitive rejection must not be monitored"),
        )
        monkeypatch.setattr(wlb, "reset_error_server_to_active", lambda *_a, **_k: False)
        from compute_maintenance.adapters.cooldown import default_store
        monkeypatch.setattr(default_store, "save", lambda *_a, **_k: True)
        monkeypatch.setattr(wlb, "append_migration_event", events.append)
        ok = wlb.execute_migration_step(
            SimpleNamespace(), SimpleNamespace(compute=Compute()), self._state(),
            step, plan, 1, enforce_server_groups=False,
        )
        assert ok is False
        assert events[-1]["result"] == "migration_failed"
        assert "migration submission rejected" in events[-1]["detail"]
        assert "submit_error=conflict" in events[-1]["detail"]
        assert "monitor_result" not in events[-1]["detail"]


class TestPlacementConvergenceRace:
    def test_make_room_step_two_uses_direct_threshold_after_chain_is_accepted(
        self, monkeypatch
    ):
        cfg = SimpleNamespace(placement_final_allocation_candidate_check=False)
        conn = SimpleNamespace(
            compute=SimpleNamespace(
                get_server=lambda _id: {"status": "ACTIVE", "host": "src"}
            )
        )
        step = SimpleNamespace(
            vm_id="vm1", src_host="src", dst_host="dst", vcpus=1, vm_ram_gb=1
        )
        state = PlanContext(
            all_vm_details={"vm1": vm_details(vcpus=1, cpu=1, ram_gb=1, ram_pct=1)},
            host_max_instances={},
            host_running_vms={"dst": 0},
            host_cpu_pct={"src": 2.0, "dst": 0.0},
            host_ram_pct={"src": 0.0, "dst": 0.0},
            host_cores={"src": 1.0, "dst": 1.0},
            host_ram={"src": 1.0, "dst": 1.0},
            mode=wlb.MODE_PRESSURE,
            max_cpu=2.0,
            max_ram=0.0,
            cpu_weight=1.0,
            ram_weight=0.0,
            baseline_weighted_mad=2.0,
            improvement_threshold=0.2,
            step_improvement_threshold=0.1,
        )

        monkeypatch.setattr(
            wlb,
            "build_nova_host_to_hypervisor_map",
            lambda _conn, hosts: {host: host for host in hosts},
        )
        monkeypatch.setattr(
            wlb, "is_nova_compute_service_ready", lambda *_args: (True, "ok")
        )
        monkeypatch.setattr(wlb, "fetch_placement_host_resource", lambda *_a, **_k: {})
        monkeypatch.setattr(wlb, "placement_can_fit", lambda *_args: (True, "ok"))
        monkeypatch.setattr(
            wlb,
            "project_runtime_move",
            lambda *_args: {
                "new_host_cpu": {"src": 1.85, "dst": -1.85},
                "new_host_ram": {"src": 0.0, "dst": 0.0},
            },
        )
        monkeypatch.setattr(wlb, "would_create_new_hotspot", lambda *_args: False)

        ok, why = wlb.validate_migration_step(
            cfg,
            conn,
            state,
            step,
            enforce_server_groups=False,
            require_runtime_improvement=True,
        )
        assert ok is True and why == "ok"

    def test_make_room_step_retries_only_stale_placement(self, monkeypatch):
        cfg = SimpleNamespace(placement_sync_timeout=30, placement_sync_interval=2)
        first, second = PlanContext(aggregate="attempt-1"), PlanContext(aggregate="attempt-2")
        states = [first, second]
        outcomes = iter([(False, "placement_memory"), (True, "ok")])
        sleeps = []
        monkeypatch.setattr(wlb, "build_cluster_cache", lambda *_a, **_k: {})
        monkeypatch.setattr(wlb, "evaluate_aggregate_state", lambda **_k: states.pop(0))
        monkeypatch.setattr(wlb, "validate_migration_step", lambda *_a, **_k: next(outcomes))
        monkeypatch.setattr(wlb.time, "sleep", sleeps.append)
        refreshed, ok, why = wlb.wait_for_make_room_step(
            cfg, object(), "agg", ["h1", "h2"], object(),
            enforce_server_groups=False, exclude_unready_hosts=True,
        )
        assert ok is True and why == "ok"
        assert refreshed is second
        assert sleeps == [2]

    def test_make_room_step_does_not_retry_topology_change(self, monkeypatch):
        cfg = SimpleNamespace(placement_sync_timeout=30, placement_sync_interval=2)
        monkeypatch.setattr(wlb, "build_cluster_cache", lambda *_a, **_k: {})
        monkeypatch.setattr(wlb, "evaluate_aggregate_state", lambda **_k: PlanContext(aggregate="attempt-1"))
        monkeypatch.setattr(
            wlb, "validate_migration_step", lambda *_a, **_k: (False, "server_group")
        )
        monkeypatch.setattr(
            wlb.time, "sleep", lambda _s: pytest.fail("non-capacity error must not retry")
        )
        _, ok, why = wlb.wait_for_make_room_step(
            cfg, object(), "agg", ["h1", "h2"], object(),
            enforce_server_groups=False, exclude_unready_hosts=True,
        )
        assert ok is False and why == "server_group"

    def test_make_room_capacity_retry_is_bounded_by_timeout(self, monkeypatch):
        cfg = SimpleNamespace(placement_sync_timeout=0, placement_sync_interval=2)
        monkeypatch.setattr(wlb, "build_cluster_cache", lambda *_a, **_k: {})
        monkeypatch.setattr(wlb, "evaluate_aggregate_state", lambda **_k: PlanContext(aggregate="attempt-1"))
        monkeypatch.setattr(
            wlb, "validate_migration_step", lambda *_a, **_k: (False, "placement_vcpu")
        )
        monkeypatch.setattr(
            wlb.time, "sleep", lambda _s: pytest.fail("zero timeout must not sleep")
        )
        _, ok, why = wlb.wait_for_make_room_step(
            cfg, object(), "agg", ["h1", "h2"], object(),
            enforce_server_groups=False, exclude_unready_hosts=True,
        )
        assert ok is False and why == "placement_vcpu"


def test_validate_step_fails_when_nova_cannot_report_current_host():
    server = {"status": "ACTIVE"}
    conn = SimpleNamespace(compute=SimpleNamespace(get_server=lambda _id: server))
    step = SimpleNamespace(vm_id="vm1", src_host="src", dst_host="dst")
    ok, why = wlb.validate_migration_step(
        SimpleNamespace(), conn, PlanContext(), step, enforce_server_groups=False
    )
    assert ok is False
    assert "current host" in why
