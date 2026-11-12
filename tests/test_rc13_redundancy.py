"""rc13 performance-only release (REVIEW-REDUNDANCY-2026-11-11, class A): API
calls whose results were thrown away or fetched twice in the same pass are gone;
decisions, output and events are unchanged.

A1/A2: no cluster cache is prefetched for a caller whose settle step discards it.
A3/A4: one service-host -> hypervisor map per validation and per snapshot pass.
A6: validation does not GET the step's VM a second time for its server group.
"""
import collections
import itertools
from types import SimpleNamespace as NS

import pytest

import compute_maintenance.adapters.openstack as OS
import compute_maintenance.application.balancing as BAL
import compute_maintenance.application.migrations as MIG
import compute_maintenance.application.snapshots as S
import compute_maintenance.config as C
from compute_maintenance.domain import policies
from compute_maintenance.domain.plan_context import PlanContext

JOURNAL = object()


def placement_record(cores=32, ram_gb=256):
    record = policies.build_placement_inventory_record
    return {"resource_provider_uuid": "rp", "resource_provider_name": "rp",
            "VCPU": record({"total": cores, "reserved": 0, "allocation_ratio": 1.0}, 0, 0),
            "MEMORY_MB": record({"total": ram_gb * 1024, "reserved": 0, "allocation_ratio": 1.0}, 0, 0)}


class Nova:
    """Fake Nova compute proxy that counts reads. Hypervisor names are "<host>.hv"."""

    def __init__(self, hosts, servers=(), groups=(), hypervisors=None, failing_details=()):
        self.calls = collections.Counter()
        self.hv = hypervisors if hypervisors is not None else [(h + ".hv", h) for h in hosts]
        self.failing_details = set(failing_details)
        self.servers_by_id = dict(servers)
        self.groups = list(groups)

    def hypervisors(self, **_):
        self.calls["hypervisors"] += 1
        return [NS(id=i, name=name) for i, (name, _host) in enumerate(self.hv)]

    def get_hypervisor(self, hv_id):
        self.calls["get_hypervisor"] += 1
        name, host = self.hv[hv_id]
        if hv_id in self.failing_details:
            raise RuntimeError("503")
        return NS(id=hv_id, name=name, service={"host": host})

    def get_server(self, vm_id):
        self.calls["get_server " + vm_id] += 1
        return self.servers_by_id[vm_id]

    def server_groups(self, all_projects=True):
        self.calls["server_groups"] += 1
        return list(self.groups)


@pytest.mark.parametrize("cfg,used", [
    (NS(), True),                                                         # dry-run / monitor / library
    (NS(settle_mode=C.SETTLE_MODE_BARRIER), True),
    (NS(migration_journal=JOURNAL), False),                               # apply: settle barrier discards it
    (NS(settle_mode=C.SETTLE_MODE_COMPENSATE), False),                    # drain dry-run
    (NS(settle_mode=C.SETTLE_MODE_COMPENSATE, migration_journal=JOURNAL), False),   # drain apply
])
def test_a_prefetched_cache_is_used_only_by_a_journal_less_barrier_caller(cfg, used):
    assert S.prefetched_cluster_cache_is_used(cfg) is used


# --- A1: drain does not prefetch the cache its compensate settle discards ----------------

def _run_drain_by_aggregate(monkeypatch, cfg):
    builds, seen = [], []
    monkeypatch.setattr(BAL._sources_openstack, "get_openstack_connection", lambda *a, **k: object())
    monkeypatch.setattr(BAL._sources_openstack, "get_aggregate_map",
                        lambda conn, aggregate_names=None, show_available_on_missing=False: {"agg": ["h1", "h2"]})
    monkeypatch.setattr(BAL._snapshot, "build_cluster_cache",
                        lambda *a, **k: builds.append(a[2]) or {"host_resources": {}})
    monkeypatch.setattr(BAL, "drain_host_workload", lambda **kw: seen.append(kw["cluster_cache"]) or 0)
    assert BAL.drain_by_aggregate(cfg, "h1", dry_run=True) == 0
    return builds, seen


@pytest.mark.parametrize("cfg", [
    NS(settle_mode=C.SETTLE_MODE_COMPENSATE),                            # what the CLI sets for --drain-host
    NS(settle_mode=C.SETTLE_MODE_COMPENSATE, migration_journal=JOURNAL),
    NS(migration_journal=JOURNAL),
])
def test_drain_does_not_prefetch_a_cache_its_settle_step_discards(monkeypatch, cfg):
    builds, seen = _run_drain_by_aggregate(monkeypatch, cfg)
    assert builds == []
    assert seen == [None]


def test_journal_less_barrier_drain_still_prefetches(monkeypatch):
    builds, seen = _run_drain_by_aggregate(monkeypatch, NS())
    assert builds == [["h1", "h2"]]
    assert seen[0] is not None                                            # the library caller reads it


def test_drain_snapshot_is_collected_once_by_the_compensate_settle(monkeypatch, tmp_path):
    """Real drain_by_aggregate -> drain_host_workload -> evaluate_aggregate_state:
    the compensate branch builds the one cache the drain plans on."""
    import compute_maintenance.adapters.prometheus as P
    import compute_maintenance.application.recovery as REC
    import compute_maintenance.presentation.reports as REP
    from compute_maintenance.adapters.cooldown import CooldownStore
    from compute_maintenance.domain import policies
    hosts = ["A", "B", "C"]
    series = {"CPU": {"A": 30.0, "B": 25.0, "C": 28.0}, "MEM": {"A": 50.0, "B": 40.0, "C": 45.0}}
    record = policies.build_placement_inventory_record
    resource = {"total_cores": 32, "free_cores": 32, "total_ram_gb": 256, "free_ram_gb": 256, "running_vms": 0,
                "placement": {"resource_provider_uuid": "rp",
                              "VCPU": record({"total": 32, "reserved": 0, "allocation_ratio": 1.0}, 0, 0),
                              "MEMORY_MB": record({"total": 262144, "reserved": 0, "allocation_ratio": 1.0}, 0, 0)}}
    details = {"cpu_pct": 20.0, "ram_pct": 50.0, "ram_rss_gb": None, "vcpus": 2, "ram_gb": 4.0, "host": "A",
               "project_id": "p", "project_name": "p", "status": "ACTIVE", "metadata_complete": True,
               "metrics_complete": True, "locked": False, "task_state": None, "availability_zone": None,
               "unsupported_resources": None}
    builds, events = [], []
    monkeypatch.setattr(BAL._sources_openstack, "get_openstack_connection", lambda *a, **k: object())
    monkeypatch.setattr(BAL._sources_openstack, "get_aggregate_map",
                        lambda conn, aggregate_names=None, show_available_on_missing=False: {"agg": hosts})
    monkeypatch.setattr(S, "build_cluster_cache", lambda conn, cfg, h, **k: builds.append(list(h)) or {
        "host_resources": {x: resource for x in h}, "host_resource_errors": {}, "host_az": None,
        "server_cache": {}, "server_groups": [], "project_cache": {}})
    monkeypatch.setattr(P, "do_query", lambda url, query, at=None: {"status": "success", "data": {"result": [
        {"metric": {"alias": h}, "value": [1_700_000_000, str(v)]} for h, v in series[query].items()]}})
    monkeypatch.setattr(REC, "scan_host_migrations", lambda conn, hosts: (False, None, []))
    monkeypatch.setattr(S._sources_openstack, "is_nova_compute_service_ready", lambda conn, host: (True, ""))
    monkeypatch.setattr(S._sources_prometheus, "get_all_vm_metrics", lambda *a, **k: {})
    monkeypatch.setattr(S, "build_aggregate_vm_inventory",
                        lambda *a, **k: ({"vm": "vm"}, {"vm": details}, {"vm": "A"}))
    monkeypatch.setattr(REP, "append_migration_event", events.append)
    cfg = NS(prometheus_query_url="http://prom.test/api/v1/query", prometheus_query_cpu_used="CPU",
             prometheus_query_mem_used="MEM", cooldown_store=CooldownStore(str(tmp_path / "cool.json")),
             max_instances_per_host={}, max_instances_per_host_default=0, planner_candidate_budget=100000,
             placement_final_allocation_candidate_check=False, settle_mode=C.SETTLE_MODE_COMPENSATE)
    assert BAL.drain_by_aggregate(cfg, "A", dry_run=True) == C.DRAIN_EXIT_OK
    assert builds == [hosts]                                              # rc12: [hosts, hosts], one discarded
    assert [e["result"] for e in events] == ["drain_dry_run"]
    assert events[0]["drain_feasible"] is True


# --- A2: the MAKE_ROOM step-2 wait loop does not build a cache the barrier discards ------

def _run_wait_for_make_room_step(monkeypatch, cfg):
    builds, seen = [], []
    monkeypatch.setattr(MIG._snapshot, "build_cluster_cache", lambda *a, **k: builds.append(a[2]) or {"c": 1})
    monkeypatch.setattr(MIG._snapshot, "evaluate_aggregate_state",
                        lambda **kw: seen.append(kw) or PlanContext(aggregate="agg", error="stop"))
    cfg.placement_sync_timeout, cfg.placement_sync_interval = 0, 1
    _state, ok, why = MIG.wait_for_make_room_step(cfg, object(), "agg", ["A", "B"], object(),
                                                  exclude_unready_hosts=True)
    assert (ok, why) == (False, "refresh evaluation failed: stop")
    assert len(seen) == 1 and seen[0]["plan_balance"] is False
    return builds, seen[0]["cluster_cache"]


def test_make_room_step_two_does_not_build_a_cache_the_barrier_discards(monkeypatch):
    builds, cache = _run_wait_for_make_room_step(monkeypatch, NS(migration_journal=JOURNAL))
    assert builds == []
    assert cache is None                                                  # the refresh collects after its barrier


def test_make_room_step_two_journal_less_caller_still_gets_a_cache(monkeypatch):
    builds, cache = _run_wait_for_make_room_step(monkeypatch, NS())
    assert builds == [["A", "B"]]
    assert cache == {"c": 1}


# --- A3/A4 premise: a map read for more hosts, restricted, IS the map of fewer hosts ---------

def test_restricting_the_hypervisor_map_equals_fetching_it_for_the_subset():
    """Per host the map does not depend on which other hosts were asked for, also
    with two hypervisors on one service host (the last one wins), a failing detail
    read (that host keeps its own name) and a hypervisor named like another host."""
    hv = [("a.hv", "a"), ("b-old.hv", "b"), ("b.hv", "b"), ("c.hv", "c"), ("a", "d"), ("e.hv", "e")]
    conn = NS(compute=Nova([], hypervisors=hv, failing_details={3}))
    hosts = ["a", "b", "c", "d", "e"]
    full = OS.build_nova_host_to_hypervisor_map(conn, hosts)
    assert full == {"a": "a.hv", "b": "b.hv", "c": "c", "d": "a", "e": "e.hv"}
    for size in range(1, len(hosts) + 1):
        for subset in itertools.permutations(hosts, size):
            fetched = OS.build_nova_host_to_hypervisor_map(conn, list(subset))
            restricted = {h: full[h] for h in subset}
            assert list(restricted.items()) == list(fetched.items()), subset     # values and order


# --- A3 + A6: one map and one GET of the step's VM per validation ---------------------------

@pytest.fixture
def validation(monkeypatch):
    """Real validate_migration_step + get_fresh_server_group_context + hypervisor map;
    vm on A, its anti-affinity peer reports only its hypervisor name B.hv."""
    servers = {"vm": NS(id="vm", status="ACTIVE", compute_host="A", task_state=None, is_locked=False),
               "peer": NS(id="peer", status="ACTIVE", compute_host=None, hypervisor_hostname="B.hv",
                          task_state=None, is_locked=False)}
    group = NS(id="g", name="g", policy="anti-affinity", member_ids=["vm", "peer"], rules={})
    nova = Nova(["A", "B", "C"], servers=servers, groups=[group])
    providers = []
    monkeypatch.setattr(MIG._sources_openstack, "is_nova_compute_service_ready", lambda conn, host: (True, ""))
    monkeypatch.setattr(MIG._sources_openstack, "fetch_placement_host_resource",
                        lambda conn, cfg, host, provider_name=None: providers.append(provider_name)
                        or placement_record())
    state = PlanContext(all_vm_details={"vm": {"vcpus": 2, "ram_gb": 4.0}}, host_az=None,
                        host_max_instances={}, host_running_vms={})

    def run(dst, groups=True):
        nova.calls.clear()
        providers.clear()
        step = NS(vm_id="vm", src_host="A", dst_host=dst, vcpus=2, vm_ram_gb=4.0)
        return MIG.validate_migration_step(NS(placement_final_allocation_candidate_check=False),
                                           NS(compute=nova), state, step, enforce_server_groups=groups,
                                           require_runtime_improvement=False, prevent_new_hotspot=False)
    return NS(run=run, calls=nova.calls, providers=providers)


@pytest.mark.parametrize("dst,verdict", [("C", (True, "ok")), ("B", (False, "server_group"))])
def test_validation_reads_the_hypervisor_map_once_and_the_vm_once(validation, dst, verdict):
    # B is refused only because the shared map turns the peer's "B.hv" into host B.
    assert validation.run(dst) == verdict
    assert validation.calls["hypervisors"] == 1                   # rc12: 3 (source alias, Placement name, groups)
    assert validation.calls["get_hypervisor"] == 3                # one detail read per hypervisor, once
    assert validation.calls["get_server vm"] == 1                 # rc12: 2 (again as its own group member)
    assert validation.calls["get_server peer"] == 1
    assert validation.providers == [dst + ".hv"]                  # the one map names the Placement provider


def test_validation_without_server_groups_reads_the_map_once(validation):
    assert validation.run("B", groups=False) == (True, "ok")
    assert validation.calls["hypervisors"] == 1                   # rc12: 2
    assert validation.calls["get_server vm"] == 1


def test_group_context_alone_still_reads_every_member_and_its_own_map():
    servers = {"vm": NS(id="vm", compute_host="A"),
               "peer": NS(id="peer", compute_host=None, hypervisor_hostname="B.hv")}
    nova = Nova(["A", "B"], servers=servers,
                groups=[NS(id="g", name="g", policy="anti-affinity", member_ids=["vm", "peer"], rules={})])
    hosts, _groups = OS.get_fresh_server_group_context(NS(compute=nova), "vm", known_hosts=["A", "B"])
    assert hosts == {"vm": "A", "peer": "B"}
    assert (nova.calls["hypervisors"], nova.calls["get_server vm"], nova.calls["get_server peer"]) == (1, 1, 1)


# --- A4: one map per snapshot pass ----------------------------------------------------------

@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    """Real evaluate_aggregate_state / build_cluster_cache / get_host_resources_map /
    build_aggregate_vm_inventory / hypervisor map; the Nova server list, Placement
    and Prometheus are faked. VM "only-hv" reports only its hypervisor name C.hv."""
    import compute_maintenance.adapters.prometheus as P
    from compute_maintenance.adapters.cooldown import CooldownStore
    hosts = ["A", "B", "C"]
    servers = {"vm1": NS(id="vm1", name="vm1", compute_host="A", hypervisor_hostname="A.hv"),
               "only-hv": NS(id="only-hv", name="only-hv", compute_host=None, hypervisor_hostname="C.hv")}
    nova = Nova(hosts)
    series = {"CPU": {"A": 30.0, "B": 25.0, "C": 28.0}, "MEM": {"A": 50.0, "B": 40.0, "C": 45.0}}
    monkeypatch.setattr(OS, "build_server_cache", lambda conn, h: dict(servers))
    monkeypatch.setattr(OS, "fetch_host_availability_zones", lambda conn, h: {})
    monkeypatch.setattr(OS, "fetch_server_groups", lambda conn: [])
    monkeypatch.setattr(OS, "fetch_placement_host_resource", lambda *a, **k: placement_record())
    monkeypatch.setattr(OS._sources_prometheus, "get_runtime_capacity_map",
                        lambda cfg, h, missing=None: {x: {"logical_cpus": 32, "ram_gb": 256.0} for x in h})
    monkeypatch.setattr(P, "do_query", lambda url, query, at=None: {"status": "success", "data": {"result": [
        {"metric": {"alias": h}, "value": [1_700_000_000, str(v)]} for h, v in series[query].items()]}})
    monkeypatch.setattr(S._sources_openstack, "is_nova_compute_service_ready", lambda conn, host: (True, ""))
    monkeypatch.setattr(S._sources_prometheus, "get_all_vm_metrics", lambda *a, **k: {})
    cfg = NS(prometheus_query_url="http://prom.test/api/v1/query", prometheus_query_cpu_used="CPU",
             prometheus_query_mem_used="MEM", cooldown_store=CooldownStore(str(tmp_path / "cool.json")),
             max_instances_per_host={}, max_instances_per_host_default=0, planner_candidate_budget=100000,
             placement_final_allocation_candidate_check=False)
    conn = NS(compute=nova)

    def evaluate(cache=None):
        return S.evaluate_aggregate_state(cfg, conn, "agg", hosts, cluster_cache=cache, plan_balance=False,
                                          exclude_unready_hosts=True)
    return NS(evaluate=evaluate, calls=nova.calls, cfg=cfg, conn=conn, hosts=hosts)


def test_a_snapshot_pass_reads_the_hypervisor_map_once(snapshot):
    state = snapshot.evaluate()
    assert state.error is None
    assert snapshot.calls["hypervisors"] == 1                     # rc12: 2 (cache + VM inventory)
    assert dict(state.vm_host_map) == {"vm1": "A", "only-hv": "C"}   # the reused map still translates C.hv


def test_a_cache_from_an_earlier_pass_does_not_lend_its_map(snapshot):
    cache = S.build_cluster_cache(snapshot.conn, snapshot.cfg, snapshot.hosts)
    assert cache["host_to_hypervisor"] == {"A": "A.hv", "B": "B.hv", "C": "C.hv"}
    snapshot.calls.clear()
    state = snapshot.evaluate(cache)                              # dry-run/monitor: prefetched cache
    assert dict(state.vm_host_map) == {"vm1": "A", "only-hv": "C"}
    assert snapshot.calls["hypervisors"] == 1                     # the inventory reads its own, as in rc12


@pytest.mark.parametrize("given,fetches", [
    ({"A": "A.hv", "B": "B.hv", "C": "C.hv"}, 0),
    ({"A": "A.hv", "B": "B.hv"}, 1),                              # does not cover C: fetched
    (None, 1),
])
def test_inventory_reuses_only_a_map_that_covers_its_hosts(given, fetches):
    nova = Nova(["A", "B", "C"])
    servers = {"x": NS(id="x", name="x", compute_host=None, hypervisor_hostname="C.hv")}
    _names, _details, vm_hosts = S.build_aggregate_vm_inventory(NS(compute=nova), ["A", "B", "C"], servers, {}, {},
                                                                host_to_hypervisor=given)
    assert vm_hosts == {"x": "C"}
    assert nova.calls["hypervisors"] == fetches


# --- A5: Keystone project names resolve once per project per run ----------------------------

@pytest.fixture
def named_snapshot(snapshot, monkeypatch):
    """The snapshot fixture with VM details: 3 VMs in projects p1 (twice) and p2,
    p-broken cannot be resolved (Keystone error); counts Keystone reads per project."""
    reads = collections.Counter()
    failing = {"p-broken"}

    def get_project(project_id):
        reads[project_id] += 1
        if project_id in failing:
            raise RuntimeError("keystone 503")
        return NS(name="name-" + project_id)
    projects = {"vm1": "p1", "vm2": "p1", "vm3": "p2", "vm4": "p-broken"}
    servers = {vm: NS(id=vm, name=vm, compute_host="A", hypervisor_hostname="A.hv", status="ACTIVE",
                      flavor={"vcpus": 2, "ram": 4096}, project_id=project) for vm, project in projects.items()}
    monkeypatch.setattr(OS, "build_server_cache", lambda conn, h: dict(servers))
    monkeypatch.setattr(S._sources_prometheus, "get_all_vm_metrics", lambda *a, **k: {
        vm: {"host": "A", "cpu_pct": 10.0, "ram_pct": 20.0, "ram_rss_gb": None, "metrics_complete": True}
        for vm in servers})
    snapshot.conn.identity = NS(get_project=get_project)
    snapshot.reads = reads
    snapshot.failing = failing

    def names(state):
        return {vm: d["project_name"] for vm, d in state.all_vm_details.items()}
    snapshot.names = names
    return snapshot


def test_two_snapshots_of_one_run_ask_keystone_once_per_project(named_snapshot):
    first = named_snapshot.evaluate()
    second = named_snapshot.evaluate()
    expected = {"vm1": "name-p1", "vm2": "name-p1", "vm3": "name-p2", "vm4": "p-broken"}
    assert named_snapshot.names(first) == named_snapshot.names(second) == expected
    assert named_snapshot.reads["p1"] == 1 and named_snapshot.reads["p2"] == 1     # rc12: 2 each
    # An unresolved project is shown by its id and asked again by the next
    # snapshot (once per snapshot, as with the rc12 per-snapshot cache).
    assert named_snapshot.reads["p-broken"] == 2


def test_a_project_unresolved_once_gets_its_name_when_keystone_recovers(named_snapshot):
    named_snapshot.evaluate()
    named_snapshot.failing.clear()                                # Keystone is back
    assert named_snapshot.names(named_snapshot.evaluate())["vm4"] == "name-p-broken"


def test_names_are_per_run(named_snapshot):
    named_snapshot.evaluate()
    named_snapshot.cfg.project_names = None                       # a new run has a new config object
    named_snapshot.evaluate()
    assert named_snapshot.reads["p1"] == 2


def test_a_read_only_config_still_resolves_names_per_snapshot():
    class Frozen:
        __slots__ = ()
    assert S.run_project_names(Frozen()) == {}
    S.remember_project_names(Frozen(), {"p": "name"})               # nowhere to keep them; no error
