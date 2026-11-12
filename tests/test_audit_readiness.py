"""Production audit regressions: scope, stale observations and recovery ownership."""
import json
import os
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import mock_open, patch

import pytest

from compute_maintenance.adapters.persistence import HostState
from compute_maintenance.application.maintenance import Runner
from compute_maintenance.application.recovery import MigrationJournal
from compute_maintenance.domain.errors import MigrationExecutionError, PrometheusError
from compute_maintenance.domain.maintenance_errors import StepFailed
from tests.test_production_contracts import Clock


@pytest.mark.parametrize('status', ['finished', 'done'])
def test_settle_accepts_quiescent_resize_and_evacuate(tmp_path, status):
    clock = Clock()
    journal = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    compute = NS(migrations=lambda: [{'status': status, 'updated_at': '1970-01-01T00:16:40'}])
    journal.wait_settled(NS(compute=compute))
    assert clock.waits == [360]


@pytest.mark.parametrize('status', ['confirming', 'reverting'])
def test_settle_waits_for_resize_transition(tmp_path, status):
    clock = Clock()
    journal = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    compute = NS(migrations=lambda: [{'status': status if clock.now == 1000 else 'confirmed',
                                        'updated_at': '1970-01-01T00:16:50'}])
    journal.wait_settled(NS(compute=compute))
    assert clock.waits == [10, 360]


def test_settle_queries_each_selected_host_and_excludes_unrelated_cloud(tmp_path):
    clock = Clock()
    journal = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    calls = []
    def migrations(**query):
        calls.append(query)
        if not query:
            return [{'status': 'running', 'source_compute': 'other', 'dest_compute': 'elsewhere'}]
        return [{'id': '1', 'source_compute': 'a', 'dest_compute': 'b', 'status': 'completed',
                     'updated_at': '1970-01-01T00:16:40'}]
    journal.wait_settled(NS(compute=NS(migrations=migrations)), hosts=['a', 'b'])
    assert {tuple(q.items()) for q in calls} == {(('host', 'a'),), (('host', 'b'),)}
    assert clock.waits == [360]


@pytest.mark.parametrize('saved', [
    {},
    {'schema_version': 1},
    {'schema_version': 1, 'pending': [], 'last_completion': 1000},
    {'schema_version': 1, 'pending': None, 'last_completion': float('nan')},
    {'schema_version': 1, 'pending': None, 'last_completion': -1},
    {'schema_version': 1, 'pending': {'vm_id': 'vm'}, 'last_completion': 1000},
])
def test_malformed_journal_cannot_become_permission_to_submit(tmp_path, saved):
    path = tmp_path / 'migration-journal.json'
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    with pytest.raises(MigrationExecutionError):
        MigrationJournal(tmp_path)
    assert path.read_bytes() == before


def test_openstack_connection_bounds_api_wait(monkeypatch):
    from compute_maintenance.adapters import openstack as adapter
    calls = []
    monkeypatch.setattr(adapter.openstack, 'connect', lambda **kw: calls.append(kw))
    adapter.get_openstack_connection()
    assert calls == [{'cloud': 'openstack', 'api_timeout': 30}]


@pytest.mark.parametrize('as_object', [False, True])
def test_service_host_takes_precedence_over_hypervisor_name(as_object):
    from compute_maintenance.domain.server import get_server_current_host
    fields = {'compute_host': 'compute-service', 'hypervisor_hostname': 'hypervisor.example'}
    server = NS(**fields) if as_object else fields
    assert get_server_current_host(server) == 'compute-service'


def test_journal_reconciles_destination_with_different_hypervisor_name(tmp_path):
    clock = Clock()
    journal = MigrationJournal(tmp_path, clock=clock, sleeper=clock.sleep)
    journal.begin(NS(vm_id='vm', src_host='a', dst_host='b'), set())
    server = NS(status='ACTIVE', compute_host='b', hypervisor_hostname='b.example', task_state=None)
    journal.reconcile(NS(compute=NS(get_server=lambda _: server)))
    assert journal.state.data['pending'] is None


def fixture_runner():
    from test_maintenance_safety import SafetyTests
    fixture = SafetyTests()
    fixture.setUp()
    return fixture


@pytest.mark.parametrize('entry', ['rolling-maint', 'rolling_maint.py', 'compute_maintenance.cli.maintain'])
def test_live_owner_recognizes_installed_and_module_entrypoints(entry):
    f = fixture_runner()
    try:
        state = f.checkpoint()
        state.save(result='running', runner=socket.gethostname(), pid=987654)
        with patch('builtins.open', mock_open(read_data=('python\0' + entry + '\0').encode())), \
                patch.object(HostState, 'load', return_value=state.data):
            assert f.runner.live_owner('h1') == 987654
    finally:
        f.doCleanups()


def test_preflight_ignores_down_services_outside_selected_aggregate():
    f = fixture_runner()
    try:
        f.cloud.nova_services = lambda: [dict(f.service, Host='h1'),
                                       dict(f.service, Host='h2'),
                                       dict(f.service, Host='other', State='down')]
        f.cloud.agents_dead = lambda _: []
        f.cloud.migrating = list
        Runner.preflight(f.runner, ['h1'])
        f.cloud.nova_services = lambda: [dict(f.service, Host='h1'),
                                       dict(f.service, Host='h2', State='down')]
        with pytest.raises(StepFailed, match='h2'):
            Runner.preflight(f.runner, ['h1'])
    finally:
        f.doCleanups()


def test_restore_cannot_adopt_external_enable_without_our_intent():
    f = fixture_runner()
    try:
        state = f.checkpoint('restore')
        f.runner.health_once = lambda *_: (True, 'healthy')
        with pytest.raises(StepFailed, match='restore intent'):
            f.runner.guard_stage('restore', f.node, state)
    finally:
        f.doCleanups()


@pytest.mark.parametrize('field,value', [('task_state', 'resize_migrating'), ('is_locked', True)])
def test_final_validation_rejects_new_task_or_lock(field, value):
    from compute_maintenance.application.migrations import validate_migration_step
    server = dict(status='ACTIVE', compute_host='a', **{field: value})
    ok, reason = validate_migration_step(NS(), NS(compute=NS(get_server=lambda _: server)), {},
                                         NS(vm_id='vm', src_host='a', dst_host='b'))
    assert not ok
    assert 'task' in reason or 'locked' in reason


def test_cpu_raw_sample_age_cannot_be_masked_by_fresh_memory(monkeypatch):
    from compute_maintenance.adapters import freshness
    def query(_url, expression):
        metric = {'alias': 'a'}
        age = '500' if 'node_cpu_seconds_total' in expression else '1'
        return {'data': {'result': [{'metric': metric, 'value': [1000, age]}]}}
    monkeypatch.setattr(freshness, 'do_query', query)
    with pytest.raises(PrometheusError, match='node_cpu'):
        freshness.validate_runtime_freshness(NS(prometheus_query_url='url',
            prometheus_node_job='node', prometheus_libvirt_job='libvirt'), ['a'], {})


APPROVED_PLAN = 'Inst pkg [1] (2 repo [amd64])\nInst lib [1] (2 repo [amd64])'


def _fake_apt(tmp_path):
    """dpkg/apt-get stubs: 'apt-get -s' prints the simulation, a real
    dist-upgrade only touches a marker file."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    for name, body in [('dpkg', 'printf "%s" "$FAKE_AUDIT"'),
                       ('apt-get', ('case " $* " in *" -s "*) printf "%s" "$FAKE_SIMULATION";; '
                                    '*) touch "$FAKE_MARKER";; esac'))]:
        executable = bindir / name
        executable.write_text('#!/bin/sh\n' + body + '\n')
        executable.chmod(0o755)
    return bindir


def _run_script(tmp_path, name, simulation, audit=''):
    bindir = _fake_apt(tmp_path)
    marker = tmp_path / 'upgraded'
    script = Path(__file__).resolve().parents[1] / 'profiles/scripts' / name
    result = subprocess.run(['sh', str(script)], env=dict(os.environ,
        PATH=str(bindir) + ':' + os.environ['PATH'], FAKE_AUDIT=audit, FAKE_SIMULATION=simulation,
        FAKE_MARKER=str(marker), WLB_PACKAGE_PLAN=APPROVED_PLAN), capture_output=True, check=False)
    return result, marker.exists()


@pytest.mark.skipif(os.name != 'posix', reason='real POSIX shell contract')
@pytest.mark.parametrize('simulation,audit,expected', [
    # same plan, nothing installed yet
    (APPROVED_PLAN + '\n', '', 10),
    # apt-daily refreshed lists: newer version of an approved package
    ('Inst pkg [1] (3 repo [amd64])\nInst lib [1] (2 repo [amd64])\n', '', 10),
    # partial upgrade / unattended-upgrades installed part of the plan
    ('Inst lib [1] (2 repo [amd64])\n', '', 10),
    # a package that was never approved appeared
    ('Inst pkg [1] (2 repo [amd64])\nInst linux-image-6.8.0-99 (1 repo [amd64])\n', '', 1),
    ('Inst pkg [1] (2 repo [amd64])\n', 'half configured', 1),
    ('Remv pkg [1]\n', '', 1),
    ('', '', 0),
])
def test_upgrade_reconcile_retry_allows_only_approved_packages(tmp_path, simulation, audit, expected):
    result, _ = _run_script(tmp_path, 'check-upgrade-result.sh', simulation, audit)
    assert result.returncode == expected, result.stderr


@pytest.mark.skipif(os.name != 'posix', reason='real POSIX shell contract')
@pytest.mark.parametrize('simulation,expected,upgraded', [
    (APPROVED_PLAN + '\n', 0, True),
    ('Inst lib [1] (3 repo [amd64])\n', 0, True),           # version bump, subset
    ('', 0, False),                                          # already fully installed
    ('Inst other [1] (2 repo [amd64])\n', 1, False),         # not approved
    ('Inst pkg [1] (2 repo [amd64])\nRemv lib [1]\n', 1, False),
])
def test_dist_upgrade_installs_only_an_approved_subset(tmp_path, simulation, expected, upgraded):
    result, did_upgrade = _run_script(tmp_path, 'dist-upgrade.sh', simulation)
    assert result.returncode == expected, result.stderr
    assert did_upgrade is upgraded
