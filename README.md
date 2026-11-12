# OpenStack Workload Balancer

English | [Tiếng Việt](README.vi.md)

Keep OpenStack compute aggregates balanced, and drain or maintain compute hosts without downtime, using live migration.

v1 replaces the single-file `wloadbalancer.py`. The old script is still in the repository but deprecated; see the [migration guide](docs/migrating-from-wloadbalancer.md).

## Features

| Command | What it does |
|---|---|
| `wbalancer` | **Balance**: per aggregate and per run, makes at most one plan, either one live migration or a two-step "make room" plan. It acts only when a host is a hotspot or CPU/RAM is uneven across hosts (MAD), and only if the move improves things enough. |
| `wbalancer --drain-host HOST` | **Drain**: empties a host and spreads its VMs so the remaining hosts stay balanced. Fail-closed: if the whole evacuation cannot be planned safely, nothing moves. |
| `rolling-maint` | **Rolling maintenance**: one host at a time it disables, drains, runs your maintenance profile, reboots, health-checks and re-enables. Re-running the same command resumes where it stopped. |
| `wbalancer --monitor-only` | **Monitoring**: read-only evaluation with Nagios/Icinga exit codes (0 OK, 1 WARNING, 2 CRITICAL, 3 UNKNOWN). |

Every command supports `--dry-run`.

Safety rules applied to every move:

- **Capacity:** Placement is the capacity source, with 5% headroom kept free.
- **Policies:** server groups as Nova applies them, including soft (anti-)affinity; availability zones; max instances per host.
- **No new hotspot:** the destination host must not become a hotspot after the move.
- **Skipped VMs:** locked or busy VMs, and VMs whose flavor cannot be modelled (CPU pinning, PCI, `resources:*`), are never moved.
- **Execution:** per-VM cooldowns, a durable migration journal, and a migration timeout that keeps waiting while the migration still makes progress.

How decisions are made, with worked examples: [docs/algorithm.md](docs/algorithm.md).

## Requirements

- **OpenStack:** Nova and Placement, admin credentials, live migration working between computes with **shared storage** (e.g. Ceph RBD), computes grouped in host aggregates.
- **Prometheus:**
  - [node_exporter](https://github.com/prometheus/node_exporter) with an `alias` label equal to the Nova host name;
  - a libvirt exporter exposing `libvirt_domain_*` with `host` and `instanceId` labels (e.g. [prometheus-libvirt-exporter](https://github.com/zhangjianweibj/prometheus-libvirt-exporter)).
- **Coordinator:** a Linux host with Python 3.12. `rolling-maint` also needs SSH access to the computes.
- **OpenStack clients:** `openstacksdk` and `python-openstackclient` are **not** installed by this package; install them to match your cloud. The tested baseline is OpenStack **2026.1** (openstacksdk 4.10.0, python-openstackclient 9.0.1).

The full checklist, including PromQL queries to verify your metrics, is in the [user guide](docs/user-guide.md#2-can-your-system-use-it).

## Quick start

```bash
git clone https://github.com/nguyenhuukhoi/OpenstackWBalancer.git && cd OpenstackWBalancer
python3 -m venv /opt/wbalancer/venv            # python3 must be 3.12 (requires-python >=3.12,<3.13)
VENV=/opt/wbalancer/venv

$VENV/bin/pip install --require-hashes -r requirements.lock.txt      # the tool's own pinned deps
$VENV/bin/pip install --no-deps .
$VENV/bin/pip install -c https://releases.openstack.org/constraints/upper/2026.1 openstacksdk python-openstackclient   # your choice

mkdir -p /etc/wbalancer && install -m 600 wbalancer.conf.example /etc/wbalancer/wbalancer.conf
vi /etc/wbalancer/wbalancer.conf   # set PROMETHEUS_QUERY_URL and the two job names; all other keys are optional

$VENV/bin/wbalancer --monitor-only                 # read-only state, exit 0-3
$VENV/bin/wbalancer --dry-run -a my-aggregate      # show the plan, change nothing
```

- **Credentials:** any cloud from `clouds.yaml` (default `openstack`; choose another with `OS_CLOUD` or `--os-cloud`), or an openrc environment with `--os-cloud envvars` (see the [user guide](docs/user-guide.md#4-configuration)).
- **Thresholds:** hotspot, pressure, MAD and minimum-improvement thresholds are config keys (see the [user guide](docs/user-guide.md#42-configuration-file-etcwbalancerwbalancerconf)).
- **Before you enable anything automatic,** follow the step-by-step [first-run path](docs/user-guide.md#5-first-run-path): read-only first, then one supervised move, one drain, and one canary maintenance.

## Documentation

| Document | For |
|---|---|
| [User guide](docs/user-guide.md) | Requirements, installation, configuration, first-run path, command reference, exit codes, troubleshooting |
| [Algorithm](docs/algorithm.md) | Modes, scoring, safety filters, make-room, drain planning, with numeric examples |
| [Architecture](docs/architecture.md) | Code layering, invariants, state files, extension recipes — for developers |
| [Operations](docs/operations.md) | State files and locks, timeouts, what to do when a run stops, recovery |
| [Maintenance profiles](docs/profiles.md) | Writing your own `rolling-maint` profile and hooks |
| [Migrating from `wloadbalancer.py`](docs/migrating-from-wloadbalancer.md) | Config and threshold differences, switch-over steps, rollback |
| [CHANGELOG](CHANGELOG.md) | Release notes and verification results |

## Tested with

OpenStack 2026.1 deployed with Kolla-Ansible, Ceph RBD shared storage, OVN, on a lab of 3 computes and 47 VMs:

- **Drain + reboot canaries:** 19 VMs each, about 70 s per VM, no VM lost.
- **Fault injection:** real failed and aborted migrations, a stalled migration, a locked VM.
- **Drain load compensation:** measured error within ±0.03 percentage points.

Other environments should repeat the first-run path before production use.

## Development

```bash
uv sync --extra test --locked            # dev group pins the 2026.1 OpenStack clients for tests
uv run --extra test python -m pytest -q
uv build --wheel
```

The test suite never contacts a cloud. POSIX lock tests run on Linux only.

## License

[MIT](LICENSE)
