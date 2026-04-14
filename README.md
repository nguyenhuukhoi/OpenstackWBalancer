# OpenStack Workload Balancer

An operator-focused Python script that evaluates OpenStack compute aggregates and performs a single live migration when that migration is expected to improve cluster balance safely.

The script is designed for production environments where predictability matters more than aggressive rebalancing. It uses Prometheus metrics, OpenStack inventory data, server-group checks, destination-host readiness checks, and cooldown windows to reduce migration churn.

## What This Script Does

The balancer works at the aggregate level.

For each selected aggregate, it:

1. Collects host CPU and RAM usage from Prometheus.
2. Collects host capacity and allocation ratios from OpenStack and Prometheus.
3. Builds an inventory of VMs running on the aggregate.
4. Chooses an operating mode:
   - RAM critical hotspot
   - RAM hotspot
   - CPU hotspot
   - Pressure
   - Normal MAD-based balancing
5. Searches for the best single VM move that improves balance while respecting safety rules.
6. Either:
   - reports the result in monitoring mode,
   - shows the proposed migration in dry-run mode,
   - or performs the migration and monitors its outcome.

This script performs at most one migration per aggregate per run.

## Core Decision Logic

The script uses both spread and pressure signals.

### 1. Hotspot detection takes priority

If any host crosses a hotspot threshold, the script switches from normal balancing to hotspot handling.

Default hotspot thresholds:

- RAM hotspot: `75%`
- RAM critical hotspot: `85%`
- CPU hotspot: `85%`

When a hotspot exists, the balancer first looks for moves from the hottest source host before falling back to a full aggregate search.

### 2. Pressure mode comes next

If there is no hotspot but the cluster average CPU or RAM exceeds the pressure threshold, the script enters `pressure` mode.

Default pressure threshold:

- Cluster average CPU or RAM above `80%`

### 3. Otherwise, use MAD-based balancing

If there is no hotspot and no pressure, the script uses a weighted Mean Absolute Deviation (MAD) model to measure imbalance across hosts.

Default acceptable host deviation:

- Target MAD: `3`

### 4. A candidate move must be safe and useful

A VM move is considered only if all of the following are true:

- The VM is not in cooldown after a recent successful migration.
- The VM is not in cooldown after a recent failed move.
- The destination host is a ready `nova-compute` service.
- The destination has enough allocatable CPU and RAM for the VM flavor.
- The move does not violate server-group affinity or anti-affinity rules.
- In hotspot mode, the hotspot actually improves and the destination must not become a hotspot.

### 5. A candidate move must clear the improvement threshold

The script does not migrate just because a move is technically possible.

It requires a minimum improvement:

- In hotspot modes: a fixed minimum improvement floor is used.
- In other modes: the threshold is proportional to the current weighted MAD.

Default minimum improvement floor:

- `0.05`

## Safety Features

This script is intentionally conservative.

It includes:

- VM cooldown after successful migration: `1800` seconds
- VM cooldown after failed migration: `3600` seconds
- Destination-host readiness checks using `nova-compute`
- Server-group policy checks by default
- Migration outcome monitoring with timeout detection
- Optional reset from `ERROR` back to `ACTIVE` after failed migration attempts
- Persistent cooldown state stored on disk
- Migration event logging to JSONL

## Monitoring and Execution Modes

### `--monitor-only`

Use this mode when you want an Icinga/Nagios-compatible health result without changing anything.

It prints a human-readable summary and exits with:

- `0` for `OK`
- `1` for `WARNING`
- `2` for `CRITICAL`
- `3` for `UNKNOWN`

### `--dry-run`

Use this mode to see what the script would migrate without performing the migration.

It still evaluates the aggregate fully and records a `dry_run` event if a valid move exists.

### Default execution mode

Without `--monitor-only` or `--dry-run`, the script:

- evaluates the aggregate,
- shows the proposed move,
- asks for confirmation,
- performs one live migration,
- monitors the result,
- updates cooldown state,
- optionally sends email alerts.

Use `-y` or `--yes` to skip the confirmation prompt.

## Requirements

You need:

- Python 3
- Access to OpenStack via `openstacksdk`
- A configured OpenStack cloud profile named `openstack`
- Prometheus endpoints and metrics reachable from the host where the script runs

Python packages used by the script:

- `openstacksdk`
- `requests`
- `python-dotenv`
- `click`
- `urllib3`

Install them with your preferred package manager, for example:

```bash
pip install openstacksdk requests python-dotenv click urllib3
```

## OpenStack Requirements

The script connects using:

- cloud name: `openstack`
- compute microversion: `2.87`

Make sure your environment has a valid `clouds.yaml` or equivalent OpenStack configuration that defines a cloud named `openstack`.

## Prometheus Requirements

The script expects Prometheus data for:

- host CPU usage
- host RAM usage
- OpenStack placement allocation ratios
- per-VM CPU usage
- per-VM RAM usage

The default queries assume the presence of metrics from:

- a node exporter job
- a libvirt exporter job
- an OpenStack exporter job

The script also expects specific labels to exist in Prometheus results:

- `alias` for host-level usage queries
- `hostname` for allocation ratio queries
- `instanceId` and `host` for per-VM queries

If your metric names or labels differ, override the Prometheus queries in the config file.

## Configuration

The script loads configuration from a dotenv-style file.

Default config file path:

```bash
/etc/loadleveller-secrets.conf
```

Supported environment variables:

- `PROMETHEUS_QUERY_URL`
- `PROMETHEUS_MEM_USED`
- `PROMETHEUS_CPU_USED`
- `PROMETHEUS_CPU_RATIO`
- `PROMETHEUS_MEM_RATIO`
- `ALERT_EMAIL_TO`
- `EMAIL_TO`
- `ALERT_EMAIL_FROM`
- `EMAIL_FROM`
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `SMTP_STARTTLS`

Example:

```dotenv
PROMETHEUS_QUERY_URL=http://kprometheus.com:9090/api/v1/query

ALERT_EMAIL_TO=ops@example.com,cloud@example.com
ALERT_EMAIL_FROM=loadleveller@example.com
SMTP_SERVER=smtp.example.com
SMTP_PORT=25
SMTP_USER=
SMTP_PASSWORD=
SMTP_STARTTLS=false
```

## Files Written by the Script

By default, the script writes:

- cooldown state: `/var/log/loadleveller_vm_cooldown.json`
- migration events: `/var/log/loadleveller_migration_events.jsonl`

You can override both at runtime:

- `--cooldown-file`
- `--events-file`

## Command-Line Usage

Show all options:

```bash
python wloadbalancer.py --help
```

Monitor all aggregates:

```bash
python wloadbalancer.py --monitor-only
```

Monitor one aggregate:

```bash
python wloadbalancer.py --monitor-only --aggregate my-compute-aggregate
```

Preview the best move without changing anything:

```bash
python wloadbalancer.py --dry-run --aggregate my-compute-aggregate
```

Run interactively and ask before migrating:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate
```

Run non-interactively:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --yes
```

Send email alerts for failed migrations only:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --send-error
```

Send email alerts for both successful and failed migrations:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --send-all
```

Disable server-group checks only if you fully understand the risk:

```bash
python wloadbalancer.py --aggregate my-compute-aggregate --no-server-groups
```

## Recommended Operator Workflow

For production use, the safest workflow is:

1. Start with `--monitor-only` to understand the current cluster state.
2. Run `--dry-run` to inspect the proposed candidate and expected improvement.
3. Run interactively without `-y` for the first execution.
4. Enable `--send-error` or `--send-all` if you want email notifications.
5. Use custom `--cooldown-file` and `--events-file` paths if your environment requires different storage locations.

## Understanding the Output

The monitoring summary includes:

- the selected decision mode
- cluster peaks and averages
- CPU and RAM MAD
- adaptive weights
- whether a valid move exists
- the best candidate move
- the reason no move was selected, if applicable

When a migration is proposed, the runtime output also shows:

- source and destination hosts
- estimated CPU and RAM impact
- post-migration host percentages
- post-migration weighted MAD
- improvement for that migration round

## Event Log Format

The JSONL events file is append-only and records operational outcomes such as:

- `dry_run`
- `cancelled`
- `skipped_destination_not_ready`
- `submit_or_monitor_failed`
- `migration_success`
- `migration_timeout`
- `migration_failed`
- `stuck_active`

Each event may contain:

- aggregate name
- result
- detail status
- severity
- duration
- timestamps
- improvement values
- VM metadata
- source and destination hosts

## Operational Notes

- The script does not continuously rebalance until the cluster is perfect.
- It chooses one best move per aggregate per invocation.
- It favors predictable improvement over aggressive churn.
- It is designed to be run repeatedly by an operator, scheduler, or monitoring system.

## Troubleshooting

### The script says no aggregates matched selection

Check the aggregate name passed with `--aggregate` and verify it exists in OpenStack.

### The script skips a destination host

The most common reasons are:

- `nova-compute` is not enabled
- `nova-compute` is not up
- the host is forced down
- the destination lacks allocatable capacity

### The script finds no valid move

Common reasons:

- all candidates are below the improvement threshold
- server-group rules block the move
- cooldown blocks the VM
- destination hosts are not ready
- no technically feasible migration exists

### The migration appears to start but times out

The script monitors live migration status for up to `600` seconds by default. A timeout means the VM did not reach a terminal success or failure condition within that window.

## Summary

Use this script when you want a cautious, explainable, single-step balancing tool for OpenStack aggregates.

It is best suited for environments where:

- live migration is allowed,
- Prometheus metrics are trusted,
- server-group policies matter,
- and operators prefer controlled, observable balancing over rapid automation.
