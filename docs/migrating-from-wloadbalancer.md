# Migrating from OpenstackWBalancer (`wloadbalancer.py`) to v1

English | [Tiếng Việt](migrating-from-wloadbalancer.vi.md)

For operators running the single-script version of [OpenstackWBalancer](https://github.com/nguyenhuukhoi/OpenstackWBalancer). v1 is an upgrade of that script: the same MAD/pressure/hotspot algorithm and the same CLI options, plus host drain and rolling maintenance.

## Differences you need to know

| Item | Old `wloadbalancer.py` | v1 |
|---|---|---|
| How to run | `python wloadbalancer.py …` | Install the package and run `wbalancer …` (plus the new `rolling-maint` command) |
| Python | ≥ 3.10 | 3.12 |
| `openstacksdk` | You install it | Still yours to install, baseline OpenStack 2026.1 (see [user guide section 3](user-guide.md#3-installation)) |
| OpenStack connection | `openstack.connect(cloud='openstack')` | Same call, so credentials that work today keep working; the cloud name can now be changed with `OS_CLOUD` / `--os-cloud` |
| Default config file | `/etc/loadleveller-secrets.conf` | `/etc/wbalancer/wbalancer.conf`, or point `--config` at the old file |
| Prometheus job names | Edited directly in the code (`PROMETHEUS_NODE_JOB = '…'`) | **Required** in the config: `PROMETHEUS_NODE_JOB`, `PROMETHEUS_LIBVIRT_JOB` |
| `PROMETHEUS_QUERY_URL` | Has a default value | **Required**; missing means an error |
| Allocation ratio | Read from openstack-exporter (`PROMETHEUS_CPU_RATIO`, `PROMETHEUS_MEM_RATIO`) | Read directly from **Placement**; openstack-exporter is not needed, the old keys are ignored with a warning |
| Cooldown file | `/var/log/loadleveller_vm_cooldown.json` | `/var/lib/wbalancer/vm-cooldown.json`. **Same format**: copy it over to keep the cooldowns |
| Event log | `/var/log/loadleveller_migration_events.jsonl` | `/var/log/wbalancer/migrations.jsonl` |
| Overlap protection | None | A lock; two cron runs never overlap |
| Per balance run | At most 1 migration per aggregate | At most 1 plan: 1 migration, or 2 migrations of the MAKE_ROOM kind (move a small VM away to make room) |
| New features | — | `--drain-host`, `rolling-maint` + profiles, a journal that reconciles unfinished migrations automatically, a progress-aware timeout, respect for availability zones and locked VMs |

All old CLI options still exist under the same names: `--config`, `--verbose`, `--dry-run`, `--monitor-only`, `-a/--aggregate`, `-y/--yes`, `--send-all`, `--send-error`, `--no-server-groups`, `--use-vm-ram-rss`, `--cooldown-file`, `--events-file`. The `--monitor-only` exit codes still follow the Icinga/Nagios convention (0–3).

## Changed thresholds: results may differ from the old version

| Threshold | Old version | v1 |
|---|---|---|
| RAM hotspot | > 75% | > 80% |
| RAM critical | > 85% | > 95% |
| CPU hotspot | > 85% | > 70% |
| Minimum improvement to move | 0.05 | 0.15 |
| Imbalance (MAD) | > 3 points | > 3 points (unchanged) |
| Pressure (average) | > 80% | > 80% (unchanged, but never reached first: a host above 80% is already a hotspot) |

Since rc7 these are config keys. To keep the old behaviour exactly, add to `wbalancer.conf`:

```ini
RAM_HOTSPOT_THRESHOLD=75
RAM_CRITICAL_THRESHOLD=85
CPU_HOTSPOT_THRESHOLD=85
MIN_IMPROVEMENT_THRESHOLD=0.05
```

With the v1 defaults, for example a host at RAM 78%, CPU 75%:
- Old version: RAM hotspot (above 75%), CPU normal (below 85%).
- v1: RAM normal (below 80%), CPU hotspot (above 70%).

The higher improvement threshold means v1 migrates **less**: a move that improves things by only 0.1 is still made by the old version and skipped by v1.

## Switch-over steps

1. **Install v1** into its own venv following [user guide section 3](user-guide.md#3-installation). Leave the old version in place.
2. **Move the config:**

   ```bash
   mkdir -p /etc/wbalancer
   cp /etc/loadleveller-secrets.conf /etc/wbalancer/wbalancer.conf
   chmod 600 /etc/wbalancer/wbalancer.conf
   ```

   Add 2 lines, using the values currently hard-coded in `wloadbalancer.py`:

   ```ini
   PROMETHEUS_NODE_JOB=<value of PROMETHEUS_NODE_JOB in the old script>
   PROMETHEUS_LIBVIRT_JOB=<value of PROMETHEUS_LIBVIRT_JOB in the old script>
   ```

   Remove `PROMETHEUS_CPU_RATIO`, `PROMETHEUS_MEM_RATIO`, `PROMETHEUS_OPENSTACK_EXPORTER_JOB` (if left in place they only cause a warning).
3. **Compare side by side, read-only:**

   ```bash
   python wloadbalancer.py --monitor-only -a AGG; echo "old=$?"
   /opt/wbalancer/venv/bin/wbalancer --monitor-only -a AGG; echo "v1=$?"
   /opt/wbalancer/venv/bin/wbalancer --dry-run -a AGG
   ```

   Different results are normal when a host sits between the two threshold sets in the table above.
4. **Disable the old version's cron** and wait for any running pass to finish. The two versions do not share a lock, so **never let both of them migrate**.
5. **Keep the cooldowns:**

   ```bash
   mkdir -p /var/lib/wbalancer
   cp /var/log/loadleveller_vm_cooldown.json /var/lib/wbalancer/vm-cooldown.json
   ```

   This way a VM the old version just moved is not moved again by v1 within 4 hours.
6. **Enable the v1 cron** following [user guide section 6.1](user-guide.md#61-balance). In Icinga/Nagios, only the check command needs to change, to `wbalancer --monitor-only`.

## Going back to the old version

Disable the v1 cron and re-enable the old cron. v1 does not modify the old version's config, cooldown or log files.
