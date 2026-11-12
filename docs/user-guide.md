# User guide

English | [Tiếng Việt](user-guide.vi.md)

For OpenStack operators using the tool for the first time. Read in order: sections 1–4 to install and configure, section 5 is the safe first-run path, the remaining sections are reference. If you currently run the old `wloadbalancer.py` script, also read the [migration guide](migrating-from-wloadbalancer.md).

## 1. What the tool does

Two commands, installed from one package (`compute-maintenance`):

| Command | Job | Example |
|---|---|---|
| `wbalancer` | **Balances** CPU/RAM across the computes of an aggregate using live migration | 3 hosts at RAM 40/50/60% → move 1 VM from the 60% host to the 40% host |
| `wbalancer --drain-host X` | **Moves every VM off host X** and spreads them evenly over the remaining hosts | Host X has 19 VMs → split them across the other 2 hosts so load stays balanced |
| `wbalancer --monitor-only` | **Evaluates only**, returns Icinga/Nagios exit codes | Use it as an alerting check |
| `rolling-maint` | **Rolling maintenance**, one host at a time: disable → drain → maintenance → reboot → health check → enable | Reboot all 6 computes in turn without shutting down a single VM |

The tool only uses live migration; it never shuts down or rebuilds a VM. Every command has `--dry-run`.

## 2. Can your system use it?

Check this whole table before installing. If anything is missing, the tool reports a configuration error or refuses to run.

| Requirement | How to check |
|---|---|
| OpenStack with Nova + Placement, an **admin** account | `openstack hypervisor list` and `openstack resource provider list` work |
| Live migration works between computes, with **shared storage** (e.g. Ceph RBD). The tool calls migration with `block_migration=False` | `openstack server migrate --live-migration <vm>` succeeds by hand |
| Computes are in **host aggregates**, at least 2 hosts per aggregate | `openstack aggregate show <agg>` |
| Prometheus has **node_exporter** on every compute, with label `alias` = Nova host name | Query A below |
| Prometheus has a **libvirt exporter** with the `libvirt_domain_*` metrics, label `host` = Nova host name and `instanceId` = VM UUID (e.g. [prometheus-libvirt-exporter](https://github.com/zhangjianweibj/prometheus-libvirt-exporter), the exporter the original project recommends) | Queries B and C below |
| A Linux **coordinator** machine with Python 3.12 that can reach the OpenStack APIs and Prometheus | `python3.12 --version` |
| `rolling-maint` only: SSH access to the computes (default `root`), with their host keys already trusted by the coordinator's SSH configuration (see section 7.2); the package/kernel profiles are for Ubuntu/Debian; Kolla-style containers (`docker` or `podman`) | `ssh -o BatchMode=yes root@<compute> uptime` |

Check the metrics (replace `node`, `libvirt` with your real job names):

```promql
# A. One line per compute; alias must match the Host column of "openstack compute service list --service nova-compute"
count by (alias) (node_memory_MemTotal_bytes{job="node"})

# B. One line per compute that has VMs, value = number of VMs, host label matches the Nova host name
#    (an empty host has no line; that is normal)
count by (host) (libvirt_domain_info_virtual_cpus{job="libvirt"})

# C. Equals the number of running VMs (each series carries instanceId = VM UUID)
count(libvirt_domain_info_cpu_time_seconds_total{job="libvirt", instanceId!=""})
```

The tool also needs `node_cpu_seconds_total`, `node_memory_MemAvailable_bytes`, `libvirt_domain_stat_memory_usable_bytes`, `libvirt_domain_stat_memory_available_bytes` and `libvirt_domain_stat_memory_rss_bytes`.

If the `alias` or `host` label is missing, attach it in the scrape config:

```yaml
- job_name: node
  static_configs:
    - targets: ["10.0.0.11:9100"]
      labels: {alias: compute-01}   # exact Nova host name
```

## 3. Installation

On the coordinator. Do not share a venv with the legacy version.

**The tool does not install `openstacksdk` and `python-openstackclient`.** Install these two packages yourself to match your cloud. The tested baseline is **OpenStack 2026.1**: `openstacksdk` 4.10.0 and `python-openstackclient` 9.0.1, from the official upper-constraints. The repository also ships an offline copy of that file as `constraints/openstack-2026.1.txt` (used to regenerate the lock file).

```bash
git clone https://github.com/nguyenhuukhoi/OpenstackWBalancer.git && cd OpenstackWBalancer
python3 -m venv /opt/wbalancer/venv            # python3 must be 3.12 (requires-python >=3.12,<3.13)
VENV=/opt/wbalancer/venv

# 1. The tool and its own libraries (hash-pinned, same versions as the 2026.1 baseline)
$VENV/bin/pip install --require-hashes -r requirements.lock.txt
$VENV/bin/pip install --no-deps .

# 2. OpenStack clients: you choose and install them, into the SAME venv
$VENV/bin/pip install -c https://releases.openstack.org/constraints/upper/2026.1 openstacksdk python-openstackclient

$VENV/bin/pip check
$VENV/bin/wbalancer --help && $VENV/bin/rolling-maint --help
```

The built-in profiles are installed to `/opt/wbalancer/venv/share/compute-maintenance/profiles/`.

To install against another baseline (e.g. a 2025.2 cloud), replace the constraints file with `https://releases.openstack.org/constraints/upper/2025.2`. Versions older than the 2026.1 baseline still run, but the tool prints the warning "older than the OpenStack 2026.1 baseline … untested"; repeat the path in section 5 before real use.

At startup the tool checks:

| Situation | Result |
|---|---|
| `openstacksdk` is not installed | Stops immediately and prints the install command; exit 1, or exit 3 (UNKNOWN) for `--monitor-only` |
| `rolling-maint` cannot find the `openstack` command | Stops before touching any host, exit 1 |
| Version older than the 2026.1 baseline | Continues and prints a warning |

`rolling-maint` uses the `openstack` command **from the same venv as the tool** first, and only looks in `PATH` when there is none. So running through `sudo`, cron or systemd does not pick up the system copy (e.g. `/usr/bin/openstack` 6.6.0). Each run's log records the path and version of the CLI it used.

To run the tests before installing (no cloud connection): `uv sync --extra test --locked && uv run --extra test python -m pytest -q`. The uv `dev` group installs both clients at the 2026.1 baseline, for tests only.

## 4. Configuration

### 4.1. OpenStack credentials

By default the tool connects to the cloud named `openstack`. To use another entry of `clouds.yaml`, set `OS_CLOUD=<name>` in the config file or the environment, or pass `--os-cloud <name>` to `wbalancer` / `rolling-maint`. Both commands pick the cloud the same way: `--os-cloud` > environment `OS_CLOUD` > `OS_CLOUD` in the config file (`--config`, for `rolling-maint` its `--wlb-config`) > `openstack`. `rolling-maint` always passes the cloud it picked to its drain step as `--os-cloud`, also to a custom `--wbalancer` child. Declare the cloud in `/etc/openstack/clouds.yaml` or `~/.config/openstack/clouds.yaml`, mode `600`:

```yaml
clouds:
  openstack:
    auth:
      auth_url: https://keystone.example:5000/v3
      application_credential_id: "..."
      application_credential_secret: "..."
    auth_type: v3applicationcredential
    region_name: RegionOne
```

The credential needs admin rights: the tool reads Placement, `os-migrations` and availability zone details, and enables/disables nova-compute.

With an openrc file instead of `clouds.yaml`: source it and use `--os-cloud envvars`. That is openstacksdk's name for the cloud built from the `OS_*` variables; `OS_CLOUD_NAME` changes it.

### 4.2. Configuration file `/etc/wbalancer/wbalancer.conf`

`KEY=VALUE` format. An environment variable with the same name overrides the value in the file. Start from [`wbalancer.conf.example`](../wbalancer.conf.example): it lists every key with its default and a short explanation.

```bash
install -m 600 wbalancer.conf.example /etc/wbalancer/wbalancer.conf
```

The minimum is 3 lines:

```ini
PROMETHEUS_QUERY_URL=http://prometheus.example:9090/api/v1/query
PROMETHEUS_NODE_JOB=node
PROMETHEUS_LIBVIRT_JOB=libvirt
```

Commonly used optional keys:

| Key | Default | Meaning |
|---|---|---|
| `EXCLUDE_VMS_FILE` | empty | File listing VMs that are never balanced |
| `MAX_INSTANCES_PER_HOST_DEFAULT` | `0` (no limit) | Maximum number of VMs per host |
| `MAX_INSTANCES_PER_HOST` | empty | Per-host JSON, e.g. `{"compute-01": 40}` |
| `PLACEMENT_VCPU_HEADROOM_PERCENT` / `PLACEMENT_RAM_HEADROOM_PERCENT` | `5` / `5` | Keep 5% of Placement capacity free instead of packing hosts full |
| `ENABLE_MAKE_ROOM` | `true` | Allow 2-step plans: move a small VM away to make room for a large one |
| `OS_CLOUD` | `openstack` | Cloud name in `clouds.yaml`; the environment variable `OS_CLOUD` overrides it, `--os-cloud` overrides both |
| `RAM_HOTSPOT_THRESHOLD` / `RAM_CRITICAL_THRESHOLD` | `80` / `95` | A host with RAM above this (%) is a hotspot / critical hotspot |
| `CPU_HOTSPOT_THRESHOLD` | `70` | A host with CPU above this (%) is a hotspot |
| `PRESSURE_THRESHOLD` | `80` | Aggregate average above this (%) means pressure mode; it only takes effect when it is lower than a hotspot threshold |
| `TARGET_AVG_MAD` | `3` | Spread (MAD, percentage points) above which the aggregate counts as unbalanced |
| `MIN_IMPROVEMENT_THRESHOLD` | `0.15` | Minimum improvement of a direct move; `MAKE_ROOM_MIN_IMPROVEMENT` defaults to twice this |
| `PLACEMENT_PRESSURE_THRESHOLD` | `90` | Placement allocation (%) from which a host is reported as nearly full (report only) |
| `MIGRATION_TIMEOUT` / `MIGRATION_MAX_WAIT` | `600` / `1800` | Seconds: stop waiting for a live migration after this long without progress / never wait longer than this in total (operations, section 4) |
| `HEAL_SOFT_ANTI_AFFINITY` | `true` | Re-spread a soft-anti-affinity server group that has 2+ members on one host, one VM per run, when no host is a hotspot (rc11); also enables the soft-anti-affinity tie-breaks. `false` = exact rc10 behaviour, broken groups are only reported |
| `SOFT_AA_HEAL_MAD_TOLERANCE` | `0.05` | Heal gate in MAD points (absolute, ≥ 0), checked for CPU and RAM separately: a MAD at or below `TARGET_AVG_MAD` must stay at or below it; a MAD already above it may grow by at most this value. Examples: balanced aggregate → a heal ending at CPU MAD 1.67 / RAM MAD 1.56 is allowed (both ≤ 3); CPU 0.5 / RAM 3.6 is refused (RAM > 3); RAM MAD 2.98 before → at most 3.00; RAM MAD 3.50 before → a heal ending at 3.51 is allowed, one ending at 3.60 is not |
| `SOFT_AA_TIEBREAK_MARGIN` | `0.15` | Relative band (0–1): among balancing moves within 15% of the best improvement, the one that heals a soft-anti-affinity group wins. Example: best 2.00, a healing move 1.80 (10% below) is chosen |
| `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM`, `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_STARTTLS` | off | Send alert emails. If a password is set, `chmod 600` this file |
| `PROMETHEUS_CPU_USED`, `PROMETHEUS_MEM_USED`, … | built from the job names | Replace the host queries if your labels differ |

Thresholds are checked at start-up (between 0 and 100, RAM critical above RAM hotspot); with `--verbose` the values in use are logged at the start of every run. To keep the old `wloadbalancer.py` behaviour, set `RAM_HOTSPOT_THRESHOLD=75`, `RAM_CRITICAL_THRESHOLD=85`, `CPU_HOTSPOT_THRESHOLD=85`, `MIN_IMPROVEMENT_THRESHOLD=0.05`.

`RUNTIME_SETTLE_SECONDS` (default 360) is an **environment variable**, not set in the file: after each migration, balance waits that many seconds so the metrics reflect the new load.

### 4.3. VM exclusion list (optional)

One entry per line: a UUID, an exact name or a glob of VM names; lines starting with `#` are comments.

```text
# DB cluster, never moved automatically
3f2a9c1e-0b7d-4e7e-9a51-2c3d4e5f6a7b
prod-db-*
```

## 5. First-run path

Go step by step; only move on when the previous step gives exactly the result you expect.

| Step | Command | Expect to see | Risk |
|---|---|---|---|
| 1. Read the state | `wbalancer --monitor-only -a AGG; echo $?` | Exit 0–2, never 3 (UNKNOWN = missing data) | None |
| 2. Review the balance plan | `wbalancer --dry-run -a AGG` | Proposed VM, source → destination, improvement | None |
| 3. Supervised balance | `wbalancer -a AGG` → answer `y` | 1–2 migrations, then it stops | Migrates a real VM |
| 4. Review the drain plan | `wbalancer --dry-run -a AGG --drain-host HOST` | Every VM has a destination, or an `unmovable` list with reasons | None |
| 5. Drain a host with few VMs | `wbalancer -a AGG --drain-host HOST` | Exit 0, host empty and still disabled | Real migrations, disables the host |
| 6. Try rolling-maint without touching the host | `dummy` profile (section 7) | `SKIPPED`, `Overall OK` | None |
| 7. rolling-maint canary | `drain-reboot` profile on 1 host | The host passes every stage and is re-enabled | Reboots a real host |
| 8. Enable the balance cron | Section 6.1 | 0–2 migrations per run | Automatic |

Run steps 1–2 daily for about a week before moving to step 3, to see whether the planner's proposals make sense.

## 6. Balancing and draining (`wbalancer`)

### 6.1. Balance

```bash
wbalancer --dry-run -a AGG                 # print the plan only
wbalancer -a AGG                           # ask [y/N] before migrating
wbalancer -y -a AGG --send-error           # auto-approve, email on error (for cron)
```

Per run, each aggregate executes **at most 1 plan**: 1 direct migration, 2 migrations of the MAKE_ROOM kind, or 1 soft-anti-affinity heal migration. A VM that was just moved is not moved again for 4 hours.

Example cron every 10 minutes, in `/etc/cron.d/wbalancer`:

```cron
*/10 * * * * root /opt/wbalancer/venv/bin/wbalancer -y --send-error >> /var/log/wbalancer/cron.log 2>&1
```

At most 6 runs/hour × 2 migrations = 12 migrations/hour per aggregate; in practice far fewer, because the tool only acts when the improvement is large enough. Two runs never overlap thanks to the lock. A run that starts while `rolling-maint` is running skips itself, prints `Not running: …` and exits 1; that is the intended behaviour.

When the tool decides balancing is needed:

| State | Condition | Example |
|---|---|---|
| Imbalance (MAD) | Mean absolute deviation of CPU% or RAM% across hosts > 3 points | RAM 40/50/60% → MAD 6.7 → balancing needed |
| Pressure | Aggregate average CPU or RAM > 80% | With the default thresholds this never triggers on its own: an average above 80% means some host is above 80%, and that host is handled as a hotspot first. Set `PRESSURE_THRESHOLD` below a hotspot threshold to use it ([details](algorithm.md#3-choosing-the-mode)) |
| Hotspot | A host with RAM > 80% (> 95% is critical) or CPU > 70% | Moving VMs off that host takes priority |
| Broken soft-anti-affinity group (rc11) | No hotspot, and 2+ members of a `soft-anti-affinity` server group share a host (e.g. after a drain) | db1 + db2 on A (RAM 70%), B 40%, C 45% → db2 (the smaller one) moves to B; one VM per run, so 3 violations take about 30 minutes with a 10-minute cron ([details](algorithm.md#8-soft-anti-affinity-heal-rc11)) |

All numbers in this table are the defaults and can be changed in the config file (section 4.2). A move is chosen only when it improves enough, does not push the destination over the hotspot threshold, and respects Placement, server groups and availability zones.

### 6.2. Draining a host

```bash
wbalancer --dry-run -a AGG --drain-host compute-03   # review the plan
wbalancer -a AGG --drain-host compute-03             # execute, asks for confirmation
```

- The tool **disables nova-compute** on the host before the drain, so the scheduler places no new VMs there. Add `--drain-no-disable` if you do not want that.
- **Fail-closed:** if a plan cannot be built for every **live-migratable** VM, no VM is migrated; the tool prints the reasons and exits 2. VMs that cannot be live-migrated at all (locked, SHUTOFF, `unsupported_resources`…, see the table below) do not block the drain: they are reported and left on the host, the other VMs are migrated, and the drain still ends with exit 2 because the host is not empty.
- Drain ignores the cooldown and the exclusion list for VMs on the drained host itself.
- Speed measured on the lab: about 70 seconds per VM, so 19 VMs take about 23 minutes.
- When finished the host **stays disabled**. After maintenance, give it back yourself:
  `openstack compute service set --enable compute-03 nova-compute`
- If the remaining hosts do not have room below the hotspot threshold, `--drain-allow-hotspot` lets the drain exceed it; the 95% RAM ceiling is always kept.

VMs in the `unmovable` list usually fall into one of these cases:

| Reason | Meaning | What to do |
|---|---|---|
| `locked` | The VM is locked | `openstack server unlock` if you are allowed to |
| `task_in_progress` | The VM has a Nova task running | Wait; the drain waits by itself |
| `unsupported_resources` | The flavor uses CPU pinning, PCI passthrough or `resources:*` | Move it by hand |
| ERROR state | The VM is in error | Fix the VM first |

### 6.3. Hooking into Icinga/Nagios

```bash
wbalancer --monitor-only -a AGG
```

| Exit | Meaning |
|---|---|
| 0 OK | Balanced |
| 1 WARNING | Imbalance, pressure or hotspot, but valid moves remain; or imbalance where the possible moves improve too little; or (rc11) a broken soft-anti-affinity group that a heal move can fix, status `SOFT_AA_HEAL` |
| 2 CRITICAL | Pressure or hotspot with **no valid move left**, or imbalance where no VM can move: more capacity or operator action is needed |
| 3 UNKNOWN | Missing metrics, Placement or OpenStack data |

Soft-anti-affinity (rc11): a broken group raises WARNING only when a heal move is possible; it never lowers or hides a CRITICAL, and the line `Load status:` shows what the balance check alone would report. A broken group that **cannot** be healed, either never (`unresolvable`, e.g. 4 members in a 3-host aggregate) or not this run (`deferred`, e.g. a member in cooldown, with the blocking counts), or any broken group with `HEAL_SOFT_ANTI_AFFINITY=false`, only adds a `Soft-anti-affinity: … unresolvable: …` / `… deferred: …` / `… heal disabled` line; the exit code stays what it would be without the group.

## 7. Rolling maintenance (`rolling-maint`)

Each host goes through the stages in this order: `preflight → pre_drain → disable → drain → verify → baseline → maintenance → reboot → health → post_boot → restore → post_restore`. The next host starts only when the previous one is completely done. The drain stage calls `wbalancer --drain-host` itself.

### 7.1. Built-in profiles

| Profile | What it does | Reboot |
|---|---|---|
| `dummy` | Does not touch the host, stops at `pre_drain`. For integration testing | No |
| `drain-reboot` | Drain, reboot, health check, re-enable | Always |
| `reboot-only` | Reboots only when a new kernel is already installed; verifies the right kernel after boot | Always |
| `kernel-update` | Runs dist-upgrade of exactly the packages in the approved plan, reboots when needed, verifies the kernel | When needed |

`kernel-update` has not yet been validated by a real upgrade; try it on a lab first.

### 7.2. How to run

Select the hosts in **one of two ways**:

```bash
P=/opt/wbalancer/venv/share/compute-maintenance/profiles

# Option 1: list the hosts and give their SSH addresses yourself
rolling-maint --profile $P/drain-reboot.yaml -a AGG \
  --host-address compute-03=10.0.0.13 --ssh-key /root/.ssh/id_ed25519 \
  --prom-url http://prometheus.example:9090 --dry-run compute-03

# Option 2: take hosts and SSH settings from the Kolla-Ansible inventory (needs ansible-inventory in PATH)
rolling-maint --profile $P/drain-reboot.yaml -a AGG -i /etc/kolla/multinode --dry-run
```

- With option 2, if you name no host, the tool runs every host that is both in the aggregate and in the inventory's `[compute]` group.
- When the dry-run looks right, drop `--dry-run`. The tool asks for confirmation for each host; add `-y` to auto-approve.
- Always set `--prom-url`: the tool then requires every Prometheus target of the host to be `up` before re-enabling it.
- SSH host keys: the tool follows your OpenSSH configuration (`~/.ssh/config`, `/etc/ssh/ssh_config`) and sets no host-key option itself. `--ssh-strict-host-key` was removed in 1.0.0rc10; drop it from your commands. ssh runs non-interactively (`BatchMode=yes`), so with OpenSSH's default policy a compute whose key is not in `known_hosts`, or whose key changed, stops the run with exit 255. Pre-populate `known_hosts` with the verified keys (`ssh-keyscan -t ed25519 compute-03 >> ~/.ssh/known_hosts`, then compare the fingerprints), or set your policy in `ssh_config` (for example `StrictHostKeyChecking accept-new` under `Host compute-*`). See [operations](operations.md#1-before-the-first-run).

Options you will often adjust:

| Option | Default | When to change it |
|---|---|---|
| `--drain-timeout` | 7200 s (~100 VMs) | Hosts with more VMs |
| `--boot-timeout` | 3600 s | Slow-booting physical machines: time one real reboot and add margin |
| `--health-timeout` / `--settle` | 1200 s / 300 s | The host must stay healthy for `settle` seconds in a row before it is enabled |
| `--required-containers` | `nova_compute nova_libvirt nova_ssh ovn_controller openvswitch_db openvswitch_vswitchd neutron_ovn_metadata_agent` | You use ML2/OVS instead of OVN |
| `--container-engine` | `docker` | You use `podman` |
| `--notify` | `all` | `fail` mails only on STOPPED, `summary` also on FINISHED, `off` disables mail |

Set the timeouts **from the very first run**, because they are part of the run's identity; if you change them mid-way the tool refuses to resume.

### 7.3. When a run is STOPPED

The tool stops as soon as something cannot be proven safe, and prints the command to continue:

```text
host compute-03 left at stage 'health' (nova-compute observed DISABLED).
  Fix the cause, then re-run (it continues there):  rolling-maint ... -y compute-03 compute-04
  Or give it back by hand:     openstack compute service set --enable compute-03 nova-compute
```

1. Read the logs in `/var/log/wbalancer/maintenance/` and check the real state in Nova.
2. Fix the cause.
3. **Re-run exactly the same command**, without changing the profile, config or tool version. The tool continues from the unfinished stage and does not repeat work already done (no second reboot, no re-migration of VMs already moved).
4. Migrations with an unknown outcome are reconciled with Nova automatically on the next run. Do not reset VM state or migrate by hand before re-running.

### 7.4. Writing your own profile

See [profiles.md](profiles.md): copy the template, write scripts for the `pre_drain`, `maintenance`, `post_boot` and `post_restore` hooks, then check the profile offline with `/opt/wbalancer/venv/bin/python tools/profile_check.py <profile.yaml>`. This command does not SSH and calls no API.

## 8. Data written by the tool

| Content | Default location |
|---|---|
| Migration history (JSONL, one event per line) | `/var/log/wbalancer/migrations.jsonl` |
| rolling-maint logs and results | `/var/log/wbalancer/maintenance/` |
| Maintenance checkpoints (for resume) | `/var/lib/wbalancer/maintenance/` |
| Migration journal | `/var/lib/wbalancer/operations/migration-journal.json` |
| VM cooldown | `/var/lib/wbalancer/vm-cooldown.json` |
| Locks | `/var/lock/wbalancer-maint.lock`, `/var/lock/wbalancer-balancer.lock` |

Do not delete or hand-edit checkpoints or the journal while a run is unfinished.

## 9. Exit codes

| Command | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `wbalancer` (balance) | Done, including when there was nothing to do | Error: config, data, migration, or the lock is held | — | — |
| `wbalancer --drain-host` | The host is empty | Drain failed mid-way, or you answered `N` | No plan could be built, nothing was migrated; **or** every live-migratable VM was migrated but non-live-migratable VMs (locked, SHUTOFF…) remain on the host | — |
| `wbalancer --monitor-only` | OK | WARNING | CRITICAL | UNKNOWN |
| `rolling-maint` | Every host DONE or SKIPPED | STOPPED (see section 7.3) | — | — |

With several aggregates, `--monitor-only` exits with the most severe one, ranked CRITICAL > WARNING > UNKNOWN > OK: one aggregate CRITICAL and another UNKNOWN (data error) exits 2, not 3.

## 10. Common errors

| Message | Cause | Fix |
|---|---|---|
| `openstacksdk is not installed in this Python` | The client was not installed, or was installed into another venv | Run the install command in section 3, step 2, in the tool's venv |
| `the 'openstack' CLI … was not found` | `python-openstackclient` is missing | Same as above |
| `… older than the OpenStack 2026.1 baseline …` | The client is older than the tested baseline | Upgrade with the 2026.1 constraints, or re-validate on a lab yourself |
| `Not running: state from an earlier version was found in /var/lib/wbalancer-v1 …` | Upgraded from 1.0.0rc8 or earlier; the state is still in the old `-v1` directory | Move it as printed ([operations](operations.md#2-state-files-and-locks)) |
| `PROMETHEUS_QUERY_URL is required` | The URL is missing from the config | Add the full URL, ending in `/api/v1/query` |
| `PROMETHEUS_NODE_JOB is required` | The job name is missing | Add `PROMETHEUS_NODE_JOB` and `PROMETHEUS_LIBVIRT_JOB` |
| `Aggregate X skipped: …` / monitor returns 3 | A host lacks metrics, the `alias` label, or Placement data | Run queries A and B from section 2. Other aggregates are still processed |
| `Not running: …` | A `rolling-maint` or another balance run is in progress | Normal; wait for the next run |
| `Cloud <name> was not found` (monitor exit 3) | `OS_CLOUD` / `--os-cloud` names a cloud that is not in the `clouds.yaml` in use | Fix the name, or point `OS_CLIENT_CONFIG_FILE` at the right file |
| `rolling-maint`: `command failed rc=255: ssh …` with `Host key verification failed` | The compute's host key is not in the coordinator's `known_hosts`, or it changed (reinstalled host) | Verify the key, then add it (`ssh-keyscan`), or remove the stale one first (`ssh-keygen -R <host>`); or set `StrictHostKeyChecking` in `ssh_config` (section 7.2) |
| `rolling-maint: error: unrecognized arguments: --ssh-strict-host-key` | The option was removed in 1.0.0rc10 | Drop it; host-key policy comes from `ssh_config` |
| `RAM_CRITICAL_THRESHOLD (…) must be above RAM_HOTSPOT_THRESHOLD`, `… must be > 0 and <= 100` | Invalid threshold in the config | Fix the value; nothing runs until it is valid |
| `ansible-inventory not found` | `-i` used outside the Kolla-Ansible venv | Add the Kolla venv to `PATH`, or use `--host-address` |
| Drain exit 2 | Not enough room, or blocked by a server group / AZ / unmovable VM | Read the list of reasons; add capacity or consider `--drain-allow-hotspot` |
| `migration_timeout` in the events | The migration made no progress for 600 s (`MIGRATION_TIMEOUT`), or was still running after 30 min (`MIGRATION_MAX_WAIT`) | Check the migration in Nova; the next run reconciles it automatically |

## 11. Known limitations

- Runs from **one coordinator** only; there is no distributed lock. Do not run in parallel with the legacy version.
- Live migration with shared storage only; block migration is not supported.
- VMs with CPU pinning, PCI passthrough or `resources:*` are not modelled.
- The planner is heuristic: not finding a plan does not mean no other arrangement exists.
- Soft-anti-affinity groups broken by maintenance are re-spread (rc11) only one VM per aggregate per run, only when no host is a hotspot, and only if the CPU MAD and the RAM MAD each stay at or below `TARGET_AVG_MAD` (or, if already above it, grow by at most `SOFT_AA_HEAL_MAD_TOLERANCE`); a group of very large VMs whose move would unbalance the aggregate is reported "deferred: … mad_tolerance". `soft-affinity` groups are never consolidated.
- Validated on a Kolla-Ansible 2026.1 lab with Ceph, 3 computes and 47 VMs. In other environments, repeat the steps in section 5.

Algorithm details: [algorithm.md](algorithm.md). Operations and recovery: [operations.md](operations.md). Changes and validation results: [CHANGELOG](../CHANGELOG.md).
