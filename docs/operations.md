# Operations guide

English | [Tiếng Việt](operations.vi.md)

For operators who already run the tool: where it keeps its state, how it waits and times out, what the planner refuses to do, and how to recover when a run stops. Installation and configuration are in the [user guide](user-guide.md#3-installation); the decision logic is in [algorithm.md](algorithm.md).

Scope: one Linux coordinator with Python 3.12, one cloud (`openstack` by default, see `OS_CLOUD` / `--os-cloud`), Nova/Placement/libvirt and Prometheus. The package/kernel profiles target Ubuntu/Debian. There is no distributed lock.

## 1. Before the first run

- Install v1 into its own venv, never into the legacy script's venv ([user guide section 3](user-guide.md#3-installation)).
- Keep `wbalancer.conf`, `clouds.yaml` and SSH keys where you already manage them; do not put credentials into the install directory.
- SSH host keys follow the coordinator's own OpenSSH configuration (`~/.ssh/config` of the user running `rolling-maint`, then `/etc/ssh/ssh_config`); the tool sets no `StrictHostKeyChecking` itself (since 1.0.0rc10, which also removed `--ssh-strict-host-key`). It runs ssh with `BatchMode=yes`, so with OpenSSH's default policy an unknown or changed host key fails the command (exit 255) and stops the run. Before the rollout, either pre-populate `known_hosts` with the computes' verified keys, e.g. `ssh-keyscan -t ed25519 compute-03 >> ~/.ssh/known_hosts` and compare the fingerprints with the hosts, or set the policy you want in `ssh_config`, e.g. `Host compute-*` / `StrictHostKeyChecking accept-new`. After a compute is reinstalled, remove its old key (`ssh-keygen -R compute-03`) and add the new one.
- Always pass `--prom-url` to `rolling-maint`. It enables the gate that requires every Prometheus target of the host to be up; without it, that gate does not run.
- Start with `--dry-run`. After reviewing the output, drop it and add `-y` for the approved execution.
- **Set the timeouts up front.** Physical hosts can take a long time to boot (firmware, memory checks, storage controllers). Time one reboot by hand and size `--boot-timeout` (default 3600 s) with margin: if a host needs 50 minutes to come back, set `--boot-timeout 5400`. Timeouts are part of the run identity, so they cannot be changed for a pending run. When the boot timeout expires, the run stops with the host still disabled; check the console (BMC/IPMI). The tool never power-cycles a host.

## 2. State files and locks

| Data | Default |
|---|---|
| Maintenance state (checkpoints) | `/var/lib/wbalancer/maintenance` (`rolling-maint --state-dir`) |
| Migration journal | `/var/lib/wbalancer/operations/migration-journal.json` (next to the cooldown file) |
| Cooldown | `/var/lib/wbalancer/vm-cooldown.json` |
| Logs / results | `/var/log/wbalancer/maintenance` (`rolling-maint --log-dir`) |
| Audit events | `/var/log/wbalancer/migrations.jsonl` |
| Coordinator lock | `/var/lock/wbalancer-maint.lock` (`rolling-maint --lock-file`) |
| Balancer lock | `/var/lock/wbalancer-balancer.lock` |
| Config (read only) | `/etc/wbalancer/wbalancer.conf` (`wbalancer --config`, `rolling-maint --wlb-config`; env `WLB_CONFIG_PATH` replaces this default) |

**One coordinator only.** The locks are local `flock` files, so a second coordinator would not see them.

| Process | Lock behaviour |
|---|---|
| `rolling-maint` (apply) | Holds the coordinator lock for the whole run |
| `wbalancer --drain-host` started by `rolling-maint` | Shares its parent's lock (inherited descriptor) |
| `wbalancer` apply started on its own (cron, by hand) | Takes the same coordinator lock, then the balancer lock. If either is held: prints `Not running: …` or "Another … apply run holds the lock" and exits 1 |
| `wbalancer --monitor-only`, `--dry-run` | Take no lock |
| Legacy `wloadbalancer.py` | Does not take this lock: never run legacy apply and v1 at the same time |

- A different `--state-dir` does not allow parallel runs: the lock path is the same.
- Keep the default lock path. If you move `rolling-maint --lock-file`, export the same path as `WLB_COORDINATOR_LOCK` for standalone `wbalancer` runs, otherwise the two no longer exclude each other.
- Do not delete or edit checkpoints or the journal while a run is pending.
- `WLB_CONFIG_PATH=/path/to/wbalancer.conf` (rc12) makes that file the default config of both commands; an explicit `--config` / `--wlb-config` still wins and an empty value counts as unset. Use it for a non-root trial run; the test suite sets it to a temporary path so it never reads the host's root-only `/etc/wbalancer/wbalancer.conf`.

**Upgrading from 1.0.0rc8 or earlier:** those versions used the same layout with a `-v1` suffix (`/var/lib/wbalancer-v1`, `/var/log/wbalancer-v1`, `/var/lock/wbalancer-v1-*.lock`). When `/var/lib/wbalancer-v1` still holds a cooldown file, a migration journal or a maintenance checkpoint and `/var/lib/wbalancer` holds none, every apply run stops before taking a lock and prints what to do; `--dry-run` and `--monitor-only` only warn. Move the state while nothing runs:

```bash
mkdir -p /var/lib/wbalancer && mv /var/lib/wbalancer-v1/* /var/lib/wbalancer/
mkdir -p /var/log/wbalancer && mv /var/log/wbalancer-v1/* /var/log/wbalancer/   # optional: history
```

## 3. Settling after a migration

| Mode | After each migration | Why |
|---|---|---|
| Balance apply (cron) | Waits a full metric window, **360 s** by default (`RUNTIME_SETTLE_SECONDS`), also after migrations started outside the tool | With a 10-minute cron there is almost never anything left to wait for |
| Drain (`--drain-host`, rolling-maint) | **Does not wait.** It still waits for running migrations to finish, then adds the projected load of the VMs that just arrived to each destination | A 28-VM drain with the full wait took 3 h 27 min |

The source host is never reduced. The drain report prints "Load added for VMs that arrived within the metric window …" with the `method` used per host.

### 3.1. RAM: measured directly

Host RAM is `avg_over_time(...[5m])`, so a VM that just arrived shows up only gradually. For each receiving host:

1. The tool waits, at most `max(90 s, 3 × scrape interval)`, until the host's **raw RAM sample** (not averaged) is newer than the moment the last VM finished arriving.
2. Fresh sample: RAM = raw sample (`method=measured`). Error measured on two lab canaries (33 VM arrivals, 504 GiB hosts): at most +0.03 / −0.02 percentage points.
3. No fresh sample in time: RAM = `max(average, raw sample) + host RSS` of the VMs that arrived after that sample (`measured+rss`, errs on the high side). Example: 504 GiB host, average 50.0%, raw sample 51.0%, two VMs of 16 GiB RSS arrived after it → 51.0 + 32/504 × 100 = 57.3%.
4. Raw sample unreadable: model `RSS × (1 − t/300)`, where 300 s is the RAM query window and t the seconds since arrival (`method=model`).
5. Always the **RSS on the host**, never the RAM the guest reports. Example: a VM reporting 5.95 GiB used inside the guest held 31.77 GiB RSS on the host.

`PROMETHEUS_MEM_USED_INSTANT` and `PROMETHEUS_MEM_SAMPLE_TIMESTAMP` (the raw-sample queries) are built from the node job by default.

### 3.2. CPU: short-term model

Host CPU is `irate`, which catches up after 2 scrapes. The added CPU decays linearly over a window of 2 × the node job's scrape interval, read from Prometheus `/api/v1/targets` (30 s scrape → 60 s window). If the interval cannot be read, 60 s is used; force a value with `COMPENSATION_CPU_WINDOW_SECONDS`.

Example: a 4-vCPU VM at 50% (2 cores) arrives on a 64-core host → +3.1 points at arrival, +1.6 points 30 s later, 0 after 60 s.

The VM's CPU comes from its own metrics; if the VM is visible on 2 hosts, the sample taken before the migration is used; failing that, the full flavor.

## 4. Migration timeout

- Stop waiting once Nova reported **no progress for `MIGRATION_TIMEOUT` = 600 s**, counted from the start or from the last progress. Progress means a change in the memory/disk counters, status or `updated_at` of the migration itself, or of the migration running ahead of it on the same source host (with `max_concurrent_live_migrations`, the next VM sits in `queued`).
- Absolute cap `MIGRATION_MAX_WAIT` = 1800 s (30 minutes). If a large VM legitimately needs longer, raise `MIGRATION_MAX_WAIT` (example: a 256 GiB VM at an effective 150 MB/s needs about 30 min).
- Both are config keys (`wbalancer.conf`), validated at start-up (`MIGRATION_MAX_WAIT` ≥ `MIGRATION_TIMEOUT`). The rc8 key `MIGRATION_STALL_TIMEOUT` is obsolete: it is ignored with a warning.
- The same rule applies to the live-migration monitor, to waiting on the migration journal, and to waiting on migrations started outside the tool.

Examples: frozen from the start → stops at minute 10; progress until minute 12, then frozen → stops at minute 22; still progressing at minute 30 → stops at the cap. On the test lab Nova updated progress about every 6 s while copying memory; the longest silent gap measured was 34 s. Nova itself aborts a migration after `live_migration_completion_timeout` (800 s per GiB by default).

**On timeout:** the tool does **not** abort the migration in Nova. It writes a `migration_timeout` event, puts the VM in cooldown, keeps the journal entry pending and submits nothing more. The next run reconciles it automatically once the migration ends (section 10).

## 5. Metric freshness and the settle window

- Every snapshot re-checks the age of the raw samples (at most 120 s old) and re-validates exactly the move that was approved.
- `RUNTIME_SETTLE_SECONDS` is an environment variable, not a config key. It must cover the largest PromQL window of any configured query (including `offset` and compound durations such as `10m30s`) plus at least 60 s; otherwise apply refuses to start. Example: a custom query using `[5m] offset 1m` has a 360 s window, so `RUNTIME_SETTLE_SECONDS` must be ≥ 420.
- Custom queries must still use the raw exporter series the freshness guard checks: `node_cpu_seconds_total`, `node_memory_MemAvailable_bytes`, `node_memory_MemTotal_bytes`, `libvirt_domain_info_cpu_time_seconds_total`, `libvirt_domain_info_virtual_cpus`, `libvirt_domain_stat_memory_rss_bytes` (and `libvirt_domain_stat_memory_usable_bytes` / `_available_bytes` when exposed).
- `PROMETHEUS_QUERY_URL` is required: without it the tool raises a configuration error; there is no default pointing at a public host.
- Numeric config values must be finite: `nan` and `inf` are rejected.

## 6. Sizing the drain timeout

The default `--drain-timeout` of 7200 s covers about 100 VMs at the drain pace (no wait between migrations). Measured on the lab: 1364 s for 19 VMs and 1335 s for 19 VMs, i.e. ~70 s per VM.

For a larger host, raise the timeout **before** starting, because settings are part of the checkpoint identity. Example: 150 VMs × 70 s = 10 500 s; with 20% margin, `--drain-timeout 12600`.

## 7. Planner constraints

- A VM that is **locked**, has a Nova **task** running, or whose flavor needs resources the planner does not model (`hw:cpu_policy=dedicated/mixed`, `pci_passthrough:*`, `resources:*`) is never chosen for balance. During a drain these VMs are reported `unmovable` with reason `locked`, `task_in_progress` (temporary; the drain waits) or `unsupported_resources`.
- `rolling-maint` **stops before disabling the host** if the host still runs locked ACTIVE VMs, the same way it handles VMs in ERROR. The `--locked` list from the CLI is only a hint (python-openstackclient 6.6.0 ignores the filter and returns every VM), so each VM is confirmed with `server show` at microversion 2.9.
- Server groups follow Nova's filter semantics: only the destination host is checked. `anti-affinity` honours `max_server_per_host`. An `affinity` group that is already split may go to any host that currently holds a member.
- `soft-anti-affinity` / `soft-affinity`: balance never makes them worse. Since rc11 a `soft-anti-affinity` group that is already broken is also re-spread, one VM per run (7.1). Drain prefers the destination with the fewest violations, then MAD (rc11: then the fewest soft-anti-affinity members); unavoidable violations are listed under "Soft (anti-)affinity cannot be fully kept".
- A VM only moves within its availability zone (read from the `os-availability-zone` detail). Consequence: **every cross-AZ move is blocked, even for a VM that was booted without an AZ**, because Nova reports the zone a VM currently runs in for every VM. This is more conservative than Nova itself, whose scheduler may place an unpinned VM in any zone. Move such a VM across zones by hand if you need to.
- The destination must not exceed the CPU/RAM hotspot threshold after the move, even if it was already above it. MAKE_ROOM never parks the buffer VM on the hot source host itself.
- A host with missing metrics/Placement data, or a Prometheus target without the `alias` label, fails only the aggregate that contains it. The other aggregates still run; the final exit code is still non-zero.

The planner is heuristic: not finding a plan does not prove that no other arrangement exists. The runtime projection does not guarantee that a guest will not increase its load after submission. Placement remains the source of scheduler capacity. Do not lower thresholds or the settle time just to get past a guard; keep server-group checks enabled when applying.

`PLANNER_CANDIDATE_BUDGET` (config, default 100 000 candidate evaluations per search) bounds the search. Exceeding it returns `budget_exhausted`, not "infeasible", and a plan that was not fully evaluated is never applied. A drain does not also run the direct/make-room planner. Benchmark on a 3-host / 47-VM lab (1.0.0rc13, every HTTP request counted): one cold snapshot (`--monitor-only`) made 31 OpenStack GET requests, one authentication POST and 7 Prometheus queries; the planner itself makes no API calls. This is a measurement, not a latency promise for every cluster.

### 7.1. Soft-anti-affinity heal (rc11)

When a `soft-anti-affinity` group has 2+ members on one host of an aggregate and no host is a hotspot, a balance run moves one member to a host without a member instead of making its normal balancing move ([algorithm, section 8](algorithm.md#8-soft-anti-affinity-heal-rc11)). Example dry-run (worked example of the algorithm guide):

```text
WARNING - soft-anti-affinity spread broken (1 violation(s) in 1 group(s)) and can be healed automatically (next: group db)
...
Proposed plan: SOFT_AA_HEAL | db2 (A -> B)
Load status: UNBALANCED (WARNING)
Soft-anti-affinity: 1 broken group(s), 1 violation(s)
  - heal db: db2 (A -> B), group violations 1 -> 0, weighted MAD 10.13 -> 8.57

INFO: Aggregate prod-agg: soft_aa_heal: group db (1 violation(s)) -> move db2 A -> B (weighted MAD 10.133 -> 8.572)
[Round 1] [SOFT_AA_HEAL] Plan=SOFT_AA_HEAL steps=1
   Step 1: db2 (A -> B) reason=soft_aa_heal
   ...
   Soft-anti-affinity heal: group db violations 1 -> 0
   Heal MAD gate: CPU 0.22 <= 3.00, RAM 10.66 <= 12.27 (per resource: TARGET_AVG_MAD 3, or before + 0.05 when already above it)
```

A group that can never be fixed in this aggregate (here 4 members on 3 hosts) is `unresolvable`; it does not change the exit code:

```text
OK - cluster balanced
...
Soft-anti-affinity: 1 broken group(s), 1 violation(s)
  - etcd (1 violation(s)): unresolvable: every host of the aggregate already holds a member (3 host(s), 4 member(s))

INFO: Aggregate prod-agg: soft_aa_heal: group etcd (1 violation(s)) unresolvable: every host of the aggregate already holds a member (3 host(s), 4 member(s))
```

A group that only cannot be healed in this run is `deferred`, with the blocking counts. Example right after a manual move, while the moved VM's metrics are still on two hosts (about 5 minutes) and the other member is in cooldown: `deferred: no movable member on the crowded host(s) (cooldown=1, missing_vm_details=1)`. Other deferred reasons: `no member-free destination passes the filters (would_create_new_hotspot=1, mad_tolerance=1)`, `planner candidate budget exhausted`. `mad_tolerance` means the move would push the CPU MAD or the RAM MAD above `TARGET_AVG_MAD` (making the aggregate `UNBALANCED`), or would grow a MAD that was already above it by more than `SOFT_AA_HEAL_MAD_TOLERANCE`. On a balanced aggregate this only happens for members large enough to unbalance it (e.g. 32 GiB guests on 256 GiB hosts: RAM MAD 0 → 8.33 > 3).

Audit events (`/var/log/wbalancer/migrations.jsonl`):

| Event `result` | When | Heal fields |
|---|---|---|
| `dry_run`, `migration_success`, `migration_failed`, `migration_timeout`, `step_revalidation_failed`, … | The heal move goes through the normal execution path (journal, revalidation, cooldowns, email) | `"plan_type": "soft_aa_heal"`, `"mode": "soft_aa_heal"`, `"step_reason": "soft_aa_heal"`, `"soft_aa_heal": {"group_name": "db", "group_violations_before": 1, "group_violations_after": 0, "violation_reduction": 1, …}` |
| `soft_aa_heal_unresolvable` | A balance run (apply or dry-run) found a broken group that can never be fixed in this aggregate (every host holds a member) | `"mode": "soft_aa_heal"`, `"detail": "etcd: unresolvable: …"`, `"soft_aa_unresolvable": [...]`, `"soft_aa_violations": 1` |
| `soft_aa_heal_deferred` | A balance run found a broken group it cannot heal this time (cooldown, lock, metrics after a move, filters, MAD gate, budget) | `"mode": "soft_aa_heal"`, `"detail": "db: deferred: …"`, `"soft_aa_deferred": [{…, "blocking": {"cooldown": 1, "missing_vm_details": 1}}]` |

- Every event carries the labels of the snapshot its plan was **made** from. A heal's `migration_success` reads `"mode": "soft_aa_heal"`, `"detail_status": "SOFT_AA_HEAL"`, `"severity": "WARNING"`, like its `dry_run` event; a direct move carries its load mode (`"mode": "mad"`, …). Drain events carry `"mode": "drain"`, `"detail_status": "DRAIN_FEASIBLE"` / `"DRAIN_INFEASIBLE"` and `"severity": null`: a drain is not a balance check, its outcome is in `result`. (Up to rc10 every `migration_success` and drain event said `CRITICAL` / `UNBALANCED_NO_VALID_MOVE`, copied from the plan-less refresh before the submit.)
- A refused heal step names its reason, e.g. `soft_aa_heal_no_longer_heals` (the group was repaired meanwhile) or `refreshed RAM MAD 3.120 above heal limit 3.000`, and gets the usual 30-minute rejected-move cooldown.
- `soft_aa_heal_unresolvable` / `soft_aa_heal_deferred` are written once per aggregate per balance run while the condition lasts (with a 10-minute cron, up to 144 lines a day for a group that stays unresolvable; a deferred group usually clears within a few runs). `--monitor-only` writes no event.
- All heal events: `jq -c 'select(.mode == "soft_aa_heal")' /var/log/wbalancer/migrations.jsonl`.

To turn it off, set `HEAL_SOFT_ANTI_AFFINITY=false` in `wbalancer.conf` (or the environment of the cron/unit); the next run behaves exactly like rc10, tie-breaks included, and still lists broken groups as `heal disabled`. To keep one VM out of the heal (and out of balancing), put it in the exclude file (`EXCLUDE_VMS_FILE`); the heal then tries the next member. A pending heal migration is never cancelled by disabling the feature: the journal reconciles it as usual.

## 8. The `kernel-update` profile

- Runs `dist-upgrade` of the **whole approved package plan**, not only the kernel.
- Refresh the APT metadata before the run: the profile requires the APT update success stamp to be less than 24 hours old.
- A simulation that removes packages is refused.
- The plan is captured in `pre_drain`, before the host is disabled.
- Before upgrading, the remaining work must be a **subset, by package name,** of the approved plan. Accepted: apt-daily bumped a version, or unattended-upgrades already installed part of the plan. Refused: a new package name, or any removal. Example: plan `{linux-image-…, openssl, libssl3}`; unattended-upgrades already installed `openssl` → remaining `{linux-image-…, libssl3}` is accepted; if `curl` shows up → refused.
- If the plan is already fully installed, `dist-upgrade.sh` returns 0.
- Reboots when `/var/run/reboot-required` exists **or** when `/boot` holds a newer kernel than the running one (hosts without the update-notifier hook).
- Then checks dpkg/initrd/modules/GRUB, stores the expected kernel and verifies the actual kernel after boot. A custom GRUB saved/next entry is refused.
- Only "no update / no new kernel" leads to SKIP; apt/dpkg errors must fail.

## 9. Hooks: schema and exit codes

How to write a profile: [profiles.md](profiles.md). Hook keys:

| Key | Default | Meaning |
|---|---|---|
| `id` | `<stage>[index]` | Stable ID; resume state is kept per `<stage>/<id>` |
| `name` | first 60 characters of the command | Label in the log |
| `script` or `cmd` | required | A script file inside the profile directory, or an inline command (not both) |
| `check` or `check_cmd` | none | Recovery check; required for `idempotency: reconcile` |
| `run` | `ssh` | `ssh` = on the compute, `local` = on the coordinator (bash) |
| `timeout` | 600 s | Expiry leaves the outcome UNKNOWN |
| `idempotency` | `never` | `never`, `idempotent` or `reconcile` |
| `capture` | none | `expected_kernel` or `package_plan` only |
| `retries` / `retry_delay` | 0 / 10 s | Retries after a non-zero exit, for `idempotent` hooks only |

Exit codes:

| Command | rc 0 | rc 1 | rc 3 | rc 10 | Other |
|---|---|---|---|---|---|
| Hook | Success | Failure | Failure; in `pre_drain`: no work needed, skip the host | Failure | Failure |
| Recovery `check` | Completed and verified: marked done, not re-run | UNKNOWN | UNKNOWN | Proven safe to run again: the hook is re-run | UNKNOWN |
| `reboot_required_cmd` | Reboot needed | Not needed | UNKNOWN | UNKNOWN | UNKNOWN |

UNKNOWN stops the run.

What happens to a hook on resume:

| Hook state from the previous run | On resume |
|---|---|
| `done` | Skipped |
| Pending or failed, with a `check` | The check decides (table above) |
| Failed, no check, `idempotency: idempotent` | Re-run |
| Anything else without a check (including a timeout or SSH exit 255) | STOPPED: the outcome is UNKNOWN and must be reconciled by hand |

Do not mark a hook `idempotent` casually. The `{host}`, `{address}` and `{profile}` placeholders are quoted as shell tokens: do not add quotes around them.

## 10. When a run is STOPPED

1. Read the state, logs, results and the real Nova state. Keep the host disabled until it is proven safe.
2. Keep the tool version, profile and config unchanged and **re-run the same command**: it resumes at the pending stage by itself. `--resume-from` cannot be used to skip a guard.
3. **Pending migration:** the migration journal reconciles itself on the next run using `/os-migrations?instance_uuid=`, considering only rows with an id greater than the highest id recorded at submit time.

   | Evidence in Nova | Journal outcome |
   |---|---|
   | Rows `error`/`failed`/`cancelled` and the VM is on the source | `failed` |
   | VM ACTIVE on the destination | `success` |
   | No row, no `task_state`, VM ACTIVE on the source 120 s after submit | `not_started`. Nova's API sets `task_state=migrating` before casting to the conductor, so this proves the migration never ran |
   | VM deleted | `vm_deleted` |
   | Migration still running | Wait, under the rule of section 4 |
   | Cannot be proven (e.g. VM on a third host) | Stays UNKNOWN: no new migration is submitted |

   The event, email and cooldown of a failed migration are written **before** reconciliation. A VM left in ERROR on the source after a failed migration closes the journal as `failed`; rolling-maint then stops at its ERROR-VM check with instructions.

   For an UNKNOWN outcome, compare Nova's migration records, instance actions, server host and task state yourself; never reset ERROR or resubmit blindly. Before any manual reset: save the checkpoint and journal; confirm the VM, operation ID, source/destination, the migration rows created after the submit and their terminal status, that no task or job is left, that the source domain is really running and no domain exists on the destination, and that the guest is reachable. Reset the VM state (`openstack server set --state active`) only after proving the guest is still running, and make sure no wbalancer/rolling-maint run can start meanwhile. Before retrying a failed migration, compare the Nova network info cache, the Neutron port and the libvirt domain XML on source and destination; do not reboot the guest or edit the port to hide a failure you have not explained.
4. **Reboot intent recorded but the boot ID has not changed:** check the transient reboot job (systemd unit `compute-maintenance-reboot`) and the host; do not delete the intent to force another reboot. Resume only waits for a new boot ID.
5. **Identity mismatch or legacy checkpoint:** reconcile by hand; never edit the hashes in the state. Finish the run with the tool version that created the checkpoint before upgrading.

Other cases:

- A hook timeout or SSH exit 255 leaves the hook pending as UNKNOWN. Finished hooks are not replayed; a hook with an unknown outcome needs a check to verify it.
- A `post_restore` failure puts the host back to disabled, owned by the run (quarantine); resume restarts at the health gate, checks the inventory, and does not drain or reboot workload that was scheduled while the host was enabled.
- The health gate must PASS continuously, on the same boot ID, with all expected containers, network agents, Cinder services and Prometheus targets.

## 11. Rollback

- Rolling back the tool does not undo migrations, package upgrades or reboots.
- Change the installed version only when no run is pending; keep the previous version and its state for recovery.
- Never enable a host by hand to get past the ownership guard: from `drain` to `restore` the host must stay disabled with the reason `rolling-maint v1:<run_id>` of this run.

## 12. Package upgrade recovery scope

- The upgrade check returns rc 10 (safe to retry) only when dpkg is clean, nothing is removed, and every remaining package name was in the approved plan; rc 0 when nothing is left to install. New package names stay UNKNOWN.
- The drain/reboot canary does not validate a live package upgrade.
- Before rebooting, rolling-maint also refuses when the host still runs QEMU guest processes although Nova lists no running VM.
- Before an approved upgrade window, record and pause the automatic apt timers/unattended-upgrades following your host runbook, verify that no other apt/dpkg process is active, and restore the previous schedule afterwards.
- Never edit the checkpoint `package_plan` to bypass the guard.

Validation results per release: [CHANGELOG](../CHANGELOG.md).
