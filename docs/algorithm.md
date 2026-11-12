# How wbalancer decides

English | [Tiếng Việt](algorithm.vi.md)

This document explains how `wbalancer` picks a live migration, how it plans a drain, and why it sometimes refuses to move anything. Every threshold and formula below is taken from the v1 source; file links point to where each rule lives. Each idea comes with a short numeric example.

Not here: installation and commands ([user guide](user-guide.md)), state files, locks and recovery ([operations](operations.md)), switching from the old script ([migration guide](migrating-from-wloadbalancer.md)).

## Changed from `wloadbalancer.py`

The core idea is the same: a hotspot comes first, then pressure, then weighted-MAD balancing; CPU/RAM weights adapt to the situation; the planner picks the single best move and only executes it if it improves things enough. The details changed:

| Item | `wloadbalancer.py` | v1 |
|---|---|---|
| RAM hotspot | > 75% | > 80% |
| RAM critical hotspot | > 85% | > 95% |
| CPU hotspot | > 85% | > 70% |
| Minimum improvement, direct move | 0.05 | 0.15 |
| Two-step "make room" plan | none | MAKE_ROOM, needs ≥ 0.30 |
| Weight floor 0.2 on a 1:9 signal split | 0.182 / 0.818 (floor, then renormalise) | 0.20 / 0.80 (true floor) |
| Scheduler capacity | Nova hypervisor stats × allocation ratios from openstack-exporter (fallback 7.0 / 1.2, 24 GiB RAM reserve hard-coded) | Placement inventory and usage, minus 5% headroom |
| Runtime denominator (how many % a VM is worth) | Nova hypervisor `vcpus`; RAM = over-committed RAM − 24 GiB | Real logical CPUs and `MemTotal` from node_exporter |
| "Destination must not become a hotspot" | hotspot modes only | every mode, MAKE_ROOM and drain too |
| Server groups | whole group judged; lookup error → continue without groups | destination only (as Nova does), `max_server_per_host`, soft policies; lookup error → stop |
| Host without metrics | left out with a warning | aggregate refused (monitor: UNKNOWN) |
| Host whose nova-compute is not ready | not a destination, still counted in MAD | removed from the calculation entirely |
| Skipped VMs | cooldown only | + exclude list, not ACTIVE, locked, running task, CPU pinning / PCI / `resources:*`, incomplete metrics or flavor |
| Availability zones, max instances per host | not checked | enforced |
| Cooldown after success / failure | 4 h / 4 h in the last script (its README still said 30 / 60 min) | 4 h / 4 h, plus 30 min after a move is refused at revalidation |
| Migration timeout | fixed 600 s | stops after 600 s without progress (from the start or the last progress), 30 min maximum (both configurable) |
| Drain a host | none | `--drain-host`, fail-closed |
| Broken soft-anti-affinity group (2+ members on one host) | never repaired | re-spread one move per run when no host is a hotspot (rc11, section 8) |
| Changing thresholds | edit the script | config keys (rc7), e.g. `RAM_HOTSPOT_THRESHOLD=75`, `RAM_CRITICAL_THRESHOLD=85`, `CPU_HOTSPOT_THRESHOLD=85`, `MIN_IMPROVEMENT_THRESHOLD=0.05` restores the old values |

## 1. One run, one aggregate

```text
snapshot (Nova + Placement + Prometheus)
  -> mode (hotspot > pressure > MAD) -> CPU/RAM weights
  -> best direct move (hotspot host first, then all hosts)
  -> if no direct move reaches 0.15: best MAKE_ROOM plan
  -> accept if improvement >= threshold -> status + Icinga severity
  -> no hotspot + broken soft-anti-affinity group: one heal move replaces the plan (section 8)
  -> execute: revalidate on fresh data, migrate, record cooldown
```

- Each aggregate gets **at most one plan per run**: one direct migration, one two-step MAKE_ROOM plan, or one soft-anti-affinity heal move ([application/balancing.py](../src/compute_maintenance/application/balancing.py)).
- After a migration the cluster data is fetched again before the next aggregate is evaluated, because aggregates can share hosts.
- A data problem in one aggregate (missing metric, missing Placement provider) skips that aggregate only; the run still exits 1.
- The planner examines at most `PLANNER_CANDIDATE_BUDGET` (default 100 000) VM × host pairs per aggregate. When the budget runs out the aggregate fails with "feasibility unknown" instead of returning a half-searched answer.

## 2. What the planner sees

### 2.1 Host load

Built in [application/snapshots.py](../src/compute_maintenance/application/snapshots.py) from the queries in [config.py](../src/compute_maintenance/config.py):

| Value | Source |
|---|---|
| Host CPU % | `1 − idle / all modes` of `node_cpu_seconds_total`, `irate` (last two scrapes) |
| Host RAM % | `100 − MemAvailable / MemTotal`, 5-minute `avg_over_time` |
| Logical CPUs, RAM size | count of idle CPU series, `MemTotal` (runtime denominators) |

Fail-closed rules:

- Any host of the aggregate without CPU or RAM metrics, or with a value outside −1…101%, makes the **whole aggregate** fail. The old script only logged a warning and dropped the host, so the MAD was computed on a partial picture.
- Fewer than 2 hosts with metrics: the aggregate is skipped.
- Balance and monitor remove hosts whose nova-compute is not enabled + up + not forced down. Their load no longer feeds the mean, MAD, mode, peak or severity. Their VMs stay in the topology, so server-group checks still see them.
- In apply mode every raw sample (host CPU, MemAvailable, MemTotal, VM CPU, VM RSS, vCPU count) must be at most 120 s old ([adapters/freshness.py](../src/compute_maintenance/adapters/freshness.py)).

### 2.2 VM load

From the libvirt exporter ([adapters/prometheus.py](../src/compute_maintenance/adapters/prometheus.py)):

| Value | Formula |
|---|---|
| VM CPU % | `irate(cpu_time) / virtual_cpus × 100`, clamped to 100 (raw values up to 1000% are accepted: the cgroup also counts emulator and vhost threads) |
| VM RAM % | `100 − usable / available` from the guest balloon driver, 5-minute average |
| Real cores moved | `vcpus × cpu% / 100` |
| Real RAM moved | `flavor_ram × ram% / 100`, or host-side RSS |

RSS (`libvirt_domain_stat_memory_rss_bytes`) replaces the guest value in two cases: for every VM with `--use-vm-ram-rss`, and automatically for a guest without balloon stats (FreeBSD appliances, Windows without the balloon service). RSS is at least what the guest uses, so the estimate errs on the safe side.

A VM is not a candidate when its series appear on two hosts at once (right after a live migration), or when Nova and Prometheus disagree on its host.

### 2.3 Scheduler capacity comes from Placement

[adapters/openstack.py](../src/compute_maintenance/adapters/openstack.py) reads `VCPU` and `MEMORY_MB` inventory and usage of each host's resource provider; [domain/policies.py](../src/compute_maintenance/domain/policies.py) turns them into capacity:

```text
nominal_capacity = (total - reserved) × allocation_ratio
policy_capacity  = nominal_capacity × (1 - headroom / 100)        headroom = 5% by default
free             = policy_capacity - used
fits             = flavor_vcpus <= free_VCPU  and  flavor_ram_mb <= free_MEMORY_MB
```

Example, `VCPU`: 64 pCPU, ratio 4.0, used 236.

```text
nominal = 64 × 4.0 = 256      policy = 256 × 0.95 = 243.2      free = 243.2 - 236 = 7.2
4-vCPU flavor fits, 8-vCPU flavor does not (reject reason placement_vcpu)
```

- The flavor is used for feasibility; observed usage is used for scoring. A VM using 10% of 32 GiB still needs 32 GiB free in Placement.
- `min_unit`, `max_unit` and `step_size` of the inventory are honoured.
- No openstack-exporter is needed any more. The old keys `PROMETHEUS_CPU_RATIO`, `PROMETHEUS_MEM_RATIO` and `PROMETHEUS_OPENSTACK_EXPORTER_JOB` are ignored with a warning.
- Hosts with allocation ≥ 90% of nominal capacity are counted on the `Placement:` line of the output. This is information only.
- `PLACEMENT_FINAL_ALLOCATION_CANDIDATE_CHECK=true` (default off) additionally asks `GET /allocation_candidates` for the final direct candidate.

### 2.4 Projecting a move

[domain/balance/filters.py](../src/compute_maintenance/domain/balance/filters.py) `project_runtime_move`:

```text
cpu_impact = real_cores / logical_cpus(host) × 100
ram_impact = real_ram_gb / MemTotal_gb(host) × 100
source -= impact(source)        destination += impact(destination)
```

Example: VM 8 vCPU at 50% (4 cores), 32 GiB at 50% (16 GiB). Source 64 CPUs / 512 GiB, destination 32 CPUs / 256 GiB.

| | CPU | RAM |
|---|---|---|
| Source | −4/64 = −6.25 points | −16/512 = −3.13 points |
| Destination | +4/32 = +12.50 points | +16/256 = +6.25 points |

The old script divided RAM by the over-committed capacity minus 24 GiB: 512 × 1.2 − 24 = 590.4 GiB, so the same VM counted as 2.71 points instead of 3.13.

## 3. Choosing the mode

[domain/balance/modes.py](../src/compute_maintenance/domain/balance/modes.py) `choose_effective_mode`, checked in this order:

| Order | Mode | Condition | CPU / RAM weight |
|---|---|---|---|
| 1 | `ram_critical_hotspot` | any host RAM > 95% | 0.1 / 0.9 fixed |
| 2 | `ram_hotspot` | any host RAM > 80% | 0.2 / 0.8 fixed |
| 3 | `cpu_hotspot` | any host CPU > 70% | 0.8 / 0.2 fixed |
| 4 | `pressure` | aggregate mean CPU or RAM > 80% | adaptive, from MAD and mean |
| 5 | `mad` | otherwise | adaptive, from MAD |

- The critical mode changes only the weights; its status names are the same as `ram_hotspot`.
- **Pressure mode cannot trigger with the v1 defaults.** A mean above 80% means at least one host above 80%, which is already a RAM hotspot (> 80%) or a CPU hotspot (> 70%). For example CPU 84/82/80 (mean 82) was pressure mode in the old script; in v1 it is `cpu_hotspot`. Since rc7 the thresholds are config keys: set `PRESSURE_THRESHOLD` below `RAM_HOTSPOT_THRESHOLD` or `CPU_HOTSPOT_THRESHOLD` to make pressure mode reachable. Example: `PRESSURE_THRESHOLD=65` with CPU 68/69/67 (mean 68, no host above 70) gives `pressure`. The run logs an INFO line when pressure mode cannot trigger.
- **Heal rung (rc11):** between `cpu_hotspot` and `pressure`/`mad` sits the soft-anti-affinity heal (section 8). It is not a mode: it runs only when no host is a hotspot, keeps the weights of `pressure`/`mad`, and falls through to them when there is nothing it can fix.
- **Hotspot-first search:** in a hotspot mode the planner first tries VMs on the hottest host (highest RAM for RAM modes, highest CPU for `cpu_hotspot`). If none of those moves reaches the threshold it searches all hosts and keeps whichever result improves more. The output shows this as `HOTSPOT-FIRST (host)`, `FALLBACK-FULL (after host)` or `FULL`.

## 4. Scoring: MAD and adaptive weights

### Example A. Mean absolute deviation (MAD)

Three hosts, CPU 60/40/35%, RAM 68/42/38%.

```text
cpu_mean = (60 + 40 + 35) / 3 = 45.00
cpu_mad  = (|60-45| + |40-45| + |35-45|) / 3 = (15 + 5 + 10) / 3 = 10.00
ram_mean = (68 + 42 + 38) / 3 = 49.33
ram_mad  = (18.67 + 7.33 + 11.33) / 3 = 12.44
```

Higher MAD means more uneven. No host is above 70% CPU or 80% RAM, so this is `mad` mode.

### Example B. Adaptive weights with a true floor

In `mad` mode the signals are the MADs themselves; in `pressure` mode they are `0.7 × MAD + 0.3 × mean`. The share of each signal becomes its weight, clamped so that neither side drops below 0.2:

```text
cpu_weight = min(max(cpu_signal / (cpu_signal + ram_signal), 0.2), 1 - 0.2)
ram_weight = 1 - cpu_weight                (both signals 0 -> 0.5 / 0.5)
```

From Example A:

```text
cpu_weight = 10.00 / (10.00 + 12.44) = 0.446      ram_weight = 0.554
weighted_mad = 10.00 × 0.446 + 12.44 × 0.554 = 11.36
```

Floor example, `cpu_mad = 1`, `ram_mad = 9`: the CPU share is 0.10, so v1 gives exactly **0.20 / 0.80**. The old script took `max(share, 0.2)` and renormalised, which turned the floor into 0.182 / 0.818.

Pressure example (only relevant if thresholds are changed): `cpu_mean 90`, `ram_mean 60`, `cpu_mad 4`, `ram_mad 8`:

```text
cpu_signal = 4 × 0.7 + 90 × 0.3 = 29.8      ram_signal = 8 × 0.7 + 60 × 0.3 = 23.6
cpu_weight = 29.8 / 53.4 = 0.558             ram_weight = 0.442
```

MAD alone would favour RAM (8 > 4); adding the mean lets the hotter resource win. The 0.7 / 0.3 split keeps balancing the main goal.

### Example C. Improvement of one move

```text
improvement = baseline_weighted_mad - weighted_mad_after_move       (weights stay fixed)
```

Move a VM worth 12 CPU points and 12 RAM points from host A to host B (equal hosts):

```text
CPU 60/40/35 -> 48/52/35      cpu_mad 10.00 -> 6.67
RAM 68/42/38 -> 56/54/38      ram_mad 12.44 -> 7.56
after = 6.67 × 0.446 + 7.56 × 0.554 = 7.16
improvement = 11.36 - 7.16 = 4.20
```

The planner simulates every movable VM to every other host, keeps the largest positive improvement, and accepts it only if it is **≥ 0.15** (`MIN_IMPROVEMENT_THRESHOLD`). One combined score lets it compare a move that helps CPU a lot but hurts RAM against one that helps both a little.

Target MAD 3 is used only to label the status (`UNBALANCED` vs `BALANCED`). It does not gate execution: a move that clears 0.15 on an aggregate with MAD ≤ 3 (`BALANCED_OPTIMIZABLE`) is still executed.

## 5. Filters: when a move is not allowed

Rejections are counted per reason and printed as `Rejected candidate pairs: …`.

### 5.1 VM filters

| Skip reason | Meaning | Drain |
|---|---|---|
| `excluded` | Matches `EXCLUDE_VMS_FILE` / `--exclude-file` | ignored |
| `cooldown` | Moved or failed in the last 4 h, or refused at revalidation in the last 30 min | ignored |
| `missing_vm_details` | No usable metrics, series on two hosts, or Nova/Prometheus host mismatch | unmovable (transient) |
| `not_active` | Nova status is not `ACTIVE` | unmovable |
| `incomplete_metadata` / `incomplete_metrics` | Flavor vCPU/RAM unknown, or CPU/RAM metric missing | unmovable (transient) |
| `locked` | VM is locked; Nova refuses the migration | unmovable |
| `task_in_progress` | Nova `task_state` is set (rebuild, snapshot, resize…) | unmovable (transient) |
| `unsupported_resources` | Flavor has `hw:cpu_policy=dedicated` or `mixed`, `pci_passthrough:*` or `resources:*`; these allocate PCPU / PCI / custom classes that the VCPU/MEMORY_MB model cannot check | unmovable |

### 5.2 Destination filters, in order

| Reject reason | Rule |
|---|---|
| `destination_not_ready` | nova-compute must be enabled, up, not forced down |
| `max_instances` | `servers on host + 1 ≤ limit` from `MAX_INSTANCES_PER_HOST` (per-host JSON) or `MAX_INSTANCES_PER_HOST_DEFAULT`; 0 = unlimited; every Nova server on the host counts |
| `server_group` | hard affinity / anti-affinity (5.3) |
| `soft_server_group` | the move makes a soft group worse (5.3) |
| `availability_zone` | VM zone (Nova `OS-EXT-AZ`, else the source host's zone) must equal the destination zone; unknown destination zone → refused. Consequence: **no move ever crosses an AZ**, not even for a VM that was booted without an AZ. Nova reports `OS-EXT-AZ` (the zone the VM runs in now) for every VM, so such a VM stays in its current zone. This is more conservative than Nova, whose scheduler may place an unpinned VM in any zone |
| `placement_vcpu`, `placement_memory`, `placement_*_min_unit`… | flavor does not fit Placement (2.3) |
| `runtime_capacity_missing` | no logical CPU / MemTotal for a host |
| `would_create_new_hotspot` | destination would end above CPU 70% or RAM 80% (5.4) |
| `hotspot_guardrail` | hotspot modes only (5.4) |

### 5.3 Server groups as Nova applies them

v1 copies Nova's `ServerGroupAffinityFilter` / `ServerGroupAntiAffinityFilter`: only the **destination** is judged ([domain/policies.py](../src/compute_maintenance/domain/policies.py)).

| Policy | Move allowed when |
|---|---|
| `affinity` | the destination already hosts another member, or no other member is placed |
| `anti-affinity` | other members on the destination < `rules.max_server_per_host` (default 1) |
| `soft-anti-affinity` | penalty = other members on the host; balance refuses a move that raises it (gatekeeper). Since rc11 a group that is already broken is also actively re-spread (section 8) |
| `soft-affinity` | penalty = 1 when other members exist but none is on the host; same rule (gatekeeper only, never consolidated) |

Example: anti-affinity group with vm1 on A and vm2, vm3 both on B (an existing violation).

- vm1 → C: C has 0 other members → **allowed**. The old script judged the whole group, saw two members on B and refused every move of every member.
- vm1 → B: B has 2 others ≥ 1 → refused. With `max_server_per_host=3` it would be allowed (2 < 3).

Soft policies matter because Nova does not enforce them when a live migration names an explicit target host. Drain does not refuse on soft policies; it prefers destinations with the lowest penalty and lists violations as warnings (section 9). Server groups are read for all projects; if that read fails the run stops (the old script continued without groups). `--no-server-groups` turns hard and soft checks off.

Soft-anti-affinity is handled in two ways since rc11:

- **Gatekeeper** (unchanged): no balancing or MAKE_ROOM move may raise a soft group's penalty, so balancing never breaks a group further.
- **Repair** (new): a group that is already broken, for example by a drain or a failover that had no other room, is re-spread by the heal rung, one move per run (section 8). Up to rc10 such a group stayed broken unless a MAD move happened to split it.

### 5.4 Hotspot rules

Two checks in [domain/balance/filters.py](../src/compute_maintenance/domain/balance/filters.py) (`destination_hot_after`, `is_hotspot_guardrail_satisfied`):

| Check | Applies in | Rule |
|---|---|---|
| No hot destination | every mode, MAKE_ROOM, balance revalidation | destination after the move must be ≤ 70% CPU and ≤ 80% RAM; a destination already above is refused too |
| Hotspot guardrail | hotspot modes | the aggregate peak of the hot resource must drop strictly, and the destination must stay strictly below its threshold |

Example D below shows both.

## 6. Accepting a move: worked examples

### Example D. RAM hotspot

Hosts (CPU / RAM): A 55 / 86, B 50 / 52, C 48 / 49. RAM 86 > 80 → `ram_hotspot`, weights 0.2 / 0.8. (The old script called 86% critical, because its critical limit was 85.)

```text
cpu_mad = 2.67   ram_mad = 15.78   baseline = 2.67 × 0.2 + 15.78 × 0.8 = 13.16
```

Hotspot-first: VMs on A only.

| Candidate | Result | Decision |
|---|---|---|
| VM1 (4 CPU, 16 RAM points) A → B | A 51 / 70, B 54 / 68; peak RAM 86 → 70; B 68 < 80 | after = 2.00 × 0.2 + 8.89 × 0.8 = 7.51, improvement **5.64 ≥ 0.15**, accepted |
| VM2 (32 RAM points) A → C | C would reach 49 + 32 = 81% | refused: `would_create_new_hotspot` |
| VM2 A → B | B would reach 84% | refused |

VM2 frees more RAM on A, but it would only move the hotspot. Status `RAM_HOTSPOT_OPTIMIZABLE`, severity WARNING.

### Example E. CPU hotspot

CPU 74 / 45 / 40, RAM 50 / 48 / 46 on 64-CPU, 512 GiB hosts. CPU 74 > 70 → `cpu_hotspot` (the old script stayed in MAD mode until 85). Weights 0.8 / 0.2, baseline 14.00 × 0.8 + 1.33 × 0.2 = 11.47. A VM of 8 vCPU at 80% (10 CPU points) and 8 GiB used (1.56 RAM points) leaves A:

| Destination | CPU after | Weighted MAD after | Improvement |
|---|---|---|---|
| B | 64 / 55 / 40 | 7.20 | 4.27 |
| C | 64 / 45 / 50 | 5.93 | **5.54**, chosen |

### Example F. A valid move that is still rejected

RAM 56 / 50 / 44, CPU 30 / 30 / 30. RAM MAD 4.00 > 3 → unbalanced. CPU MAD is 0, so the floor gives 0.2 / 0.8 and the baseline is 3.20. Suppose every large VM is in cooldown and the only movable VM is worth 0.2 RAM points and 0.1 CPU points (about 1 GiB and 0.06 cores on a 512 GiB, 64-CPU host):

```text
RAM 56/50/44 -> 55.8/50/44.2    ram_mad 3.87      CPU MAD 0.07
after = 0.07 × 0.2 + 3.87 × 0.8 = 3.11      improvement = 0.09 < 0.15
```

The move passes every filter, but a live migration for 0.09 is churn. Nothing is migrated. Status `UNBALANCED_NO_VALID_MOVE`, reason `no_candidate_above_improvement_threshold`, severity WARNING. The old threshold (0.05) would have migrated it.

## 7. MAKE_ROOM: two-step plans

When no direct move reaches 0.15, the planner looks for a two-step plan ([domain/balance/makeroom.py](../src/compute_maintenance/domain/balance/makeroom.py) `find_best_make_room_plan`): a host B looks good for VM V but is blocked **by Placement capacity only**, so first move a small VM R from B to a third host C, then V from A to B. The rc11 soft-anti-affinity tie-break (8.5) applies only to direct candidates in the balancing modes (hotspot, MAD and pressure alike); MAKE_ROOM plan selection deliberately does not use it.

| Rule | Value |
|---|---|
| Enabled | `ENABLE_MAKE_ROOM=true` (default) |
| Depth | exactly 2 steps (`MAKE_ROOM_MAX_DEPTH` must be 2) |
| Target B | V does not fit B **only** because of `placement_vcpu` / `placement_memory` |
| Room VM R | passes the VM filters; own CPU ≤ 10% (`MAKE_ROOM_MAX_VM_CPU_PERCENT`) and guest RAM ≤ 50% (`MAKE_ROOM_MAX_VM_RAM_PERCENT`) |
| Buffer host C | a third host (neither A nor B); same filters as a direct destination, including no new hotspot |
| Step 2 check | after simulating R's release, V must fit B; server groups and instance counts are rechecked with R already on C |
| Final state | no host becomes a new hotspot; B and C stay ≤ 70% CPU / 80% RAM; hotspot guardrail in hotspot modes |
| Threshold | total improvement ≥ `MAKE_ROOM_MIN_IMPROVEMENT`, default 0.30 (= 0.15 × 2 migrations; a configured value below 0.15 is rejected) |

### Example G. MAKE_ROOM

Hosts 64 CPUs / 512 GiB. A 60 / 78, B 30 / 45, C 45 / 60 (CPU / RAM). MAD mode, weights 0.469 / 0.531, baseline 10.71.

- V on A: 8 vCPU at 80%, 32 GiB at 80% → 10.00 CPU points, 5.00 RAM points.
- B Placement VCPU free 7.2 (the example in 2.3) → V does not fit (`placement_vcpu`).
- V → C is refused: V is in an anti-affinity group with a VM on C. No other direct move reaches 0.15.
- R on B: 4 vCPU at 5% CPU, 8 GiB at 30% RAM → qualifies (5 ≤ 10, 30 ≤ 50).

```text
Step 1  R: B -> C     B frees 4 VCPU -> free 11.2      C 45.31 / 60.47 (no hotspot)
Step 2  V: A -> B     V now fits (8 <= 11.2)
Final   CPU 50 / 39.69 / 45.31   RAM 73 / 49.53 / 60.47
        cpu_mad 3.54   ram_mad 8.00   after = 5.91   improvement = 4.80 >= 0.30
```

Execution ([application/balancing.py](../src/compute_maintenance/application/balancing.py)):

1. Step 1 is revalidated and must not create a hotspot on C, but it does not have to improve MAD by itself.
2. Before step 2 the tool waits for the metric window to settle (10.2), fetches everything again, and retries only while Placement still reports the old usage (`PLACEMENT_SYNC_TIMEOUT` 30 s, every 2 s).
3. Step 2 must improve the **fresh** state by ≥ 0.15 on its own (4.91 if the metrics match the projection). If it is no longer safe or worthwhile it is not executed; R simply stays on C and the event log records why.

When MAKE_ROOM is searched, the planner keeps whichever of the two candidates improves more, and that one must clear its own threshold. Example: direct 0.10, MAKE_ROOM 0.25 → MAKE_ROOM is kept, needs 0.30, so nothing runs.

## 8. Soft-anti-affinity heal (rc11)

Nova checks a soft-anti-affinity group only when it places a VM. A drain or a failover with no other room may therefore leave two members of a group on one host: rescue the workload first, repair the spread later, the same trade-off VMware HA makes by default. Up to rc10 nothing repaired it. Since rc11 the balancer does, following VMware DRS's constraint-correction pattern: correct a rule violation at the next run instead of hoping a load-balancing move fixes it by chance ([domain/balance/heal.py](../src/compute_maintenance/domain/balance/heal.py) `plan_soft_aa_heal`). It is on by default; `HEAL_SOFT_ANTI_AFFINITY=false` restores the rc10 behaviour exactly, and broken groups are then only reported.

### 8.1 Detection

For each `soft-anti-affinity` group, per aggregate:

```text
violations = sum over hosts of max(0, members on the host - 1)
```

Only VMs in the snapshot that sit on a host of the aggregate count (hosts left out as not ready, 2.1, do not). A group with at least 1 violation is **broken**. Example: members on A, A, A, B → (3 − 1) + (1 − 1) = 2 violations. Hard `anti-affinity` and `soft-affinity` groups are not healed.

### 8.2 Position in the ladder

```text
ram_critical_hotspot > ram_hotspot > cpu_hotspot > soft_aa_heal > MAD balancing (pressure / mad)
```

- **After the hotspots.** With a hotspot present the heal is deferred (`heal deferred while a hotspot is handled`). A host at 95% RAM is a resource-safety problem now; two members of one group on the same host is a redundancy problem that can wait a few cron runs. VMware DRS orders it the other way: it corrects constraint violations before load balancing, because for DRS imbalance is only an optimisation. In wbalancer a hotspot is a safety threshold, so resource safety outranks redundancy repair.
- **Before MAD balancing.** With no hotspot, a feasible heal move replaces the balancing plan of that run. The normal MAD search still runs; its verdict is shown as `Load status:`.
- **Never blocks balancing.** Heal disabled, no broken group, or no feasible heal move: the run continues with MAD balancing exactly as before.
- Still **one move per aggregate per run**.

### 8.3 Choosing the heal move

1. **Group:** the group whose worst host holds the most members goes first. If it cannot be fixed, the next broken group is tried.
2. **Source:** the host holding the most members of the group (other crowded hosts are tried after it).
3. **VM:** the member with the smallest flavor RAM that passes the VM filters (5.1: not excluded, not in cooldown, not locked, no task, no CPU pinning / PCI / `resources:*`). If it has no valid destination, the next smallest member is tried.
4. **Destination:** only hosts with **no** member of the group, and only if they pass **every** destination filter of 5.2: ready, instance limit, hard server groups, soft groups never worse, AZ, Placement with its 5% headroom, runtime capacity, no new hotspot.
5. **Net gain:** the move must remove at least one violation in total, counting every soft-anti-affinity group of the VM.
6. **MAD gate, per resource:** for CPU and for RAM separately, `MAD after ≤ limit`, where `limit = TARGET_AVG_MAD` when `MAD before ≤ TARGET_AVG_MAD`, and `limit = MAD before + SOFT_AA_HEAL_MAD_TOLERANCE` when `MAD before > TARGET_AVG_MAD` (defaults 3 and 0.05; `soft_aa_heal_mad_check`). The `UNBALANCED` label uses the same test (CPU MAD or RAM MAD above `TARGET_AVG_MAD`, section 11), so a heal never pushes a resource's MAD above the target unless it already was, and never hands new work to the MAD rung. A resource already above the target may grow by at most the tolerance. A heal must not buy redundancy with an unbalanced aggregate.
7. Among the destinations left, the one with the most **headroom** wins: `min(70 − CPU after, 80 − RAM after)` (the hotspot thresholds), then the lower weighted MAD.

### Example I. Heal

Hosts 32 CPUs / 256 GiB. CPU / RAM: A 30 / 70, B 25 / 40, C 28 / 45. No hotspot, `mad` mode: CPU MAD 1.78, RAM MAD 12.22, weights 0.20 / 0.80 (floor), baseline 10.13. Group `db` (soft-anti-affinity) has db1 (8 vCPU, 16 GiB, CPU 50%, RAM 90% = 14.4 GiB) and db2 (4 vCPU, 8 GiB, CPU 20%, RAM 75% = 6 GiB), both on A: 1 violation.

- Source A (2 members). Smallest flavor: db2 (8 GiB), worth 0.8 cores = 2.50 CPU points and 6 GiB = 2.34 RAM points.
- Destinations: B and C (no member of `db`); both pass every filter.

| Destination | CPU / RAM after | Headroom | CPU MAD / RAM MAD after (limits 3.00 / 12.27) | Weighted MAD after |
|---|---|---|---|---|
| B | 27.50 / 42.34 | min(70 − 27.50, 80 − 42.34) = **37.66** | 0.22 / 10.66, gate passed | 8.57 |
| C | 30.50 / 47.34 | min(39.50, 32.66) = 32.66 | 1.89 / 10.66, gate passed | 8.91 |

The limits are per resource: CPU MAD 1.78 ≤ 3, so its limit is 3.00; RAM MAD 12.22 > 3, so its limit is 12.22 + 0.05 = 12.27. db2 A → B: CPU MAD 1.78 → 0.22, RAM MAD 12.22 → 10.66 (weighted 10.13 → 8.57), violations 1 → 0. Status `SOFT_AA_HEAL`, severity WARNING, `Load status: UNBALANCED (WARNING)`. The MAD rung alone would have moved db1 (improvement 2.00); the heal rung moves the smaller member, the cheaper migration.

### 8.4 Groups without a heal move, execution, recovery speed

- **No heal move this run:** MAD balancing runs, and the output, the log and an event say why. The monitor severity is **not** raised: like DRS's "Could not fix anti-affinity rule violation" fault, it is reported without alert spam. Two labels:
  - `unresolvable`, only for the structural case: every host of the aggregate already holds a member. Example: a 4-member group in a 3-host aggregate always keeps at least 1 violation, and says so on every run (event `soft_aa_heal_unresolvable`).
  - `deferred`, for causes that can clear by themselves: members in cooldown, locked, busy, or with metrics still on two hosts right after a move (`missing_vm_details`, about 5 minutes); every member-free destination refused by a filter or by the MAD gate; the search budget. The blocking counts are listed, e.g. `deferred: no movable member on the crowded host(s) (cooldown=1, missing_vm_details=1)` (event `soft_aa_heal_deferred` with `"blocking": {"cooldown": 1, "missing_vm_details": 1}`).
- **The gate on a balanced aggregate** (the main use case: the group is broken, the load is fine). Same hosts and group as Example I, but CPU 30 / 30 / 30 and RAM 50 / 50 / 50: CPU MAD 0 and RAM MAD 0, both at or below the target, so both limits are **3**. db2 (8 GiB flavor, 6 GiB used: 2.50 CPU and 2.34 RAM points) A → B gives CPU 27.5 / 32.5 / 30 and RAM 47.66 / 52.34 / 50: **CPU MAD 1.67 ≤ 3 and RAM MAD 1.56 ≤ 3, allowed**; the aggregate stays `BALANCED`. Counterexample: if both members were 8 vCPU / 32 GiB guests using all of their RAM and half of their CPU (12.5 CPU and 12.5 RAM points each), any heal ends at CPU MAD 8.33 and RAM MAD 8.33 > 3 and stays refused (`mad_tolerance`); move such a group by hand or add capacity.
- **One resource is enough to refuse.** Weights 0.2 / 0.8 (CPU MAD 0, RAM MAD 2.0 before): a heal ending at CPU MAD 0.5 / RAM MAD 3.6 is refused because RAM 3.6 > 3 (RAM MAD 2.0 was below the target, so its limit is the target), even though its weighted MAD (2.98) is below 3; the same heal with a member using less RAM (RAM MAD 2.4) passes. **Just under the target the target still rules:** RAM MAD 2.98 before → limit 3.00 (not 3.03), so a heal ending at 3.00 passes and one ending at 3.02 does not. **Above the target the tolerance rules:** RAM MAD 3.50 before → RAM limit 3.55, so a heal ending at RAM MAD 3.51 passes and one ending at 3.60 does not.
- **Execution** is the direct-move path: journal, freshness, revalidation, cooldowns, events. Revalidation (10.1) re-reads the server groups and checks that the move **still heals** (else `soft_aa_heal_no_longer_heals`) and that the fresh CPU and RAM MAD stay within the same per-resource limits, computed from the fresh snapshot's own CPU/RAM MAD by the same function as the planner; the 0.15 improvement threshold does not apply. Plan type, step reason and event mode are `soft_aa_heal`.
- **Recovery speed:** one move per aggregate per run, and the settle barrier (10.2) before each balance snapshot. N violations take about N runs, about **N × 10 minutes** with a 10-minute cron. Example: a drain that left 3 violations is repaired in about 30 minutes, longer if a hotspot needs handling first or a member is in cooldown.

### 8.5 Tie-breaks

Both are off with `HEAL_SOFT_ANTI_AFFINITY=false`.

- **Balancing (every balancing mode — hotspot, MAD, pressure):** among direct moves that pass 0.15 and lie within `SOFT_AA_TIEBREAK_MARGIN` (default 0.15, relative) of the best improvement, the one removing the most soft-anti-affinity violations wins; on equal removal the better improvement stays. Example: best move 10.25 (no group), a move that heals a group 9.00: 9.00 ≥ 10.25 × 0.85 = 8.71, so the healing move is chosen; a healing move worth 4.50 is not. A move below 0.15 is never admitted.
- **Drain:** the packing key `(soft penalty, weighted MAD)` (9.1) gets a third key, the number of members of the VM's soft-anti-affinity groups on the destination. It decides only when the first two are equal, for example a VM in a soft-anti-affinity and a soft-affinity group facing two identical hosts, one holding a member of each group. It never changes which drains are feasible.

## 9. Drain planning

`--drain-host H` empties one host and keeps the remaining hosts balanced ([domain/drain.py](../src/compute_maintenance/domain/drain.py)). The aim is to empty the host, not to improve MAD, so the improvement threshold and the hotspot guardrail are off. Placement, hard server groups, AZ, instance limits and runtime caps stay on. Exclude list and cooldown are ignored.

### 9.1 Steps

1. **Classify** each VM on H as movable or unmovable (reasons in 5.1). `locked`, `not_active:SHUTOFF`, `unsupported_resources` and similar need the operator. When only transient ones are left (`task_in_progress`, `incomplete_metrics`, `incomplete_metadata`, `missing_vm_details`, e.g. a VM hidden behind the previous migration's metric window), the drain re-plans every 15 s for up to 360 s before concluding.
2. **Capacity budget:** sum of the movable flavors vs the sum of Placement `free` on ready remaining hosts. A shortfall makes the drain infeasible at once, with the exact numbers (`not enough VCPU: need 14, only 10 safe-free on 2 host(s) (short 4)`).
3. **Greedy packing**, tried in three orders (flavor RAM desc, flavor vCPU desc, observed RAM desc). For each VM, every remaining host is checked: ready, instance limit, hard server group, AZ, Placement, then the runtime caps. Among the hosts left it picks the lowest `(soft-group penalty, weighted MAD of the remaining hosts)`; since rc11 a full tie is broken by the fewest members of the VM's soft-anti-affinity groups on the host (8.5). Placement usage, load, instance counts and group topology are updated after every placement.
4. **Best order:** fewest blocked VMs, then fewest soft-group warnings, then lowest final weighted MAD.
5. **Fail-closed:** feasible only when no VM is blocked, there is no capacity shortfall and no host is projected above 95% RAM. Otherwise nothing is migrated and the nova-compute service is not touched; exit 2.

Runtime caps on a receiving host ([config.py](../src/compute_maintenance/config.py)):

| | CPU cap | RAM cap |
|---|---|---|
| default | 70% | 80% |
| `--drain-allow-hotspot` | none (100%) | 95%, the absolute ceiling, never lifted |

The weights come from the current aggregate snapshot, drained host included; the score counts only the remaining hosts. An affinity group with 2+ members on H cannot leave one VM at a time; the report names it and suggests moving it by hand or `--no-server-groups`.

### Example H. Drain

H1 is drained; H2 and H3 remain (64 CPUs / 512 GiB each). CPU / RAM: H1 30 / 60, H2 40 / 72, H3 35 / 55. MAD mode, weights 0.341 / 0.659.

| VM on H1 | Flavor | Use | CPU / RAM points |
|---|---|---|---|
| vm1 | 8 vCPU, 64 GiB | CPU 50%, RAM 75% (48 GiB) | 6.25 / 9.38 |
| vm2 | 4 vCPU, 32 GiB | CPU 25%, RAM 50% | 1.56 / 3.13 |
| vm3 | 2 vCPU, 16 GiB | CPU 50%, RAM 50% | 1.56 / 1.56 |

Order by flavor RAM: vm1, vm2, vm3. Placement has room on both hosts.

| VM | → H2 | → H3 | Chosen |
|---|---|---|---|
| vm1 | RAM 81.4% > 80 → `ram_cap_exceeded` | RAM 64.4%, score 2.73 | H3 |
| vm2 | RAM 75.1%, score 3.60 | RAM 67.5%, score 1.96 | H3 |
| vm3 | RAM 73.6%, score 2.21 | RAM 69.1%, score 1.71 | H3 |

Final H2 40 / 72, H3 44.4 / 69.1: feasible.

Same aggregate, but H3 has only **48 GiB** safe-free RAM in Placement. vm1 (64 GiB) no longer fits H3 (`placement_memory`) and would push H2 to 81.4%. vm1 is blocked, so the **whole drain is infeasible**: nothing moves, exit 2, the report lists vm1 with both reasons. With `--drain-allow-hotspot` the RAM cap becomes 95%: vm1 → H2 (81.4%), vm2 and vm3 → H3, feasible.

### 9.2 Execution loop

[application/balancing.py](../src/compute_maintenance/application/balancing.py) `drain_host_workload`:

- nova-compute on H is disabled first (unless `--drain-no-disable`); it is re-enabled only if the drain aborts and this run disabled it.
- Each round fetches fresh Nova, Placement and Prometheus data, re-plans the whole remaining evacuation and migrates **only the first step**. No migration is ever submitted from a stale plan.
- Before the submit the step is revalidated. A refused step (nothing submitted) is re-planned after 30 s, up to 3 times in a row. A failed migration, or a re-plan that is no longer feasible, aborts the drain (exit 1).
- At most `2 × steps + 3` rounds. The drain ends when the plan has no step left; exit 0 only when no VM is left on H (unmovable VMs → exit 2).

### 9.3 Load compensation: why the drain does not wait 6 minutes per VM

Host RAM is a 5-minute average and host CPU uses `irate`. A VM that arrived t seconds ago is still missing `(1 − t/300)` of its RAM from the destination's reading. Without a correction the next round sees a destination emptier than it is and keeps filling it. Balance runs avoid this by waiting the full window (10.2). Drain would then cost 6 minutes per VM, so instead it adds the load of recently arrived VMs to their destination ([application/snapshots.py](../src/compute_maintenance/application/snapshots.py) `apply_arrival_compensation`):

| Method | When | RAM of the receiving host |
|---|---|---|
| `measured` | a raw `MemAvailable` sample newer than every arrival exists (the tool waits up to max(90 s, 3 × scrape interval) for it) | that raw sample, no estimate |
| `measured+rss` | raw sample exists but some VMs arrived after it | max(average, raw) + host RSS of those later VMs |
| `model` | no raw sample | average + RSS × (1 − age / RAM window) |

CPU is added as `VM cores / host CPUs × 100 × (1 − age / cpu_window)`, where `cpu_window = 2 × scrape interval` (60 s fallback, or `COMPENSATION_CPU_WINDOW_SECONDS`). The source host is never reduced (over-estimating is the safe side). The VM's load comes from planner details; for a just-arrived VM without details, its CPU 15 s before the migration started; as a last resort its full flavor.

Example: vm1 (RSS 48 GiB, 4 cores) arrived on H3 (512 GiB, 64 CPUs) 60 s ago; scrape interval 30 s.

```text
RAM average still misses 48 × (1 - 60/300) = 38.4 GiB = 7.5 points   -> model adds +7.5
CPU: 4/64 × 100 × (1 - 60/60) = 0        (irate has caught up)
     had it arrived 15 s ago: 6.25 × (1 - 15/60) = +4.69 points
```

Measured on a 3-compute lab: the compensated RAM was within ±0.03 points of the next raw sample, against about 2.6 points of error without compensation.

## 10. Execution safety

### 10.1 Revalidation before submit

[application/migrations.py](../src/compute_maintenance/application/migrations.py) `validate_migration_step` re-reads Nova and Placement right before the migration:

- VM still exists, `ACTIVE`, no task, not locked, still on the planned source.
- Destination ready, same AZ, flavor fits **fresh** Placement, instance limit.
- Server groups re-read from Nova (soft groups too, except for drain).
- Fresh projection: destination not above 70% CPU / 80% RAM (drain with `--drain-allow-hotspot`: not above 95% RAM), never above 95% RAM.
- Balance moves (direct and MAKE_ROOM step 2): hotspot guardrail, and improvement on the fresh data ≥ 0.15.
- Soft-anti-affinity heal moves (rc11): instead of the 0.15 improvement, the move must still remove a violation on the fresh server-group topology, and the fresh CPU MAD and RAM MAD must each stay at or below `TARGET_AVG_MAD`, or grow by at most `SOFT_AA_HEAL_MAD_TOLERANCE` when already above it: the planner's own gate (8.3).
- The fresh snapshot is built with the planner's own host rule: balance leaves hosts that cannot receive a VM out of the baseline, drain keeps them. A host that became unready after planning therefore never shifts the mean/MAD the improvement is measured against; if it is the step's source or destination, the step is refused.
- More than 120 s between the fresh snapshot and the submit → refused, re-plan next run.

A refused step submits nothing and puts the VM in a 30-minute rejected-move cooldown, so a 10-minute cron does not re-plan the same move every run.

### 10.2 Settle barrier (balance)

In apply mode a balance snapshot waits until no migration touching the aggregate's hosts is running (including migrations not started by this tool) and `RUNTIME_SETTLE_SECONDS` (default 360) have passed since the last one finished. 360 = the 5-minute query window + 60 s of scrape margin; a smaller value that does not cover the largest PromQL range + 60 s is rejected.

### 10.3 Cooldowns

| Event | Cooldown | Stored in |
|---|---|---|
| Successful migration | 4 h (`VM_COOLDOWN_SECONDS`) | `moved` |
| Failed / timed-out migration | 4 h (`VM_FAILED_MOVE_COOLDOWN_SECONDS`) | `failed` |
| Refused at revalidation | 30 min (`VM_REJECTED_MOVE_COOLDOWN_SECONDS`) | `rejected` |

The longest remaining entry wins; a success clears the other two. Drain ignores all three.

### 10.4 Migration timeout that follows progress

[domain/errors.py](../src/compute_maintenance/domain/errors.py) `ProgressTimer(600, 600, 1800)`: one no-progress window (config key `MIGRATION_TIMEOUT`, default 600 s) and a cap (`MIGRATION_MAX_WAIT`, default 1800 s):

- stop waiting once nothing changed for 600 s, counted from the start or from the last change: migration status, memory/disk processed and remaining, the row's `updated_at`, or a migration queued ahead of it on the same source host (polled every 15 s);
- never wait longer than 30 min (1800 s); the migration itself keeps running in Nova and the next run reconciles it.

| Migration | Old script | v1 |
|---|---|---|
| Progressing all along, done at minute 25 | timeout at minute 10 | success at minute 25 |
| Frozen from the start | timeout at minute 10 | stops at minute 10 |
| Progress until minute 2, then frozen | timeout at minute 10 | stops at minute 12 |
| Progress until minute 12, then frozen | timeout at minute 10 | stops at minute 22 |
| Still progressing at minute 30 | timeout at minute 10 | stops at minute 30 (`max_wait`); raise `MIGRATION_MAX_WAIT` for such VMs |

A VM still `ACTIVE` on its source with no migration record after 120 s is reported as `stuck_active`. A timeout is not treated as proof of failure: the migration journal reconciles the real outcome with Nova on the next run ([operations](operations.md)). Unlike the old script, apply mode does not reset an `ERROR` VM to `ACTIVE`.

## 11. Monitor status and Icinga severity

[domain/balance/classify.py](../src/compute_maintenance/domain/balance/classify.py) `classify_cluster_detail_status` and [domain/severity.py](../src/compute_maintenance/domain/severity.py) `map_detail_status_to_severity`. A "valid move" is a direct or MAKE_ROOM plan that clears its threshold.

| Mode / state | Valid move | Status | Severity | Exit |
|---|---|---|---|---|
| RAM hotspot or critical | yes | `RAM_HOTSPOT_OPTIMIZABLE` | WARNING | 1 |
| RAM hotspot or critical | no | `RAM_HOTSPOT_NO_VALID_MOVE` | CRITICAL | 2 |
| CPU hotspot | yes / no | `CPU_HOTSPOT_OPTIMIZABLE` / `CPU_HOTSPOT_NO_VALID_MOVE` | WARNING / CRITICAL | 1 / 2 |
| Pressure | yes / no | `RAM_PRESSURE` or `CPU_PRESSURE` / `PRESSURE_NO_VALID_MOVE` | WARNING / CRITICAL | 1 / 2 |
| MAD > 3 (CPU or RAM) | yes | `UNBALANCED` | WARNING | 1 |
| MAD > 3 | no, some move passed the filters | `UNBALANCED_NO_VALID_MOVE` | WARNING | 1 |
| MAD > 3 | no, no move passed the filters (`no_technical_candidate`) | `UNBALANCED_NO_VALID_MOVE` | CRITICAL | 2 |
| MAD ≤ 3 | yes / no | `BALANCED_OPTIMIZABLE` / `BALANCED` | OK | 0 |
| No hotspot, broken soft-anti-affinity group, heal move feasible (rc11) | heal move | `SOFT_AA_HEAL` | WARNING, or the load row's severity if that is worse | 1 (2) |
| Broken group, heal not possible / disabled / deferred by a hotspot | — | load row unchanged + a `Soft-anti-affinity:` note | unchanged | unchanged |
| Data error, < 2 hosts, budget exhausted | — | `UNKNOWN` | UNKNOWN | 3 |

- A heal WARNING is combined with the MAD rung's own result (shown as `Load status:`) by the same CRITICAL > WARNING > UNKNOWN > OK ranking as the next point, so it never hides a CRITICAL. Example: pressure mode with no valid move (CRITICAL) plus a feasible heal stays CRITICAL; a balanced aggregate (OK) with a feasible heal becomes WARNING.

- `--monitor-only` prints one line per aggregate and exits with the most severe result, ranked CRITICAL > WARNING > UNKNOWN > OK (not the highest number). A data error in one aggregate never hides an alert in another: CRITICAL + UNKNOWN exits 2, WARNING + UNKNOWN exits 1, OK + UNKNOWN exits 3. Each aggregate's own line still shows its UNKNOWN.
- Example: RAM 96% on one host with a valid move is only WARNING; the same host with no valid move is CRITICAL.
- Change from the old script: there, a move that passed the filters but had zero or negative improvement still counted as "no technical candidate" (CRITICAL). v1 counts it as a candidate below the threshold (WARNING).

## 12. Reading a decision as an operator

Read the output top-down:

| Line | Tells you |
|---|---|
| Headline | Severity, status, best improvement vs threshold, hottest host |
| `Mode:` / `Peak:` | Which rule drives the decision and the highest CPU/RAM |
| `Spread:` / `Priority:` | MADs, weighted MAD and the CPU/RAM weights in use |
| `Placement:` | Hosts ≥ 90% allocated or with no safe-free capacity |
| `Reason:` | Why nothing was accepted (`no_technical_candidate`, `…_above_improvement_threshold`) |
| `Load status:` | Only when a heal move is proposed: what the MAD rung itself concluded (rc11) |
| `Soft-anti-affinity:` | Broken groups and violations, the heal move, or per group `unresolvable: <reason>` (structural) / `deferred: <reason>` (temporary); `heal disabled` / `heal deferred` (rc11) |
| `Rejected candidate pairs:` | Top filters that blocked moves; hosts most often blocked by Placement are listed under it |
| `Search scope:` | `HOTSPOT-FIRST`, `FALLBACK-FULL`, `FULL`, `MAKE_ROOM …`, `SOFT_AA_HEAL (group)` or `DRAIN host` |
| `Step n:` | VM, source → destination, flavor need, observed load, Placement free before the step, impact on both hosts |

This separates "the aggregate is fine", "not ideal but no safe or worthwhile move exists" and "a safe corrective move is available".

## 13. Constants

All in [config.py](../src/compute_maintenance/config.py). "Config" means it can be set in `wbalancer.conf` or the environment.

| Name | Value | Config |
|---|---|---|
| `RAM_HOTSPOT_THRESHOLD` / `RAM_CRITICAL_THRESHOLD` / `CPU_HOTSPOT_THRESHOLD` | 80 / 95 / 70 % | yes |
| `PRESSURE_THRESHOLD` (internally `AUTO_PRESSURE_THRESHOLD`) | 80 % | yes |
| `TARGET_AVG_MAD` | 3 | yes |
| `MIN_IMPROVEMENT_THRESHOLD` | 0.15 | yes |
| `PLACEMENT_PRESSURE_THRESHOLD` (report only) | 90 % | yes |
| `MIN_CPU_WEIGHT` / `MIN_RAM_WEIGHT` | 0.2 / 0.2 | no |
| `PRESSURE_MAD_WEIGHT` / `PRESSURE_MEAN_WEIGHT` | 0.7 / 0.3 | no |
| `PLACEMENT_VCPU_HEADROOM_PERCENT` / `PLACEMENT_RAM_HEADROOM_PERCENT` | 5 / 5 % | yes |
| `ENABLE_MAKE_ROOM`, `MAKE_ROOM_MAX_DEPTH` | true, 2 (only 2 allowed) | yes |
| `MAKE_ROOM_MIN_IMPROVEMENT` | 2 × `MIN_IMPROVEMENT_THRESHOLD` = 0.30 (must be ≥ it) | yes |
| `MAKE_ROOM_MAX_VM_CPU_PERCENT` / `MAKE_ROOM_MAX_VM_RAM_PERCENT` | 10 / 50 % | yes |
| `PLACEMENT_SYNC_TIMEOUT` / `PLACEMENT_SYNC_INTERVAL` | 30 / 2 s | yes |
| `MAX_INSTANCES_PER_HOST_DEFAULT`, `MAX_INSTANCES_PER_HOST` | 0 (unlimited), JSON map | yes |
| `PLANNER_CANDIDATE_BUDGET` | 100 000 | yes |
| `HEAL_SOFT_ANTI_AFFINITY` | true | yes |
| `SOFT_AA_HEAL_MAD_TOLERANCE` | 0.05 MAD points (≥ 0); heal gate per resource: `TARGET_AVG_MAD`, or MAD before + 0.05 when already above it | yes |
| `SOFT_AA_TIEBREAK_MARGIN` | 0.15, relative (0–1) | yes |
| Drain caps: CPU / RAM / absolute RAM | 70 / 80 / 95 % | follow `CPU_HOTSPOT_THRESHOLD` / `RAM_HOTSPOT_THRESHOLD` / `RAM_CRITICAL_THRESHOLD` |
| `DRAIN_METRIC_SETTLE_TIMEOUT` / interval | 360 / 15 s | no |
| `VM_COOLDOWN_SECONDS` / `VM_FAILED_MOVE_COOLDOWN_SECONDS` | 14 400 / 14 400 s | no |
| `VM_REJECTED_MOVE_COOLDOWN_SECONDS` | 1 800 s | no |
| `MIGRATION_TIMEOUT` (no-progress window) / `MIGRATION_MAX_WAIT` | 600 / 1 800 s | yes |
| `RUNTIME_SETTLE_SECONDS` | 360 s | environment only |
| `COMPENSATION_CPU_WINDOW_SECONDS` | 2 × scrape interval (60 s fallback) | yes |

## Related documents

- [User guide](user-guide.md): installation, configuration, commands, exit codes.
- [Operations](operations.md): state files, locks, journal and recovery.
- [Migrating from `wloadbalancer.py`](migrating-from-wloadbalancer.md): config and threshold differences, switch-over steps.
