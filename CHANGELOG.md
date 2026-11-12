# Changelog

## 1.0.0rc13 (2026-11-12)

Performance-only release: no behavior change — removed wasted API calls. Decisions, plans, report text, JSONL events, exit codes, the journal and the cooldown file are identical to rc12: a differential harness ran the real code of both builds against one deterministic fake cloud (24 scenarios: apply snapshot, revalidation, `--dry-run`, `--monitor-only`, `-y`, the MAKE_ROOM two-step flow, drain dry-run and apply with and without arrival compensation, a pending journal intent, Keystone and hypervisor-detail read failures, journal-less library calls) and found 0 differences in stdout, events, exit codes, journal and cooldown files, final VM placement and log lines, apart from the `Built all-project Nova cache …` INFO line of each removed Nova server-list read. The six items are class A of the 2026-11-11 redundancy review; its class B (safety double checks such as the pre-submit refresh and the MAKE_ROOM step-2 double refresh) is deliberately untouched.

### Removed (reads whose result was thrown away or fetched twice)

1. **A1 — drain no longer prefetches a cluster cache it throws away** (`application/balancing.py:737-745`). Drain runs in the compensate settle mode, whose snapshot always collects its own cache after waiting for running migrations, so the prefetch (Nova server list, availability zones, hypervisor map, 4 Placement REST calls per host, server groups, 2 Prometheus queries) was discarded on every `--drain-host` call, dry-run and apply alike; the rc11 lab measured it at about 3 s per call (drain dry-run 6.4 s against a 3.0–3.27 s snapshot). One predicate, `snapshots.prefetched_cluster_cache_is_used` (`application/snapshots.py:434-444`), mirrors the settle branches: only a journal-less barrier caller (dry-run, monitor, library use) still prefetches.
2. **A2 — the MAKE_ROOM step-2 wait loop no longer builds a cache before the settle barrier** (`application/migrations.py:1080-1090`): the same waste as the rc12 `balance_by_aggregate` fix, once per wait-loop attempt, collected before the up-to-360 s barrier that discarded it.
3. **A3 — one hypervisor map per revalidation** (`application/migrations.py:553-563`, `:587`, `:610-617`): the source alias, the destination's Placement provider name and the server-group refresh read the same service-host → hypervisor-name map, which was fetched three times (each `GET /os-hypervisors` plus one `GET /os-hypervisors/{id}` per hypervisor of the whole cloud). Saves 2 × (1 + hypervisors) GET per validation, 1 × without server groups.
4. **A4 — one hypervisor map per snapshot pass** (`adapters/openstack.py:600-612`, `application/snapshots.py:91-110`, `:410-423`, `:493-501`, `:616`): the cluster cache keeps the map it read for the Placement lookups, and the VM inventory of the same pass reuses it (restricted to its hosts, it is exactly the map a fetch for those hosts returns; a test checks this on duplicated, failing and colliding hypervisors). Saves 1 + hypervisors GET per apply and drain snapshot. A cache from an earlier pass (dry-run, monitor) does not lend its map: the inventory still reads its own, as before.
5. **A5 — Keystone project names once per project per run** (`application/snapshots.py:371-397`, `:430`, `:618`): every snapshot started an empty name cache, so each apply and drain snapshot asked Keystone again for every project, contrary to the code comment "once per project per run" (corrected). Names Keystone resolved are kept for the run on its config object; a project Keystone could not name is not kept and is asked again by the next snapshot, as before. Names are display-only (`project_name` in events and alert mails); no decision reads them.
6. **A6 — revalidation reads the step's VM once** (`adapters/openstack.py:150-189`, `application/migrations.py:610-617`): for a VM in a server group, the group refresh fetched it again as one of the members; it now reuses the record read at the start of the same validation.

Nothing is reused across a settle barrier, a journal settle or a refresh that exists to observe post-migration state; reads are shared only inside one snapshot or one revalidation pass (see [architecture, invariant 4](docs/architecture.md#4-invariants-never-break-these)).

### Measured (harness, fake cloud with 4 hosts, 14 VMs, 5 projects, one 3-member soft-anti-affinity group)

OpenStack REST calls per flow, `find_resource_provider` counted as the 2 REST calls the SDK makes:

| Flow | rc12 | rc13 | Saved |
|---|---:|---:|---:|
| `wbalancer -y`, 2 aggregates, 2 direct moves | 212 | 156 | 56 (26 %) |
| MAKE_ROOM two-step flow (step 1, step-2 wait, step 2) | 280 | 189 | 91 (32 %) |
| `--drain-host --dry-run` | 69 | 38 | 31 (45 %) |
| `--drain-host -y`, 2 VMs | 344 | 243 | 101 (29 %) |
| `--drain-host -y`, 6 VMs, one in a group | 807 | 585 | 222 (28 %) |
| One revalidation, VM in a server group | 27 | 16 | 11 (41 %) |
| One apply snapshot | 42 | 37 | 5 (12 %) |
| `--dry-run`, `--monitor-only` (2 aggregates) | 51 | 51 | 0 |

In general, with H hypervisors in the cloud and P projects in an aggregate: A1 saves one full cluster-cache collection per drain call and A2 one per step-2 attempt; A3 saves 2 × (1 + H) GET per validation, A4 1 + H per apply or drain snapshot, A5 P per apply or drain snapshot after the first, A6 one GET per validation of a grouped VM. A3 + A4 + A5 run inside the 120 s pre-submit budget.

### Error path (the same order `wbalancer -y` has had since the rc12 prefetch fix)

When the Nova server list cannot be read at all, a drain now fails at its snapshot's own read instead of at the removed prefetch: the same error, exit code 1 and no event, but the `[Aggregate: …] Drain evaluation for host …` header is printed first, and the settle step (migration scan, reconcile of a pending journal intent that Nova can prove) runs before the failing read.

### Tests

New [`tests/test_rc13_redundancy.py`](tests/test_rc13_redundancy.py) (26 tests): the prefetch predicate for every settle/journal combination; drain and the step-2 loop build no discarded cache while journal-less callers still get theirs; a real drain dry-run collects exactly one cache; one hypervisor map and one read of the step's VM per validation (with and without server groups), the shared map naming the Placement provider and translating a group member that reports only its hypervisor name; one map per snapshot pass, none borrowed from an earlier pass, a fetch when the given map does not cover the hosts; Keystone asked once per project over two snapshots, an unresolved project asked again and named once Keystone recovers, names per run. Mutation checks: reverting each fix or its guard (13 mutations) makes at least one new test fail; files restored byte-identical. The `revalidate` fixture's fake `get_fresh_server_group_context` in `test_rc11_soft_aa_heal.py` now accepts the two new keyword arguments (signature only). Windows: 626 passed, 15 skipped.

### Documentation

- `docs/architecture(.vi).md`: "as of 1.0.0rc13"; invariant 4 states where a read may be reused.

## 1.0.0rc12 (2026-11-11)

Internal refactor: technical-debt cleanup with no behaviour change. Decisions, monitor and dry-run text, JSONL events and exit codes are identical to rc11: differential sampling against the rc11 build, 1 500 random scenarios per configuration (heal on with server groups, heal on without soft-anti-affinity groups, heal on without any group, heal off, non-default thresholds) found 0 differences, and 13 fixed scenarios (dry-run, apply, MAKE_ROOM, drain, monitor, heal) print byte-identical output and events.

### Changed (internal only)

1. **Typed `PlanContext`** (`domain/plan_context.py`) replaces the stringly-typed state dict that crossed `application/` boundaries. `application/snapshots.py` builds it; balancing, migrations, monitoring, reports and the drain planner read attributes. Item access (`ctx["mode"]`, `ctx.get(…)`, `in`, `dict(ctx)`) still works for existing readers and shows exactly the keys rc11 had (a failed evaluation still has no `best_plan` key). Event labels are a frozen `PlanLabels`.
2. **`domain/balance.py` (1 334 lines) split into the package `domain/balance/`**: `modes`, `filters`, `scoring`, `makeroom`, `heal`, `classify`. Every definition moved verbatim; `from compute_maintenance.domain import balance` and `balance.X` keep working.
3. **Config object:** the live thresholds, drain limits, heal settings and migration waits are one frozen `config.RuntimeSettings`, read through `config.runtime()`; `apply_thresholds` / `apply_soft_aa_heal` / `apply_migration_timeouts` replace it. The module constants (`RAM_HOTSPOT_THRESHOLD`, …) stay as a compatibility layer kept in sync both ways. Keys, defaults, validation, precedence and `resolve_os_cloud` are unchanged; no source module imports a config value by name any more (enforced by a test).
4. **One severity/status mapping** (`domain/severity.py`): detail status → severity → Icinga exit code, the `worst_exit_code` ranking, the heal verdict, the UNKNOWN and drain detail statuses. The monitor, the balance classification and the events use it; the old names stay importable from `domain.balance` and `application.monitoring`.
5. **Hermetic tests:** the suite no longer reads the host's `/etc/wbalancer/wbalancer.conf`, so it also passes as an unprivileged user on a host where that file is root-only. New environment variable `WLB_CONFIG_PATH` replaces the default config file of `wbalancer --config` and `rolling-maint --wlb-config` (an explicit option still wins; [operations, section 2](docs/operations.md#2-state-files-and-locks)); the root `conftest.py` points it at a temp path.
6. **`ruff check src tests` is clean** (was 458 findings with ruff 0.17 defaults): mechanical fixes (f-strings, import order, literals, `check=False`, collapsed `if`s), a `# noqa: RULE - reason` on each deliberate broad `except` of the fail-closed I/O boundaries, and one ignored rule in `pyproject.toml` (`LOG015`, root logger by design).

### Fixed (internal, from the MAKE_ROOM review)

- `balance_by_aggregate` no longer prefetches the cluster cache in apply mode:
  with a journal, `evaluate_aggregate_state` waits on the settle barrier and
  discards any cache collected before it, so the prefetch (and the
  post-migration refresh) were collected and thrown away on every `-y` run —
  one wasted Nova/Placement/Prometheus collection per run. Dry-run and monitor
  behavior is unchanged. Two tests pin the rule both ways.

- **The MAKE_ROOM step-2 refresh no longer runs the planner.** `wait_for_make_room_step` evaluated its snapshot with the default `plan_balance=True`, so every Placement-convergence retry between the two migrations ran the whole planner and threw the plan away. It now passes `plan_balance=False`, like every other pre-submit refresh. Step-2 validation reads only the snapshot and the step's events keep the plan's own labels, so decisions, output and events are unchanged; the only effect is that the discarded planning (and what it could add: an exhausted planner budget, the optional allocation-candidate check) no longer happens inside that loop.
- **Tests for the two-step executor** (real balancing/migrations code and journal, faked Nova): step 2 unsafe after the refresh records exactly one `make_room_aborted_after_refresh` event (`step_index` 2) and raises; step 2 refused before submit keeps the successful step 1 (no error) and gives the step-2 VM a rejected-move cooldown; a crash between the steps leaves no step-2 journal intent, so nothing can be resubmitted from the old simulation. Plus a test that the step-2 refresh passes `plan_balance=False`.

### Documentation

- `docs/architecture(.vi).md`: module map and "where to change what" for the new modules, section 6 rewritten for `RuntimeSettings` (the compatibility layer and the remaining by-name-import trap), resolved debts removed from section 9.
- `docs/algorithm(.vi).md`: source links point at the `domain/balance/` modules and `domain/severity.py`; section 7 states that the rc11 soft-anti-affinity tie-break applies only to direct candidates, deliberately not to MAKE_ROOM plan selection.
- `docs/operations(.vi).md`: `WLB_CONFIG_PATH`.

## 1.0.0rc11 (2026-11-11)

Soft-anti-affinity heal, modelled on VMware DRS's constraint correction: a rule violation left behind by a drain or a failover is corrected by a later run instead of staying until a balancing move happens to fix it.

### Behaviour change (default on)

- **Broken `soft-anti-affinity` groups are re-spread.** A deployment that has groups with 2+ members on one host will see heal migrations (at most one per aggregate per run) and `--monitor-only` WARNING with status `SOFT_AA_HEAL` where rc10 reported OK. `HEAL_SOFT_ANTI_AFFINITY=false` restores the rc10 behaviour exactly (decisions, moves, exit codes, tie-breaks); broken groups are then only listed with `heal disabled`.

### Added

- **Heal rung** between the hotspots and MAD balancing: `ram_critical_hotspot > ram_hotspot > cpu_hotspot > soft_aa_heal > MAD`. See [algorithm, section 8](docs/algorithm.md#8-soft-anti-affinity-heal-rc11).
  - Detection per aggregate: `violations = sum over hosts of max(0, members on the host - 1)`, counting only VMs in the snapshot on the aggregate's hosts.
  - Only when no host is a hotspot: resource safety outranks redundancy repair (DRS orders it the other way because its imbalance is only an optimisation).
  - Move: the smallest-flavor-RAM movable member on the most crowded host, to a host with no member of the group that passes the whole direct-move filter chain (`check_direct_destination`, now shared with balancing: readiness, instance cap, hard and soft server groups, AZ, Placement with headroom, no new hotspot). MAD gate, per resource (one function, `soft_aa_heal_mad_check`, shared by the planner and the revalidation): a CPU or RAM MAD at or below `TARGET_AVG_MAD` must stay at or below it; one already above it may grow by at most `SOFT_AA_HEAL_MAD_TOLERANCE`. The `UNBALANCED` label uses the same test, so a heal never pushes a resource's MAD above the target unless it already was, and never hands new work to the MAD rung (RAM MAD 2.98 → at most 3.00, not 3.03; RAM MAD 3.50 → at most 3.55). Among the rest the destination with the most headroom below the hotspot thresholds wins. Examples: A 70% RAM holding db1 + db2, B 40%, C 45% → db2 (8 GiB) to B, CPU MAD 1.78 → 0.22 (limit 3.00), RAM MAD 12.22 → 10.66 (limit 12.27); a balanced aggregate (RAM 50 / 50 / 50) heals the same db2 at CPU MAD 0 → 1.67 and RAM MAD 0 → 1.56 (both ≤ 3), while 32 GiB members (8.33 / 8.33) stay refused, and so does a heal ending at CPU MAD 0.5 / RAM MAD 3.6 although its weighted MAD is 2.98.
  - Heal disabled, nothing broken or nothing feasible: MAD balancing runs exactly as before. Still one move per aggregate per run; N violations take about N runs (~N × 10 min with a 10-minute cron).
  - Execution is the direct-move path (journal, freshness, revalidation, cooldowns, events). Revalidation re-reads the server groups and requires the move to still heal (`soft_aa_heal_no_longer_heals`) and the fresh MAD to stay within the same gate, instead of the 0.15 improvement threshold.
  - A broken group without a heal move is reported in the output (`Soft-anti-affinity: …`), the log and an event, without raising the monitor severity, under one of two labels: `unresolvable` only for the structural case (`every host of the aggregate already holds a member`, event `soft_aa_heal_unresolvable`), `deferred` for causes that can clear by themselves (`no movable member …` from cooldown, lock or metrics still on two hosts after a move; `no member-free destination passes the filters (…)` including the MAD gate `mad_tolerance`; the search budget), with the blocking counts (event `soft_aa_heal_deferred`, `"blocking": {"cooldown": 1, "missing_vm_details": 1}`).
- **Monitor:** a feasible heal is WARNING, combined with the load result through `worst_exit_code`, so it never hides a CRITICAL; the load result is shown as `Load status:`. Icinga exit codes stay 0–3.
- **Events:** heal moves carry `"plan_type": "soft_aa_heal"`, `"mode": "soft_aa_heal"`, `"step_reason": "soft_aa_heal"` and a `soft_aa_heal` object (group, violations before/after, per-resource MAD limits). Every event now carries a `mode` (the planning load mode, `soft_aa_heal` or `drain`). Logs: `soft_aa_heal: group … -> move …` / `… unresolvable: …` / `… deferred: …`.

### Fixed

- **Fixes mislabeled JSONL severity present since rc10.** Every `migration_success` event (direct, MAKE_ROOM, heal, drain step) and every drain event said `"severity": "CRITICAL"`, `"detail_status": "UNBALANCED_NO_VALID_MOVE"`, `"best_improvement": 0.0`, while the run exited 0. The labels were copied from the snapshot refreshed just before the submit, which plans nothing (`plan_balance=False`) and therefore classifies itself "no technical candidate". Events now carry the labels of the snapshot the plan was made from (`reports.plan_event_labels`, carried through the pre-submit refresh and to MAKE_ROOM step 2), the same as the plan's `dry_run` event: a heal's `migration_success` reads `"mode": "soft_aa_heal"`, `"detail_status": "SOFT_AA_HEAL"`, `"severity": "WARNING"`. Drain events read `"mode": "drain"`, `"detail_status": "DRAIN_FEASIBLE"` / `"DRAIN_INFEASIBLE"`, `"severity": null` (a drain is not a balance check). Exit codes and monitor output are unchanged. Found by the rc11 lab canary.
- **Tie-break in balancing (hotspot and MAD modes):** among direct moves that pass `MIN_IMPROVEMENT_THRESHOLD` and lie within `SOFT_AA_TIEBREAK_MARGIN` (relative) of the best improvement, the one removing the most soft-anti-affinity violations wins. It only reorders valid candidates and never admits a move below the threshold.
- **Drain tie-break:** the packing key `(soft penalty, weighted MAD)` gets a third key, the number of the VM's soft-anti-affinity group members on the destination. It decides only full ties (e.g. a VM in a soft-anti-affinity and a soft-affinity group facing two identical hosts) and never changes feasibility. Note: rc5 already ranked destinations by soft-group penalty first, so drain was largely spread-aware before.
- **Config keys** (`wbalancer.conf`, validated at start-up, logged with the thresholds): `HEAL_SOFT_ANTI_AFFINITY` (default `true`), `SOFT_AA_HEAL_MAD_TOLERANCE` (default `0.05` MAD points, ≥ 0; how much a MAD already above `TARGET_AVG_MAD` may grow in the per-resource gate above), `SOFT_AA_TIEBREAK_MARGIN` (default `0.15`, 0–1). Listed in `wbalancer.conf.example`.

### Documentation

- `docs/algorithm(.vi).md`: new section 8 with worked example I (former sections 8–12 are now 9–13); section 5.3 now says the tool both gatekeeps (never worsens) and, since rc11, re-spreads broken groups; heal rows in the monitor table, the output-reading table and the constants.
- `docs/user-guide(.vi).md`: the three keys, the monitor WARNING/note semantics, the new balancing case; the "no re-spread" limitation is replaced by the heal's own limits.
- `docs/operations(.vi).md`: section 7.1, what a heal looks like in the output, the log and the events (`unresolvable` vs `deferred`, event labels), and how to turn it off.
- `docs/algorithm(.vi).md`: the VMs of the server-group example and of drain example H (and its load-compensation follow-up) are renamed `vm1`/`vm2`/`vm3`, so they no longer read like the package version "v1".
- Fixes from the 2026-11-11 documentation reviews:
  - user guide: drain exit 2 also means "every live-migratable VM was migrated but non-live-migratable VMs remain on the host"; the fail-closed rule applies to live-migratable VMs.
  - algorithm and operations: the AZ check blocks every cross-AZ move, also for VMs booted without an AZ (more conservative than Nova).
  - `docs/algorithm.vi.md`: the missing "changing thresholds" row (old values 75/85/85/0.05) of the comparison table.
  - `wbalancer.conf.example`: `COMPENSATION_CPU_WINDOW_SECONDS=60` is only the fallback; unset, the window is 2 × the detected scrape interval.

### Tests

New [`tests/test_rc11_soft_aa_heal.py`](tests/test_rc11_soft_aa_heal.py) (119 tests): violation counting (several groups and hosts, frozen inputs); source/VM choice; member-free destinations only; every filter of the chain on heal moves; the per-resource MAD gate (balanced aggregate allowed while CPU and RAM stay ≤ `TARGET_AVG_MAD`, the weighted blind spot CPU 0.5 / RAM 3.6 refused with its passing counterpart, RAM MAD 2.98 → 3.00 allowed / 3.02 refused, only `+tolerance` for a resource above the target, planner and revalidation giving the same verdict); fall-through to MAD; hotspot deferral; the balancing tie-break inside and outside the margin and below the threshold; the drain tie-break only on equal scores; monitor WARNING vs note vs disabled, and CRITICAL never masked; revalidation; events and logs (`unresolvable` only for the structural case, `deferred` with blocking counts for cooldown / lock / `missing_vm_details` / filters / budget); the event labels of a successful heal, a successful direct move (both through the real pre-submit `plan_balance=False` refresh) and a drain dry-run; config parsing, validation and defaults; `HEAL_SOFT_ANTI_AFFINITY=false` = rc10 decisions.

- Mutation checks: reverting the ladder rung, the MAD gate, the gate in the planner or in the revalidation back to a weighted-MAD comparison, the per-resource limit back to `max(before + tolerance, target)`, dropping its tolerance term, the balancing tie-break, the drain tie-break, the member-free filter, the filter chain, the severity combination, either revalidation check, the planned event labels (dropped at the refresh, taken from the given state, or removed from drain), or labelling every failure `unresolvable` makes at least one new test fail (17/17).
- Differential check against rc10 (1 500 random scenarios: direct search, MAKE_ROOM, rejection statistics, drain plan): identical with the heal off, and identical with the heal on when no soft-anti-affinity group exists.
- The 440 rc10 tests pass unchanged. Windows: 559 passed, 15 skipped. Lab canary (3 computes, 56 VMs) of the frozen build before this last round: 13/13 cases passed; it found the label bug fixed above.

## 1.0.0rc10 (2026-11-11)

Fixes for the independent review of rc9 (findings M1–M4) and the test gaps it listed.

### Breaking CLI change

- **`rolling-maint --ssh-strict-host-key` is removed.** It could never be turned off (`store_true` with default `True`), and its non-strict branch was dead code. ssh no longer gets any `StrictHostKeyChecking` / `UserKnownHostsFile` option from the tool: the coordinator's OpenSSH configuration (`~/.ssh/config`, `/etc/ssh/ssh_config`) decides. The other ssh options are unchanged (`BatchMode=yes`, `ConnectTimeout=8`, `ServerAliveInterval=5`, `ServerAliveCountMax=2`, `-i` when a key is given).
  - A command line that still passes the flag now fails with `unrecognized arguments`; drop it.
  - rc9 used `accept-new`, which recorded the key of a compute it had never seen. With OpenSSH's default policy and `BatchMode=yes`, such a compute now fails with exit 255 (`Host key verification failed`). Before upgrading, pre-populate `known_hosts` with the verified keys, or set `StrictHostKeyChecking accept-new` for the computes in `ssh_config` to keep the rc9 behaviour. See [operations](docs/operations.md#1-before-the-first-run).

### Fixed

- **M1 — a drain child could migrate on one cloud and report another.** `--os-cloud` > environment `OS_CLOUD` > `OS_CLOUD` in the config file > `openstack` is now one function (`config.resolve_os_cloud`), used for the OpenStack connection, the drain-child result identity and the `rolling-maint` parent alike.
  - rc9 built the child's identity without the config file, while the connection used it. Example: `OS_CLOUD=prod` in `wbalancer.conf`, nothing in the environment, a custom `--wbalancer`: the child migrated on `prod` and reported `openstack`, so the parent accepted the result.
  - `rolling-maint` now also reads `OS_CLOUD` from `--wlb-config` (rc9: only `--os-cloud` / environment), as the user guide already said.
  - `rolling-maint` always passes `--os-cloud` to the drain child, also to a custom `--wbalancer` child on the default cloud.
  - A cloud name that is set but blank (`--os-cloud ' '`, `OS_CLOUD='  '`) is refused; an empty `OS_CLOUD=` counts as unset. A drain child that cannot resolve its cloud writes no result, so the parent stops (outcome UNKNOWN).
- **M2 — `--monitor-only`: UNKNOWN no longer hides CRITICAL.** Aggregates are combined by severity CRITICAL > WARNING > UNKNOWN > OK instead of the numeric maximum (UNKNOWN = 3 > CRITICAL = 2). CRITICAL + UNKNOWN exits 2, WARNING + UNKNOWN exits 1, OK + UNKNOWN exits 3. Every aggregate still prints its own line.
- **M3 — revalidation used another baseline than the planner.** The snapshot now records its `exclude_unready_hosts`; the refresh before submit and the MAKE_ROOM step-2 refresh reuse the plan's value instead of the default `False` / a hard-coded `True`.
  - In rc9 a host that became unready after planning (balance excludes such hosts) stayed in the refreshed mean/MAD, the improvement gate was evaluated on a different baseline, and the move could be refused with a 30-minute rejected-move cooldown. Drain was not affected (both sides `False`).
  - If the step's own source or destination left the fresh snapshot, the refusal now names it (`destination host … is no longer part of the fresh snapshot (excluded: service disabled)`) instead of the misleading RAM-ceiling message.

### Tests

New [`tests/test_rc10_review_fixes.py`](tests/test_rc10_review_fixes.py) (63 tests), each checked to fail when its fix is reverted:

- M1: resolution order; the result identity equals the cloud the run used (config file, environment, CLI); a custom child always receives `--os-cloud`; parent and child agree end to end.
- M2: severity ranking and `monitor_by_aggregate` exit codes for mixed results.
- M3: the refresh receives the plan's `exclude_unready_hosts`; a real snapshot shows the baseline difference.
- M4: no host-key option in any ssh command line; the removed flag is refused.
- Review test gaps: the journal intent is on disk before `live_migrate_server` (and no submit when it cannot be written); snapshot fail-closed on missing, empty, out-of-range or non-finite Prometheus series, missing Placement data and fewer than 2 destination-ready hosts; the 120 s freshness gate through the real `wbalancer -y` wiring (with a positive control); `allocation_ratio` math with concrete numbers.

Windows: 440 passed, 15 skipped (rc9: 377 passed). Not yet run on a lab or on Linux.

## 1.0.0rc9 (2026-11-11)

### Changed

- **One no-progress window for migrations.**
  - `MIGRATION_TIMEOUT` (default 600 s) now means "stop waiting after this long without progress", counted from the start or from the last progress. Together with `MIGRATION_MAX_WAIT` (default 1800 s) these are the only two keys.
  - `MIGRATION_STALL_TIMEOUT` (rc8) is obsolete; it is ignored with a warning.
  - Compared with rc8: frozen from the start still stops at minute 10. Progress until minute 12 and then frozen stops at minute 22 (rc8: 17). Still progressing at minute 30 hits the cap as before.
- **Default paths drop the `-v1` suffix**, with the same layout:
  - `/var/lib/wbalancer/{vm-cooldown.json, operations/, maintenance/}`
  - `/var/log/wbalancer/{migrations.jsonl, maintenance/}`
  - `/var/lock/wbalancer-{maint,balancer}.lock`
- **State from an earlier version is never silently ignored.** When `/var/lib/wbalancer-v1` still holds a cooldown, journal or checkpoint and the new directory holds none:
  - apply runs stop before taking any lock and print the `mv` command;
  - `--dry-run` / `--monitor-only` only warn.

  See [operations](docs/operations.md#2-state-files-and-locks).

### Verified on real Nova (lab: 3 computes, 47 VMs)

Each case set `MIGRATION_TIMEOUT` (N) and `MIGRATION_MAX_WAIT` through the real config path and used the real submit, monitor, journal and reconcile code.

| Case | Setup | Result |
|---|---|---|
| A: progress keeps it alive | 32 GiB VM, N = 40 s | Ran 75.6 s (longer than N), success, not cut |
| M: cap | Same VM, N = 40 s, cap 45 s | `stopped after 46s: maximum wait reached`; next run reconciled `success` in 21 s; exactly one migration row (no resubmit) |
| B: no progress | Small VM, N = 3 s | `stopped after 8s: no progress for 3.0s`; next run reconciled `success`; one migration row |
| Q: queued behind another migration | Small VM `queued` from 11.7 s to 87.3 s behind a 32 GiB migration, N = 30 s | Not cut (the migration ahead counted as progress); success at 103.6 s |

- Both VMs ended back on their original hosts, with no active migrations left, and the production state files were unchanged.

**Path change, on a coordinator that still has rc5–rc8 state in `/var/lib/wbalancer-v1`** (final wheel):

| Command, default paths | Result |
|---|---|
| `wbalancer -y` | Exit 1: `Not running: state from an earlier version was found in /var/lib/wbalancer-v1 …`, before any lock is taken |
| `wbalancer --dry-run` | Warning, then the normal dry-run (exit 0) |
| `rolling-maint -y` | `STOPPED` with the same message, exit 1, no host touched |
| `--cooldown-file /var/lib/wbalancer-v1/vm-cooldown.json` given explicitly | No guard; runs normally |

- Nothing was created under `/var/lib/wbalancer` or `/var/log/wbalancer`.
- Linux test suite against the installed wheel: 392 passed. Windows: 377 passed, 15 skipped.

## 1.0.0rc8 (2026-11-11)

### Changed

- **The migration wait is capped at 30 minutes (was 4 hours) and is configurable.**
  - Keys: `MIGRATION_TIMEOUT` (600 s, always wait), `MIGRATION_STALL_TIMEOUT` (300 s without progress), `MIGRATION_MAX_WAIT` (1800 s cap).
  - They are validated at start-up: positive values, and `MIGRATION_MAX_WAIT` ≥ `MIGRATION_TIMEOUT`.
  - Hitting the cap does not abort the migration in Nova. The run records `migration_timeout`, keeps the journal entry pending and stops; the next run reconciles the outcome.
  - Raise `MIGRATION_MAX_WAIT` if very large VMs legitimately need longer. Example: a 256 GiB VM at an effective 150 MB/s takes about 30 minutes.
- **New [`wbalancer.conf.example`](wbalancer.conf.example)** listing every config key with its default and a short explanation. A test keeps it valid and limited to keys the code actually reads.

### Verified (lab: 3 computes, 47 VMs; read-only)

| Check | Result |
|---|---|
| Linux test suite against the installed wheel | 375 passed |
| Production config, no new keys | `migration wait: minimum 600 s, stall 300 s, maximum 1800 s` |
| `MIGRATION_MAX_WAIT=3600` | `maximum 3600 s` |
| `MIGRATION_MAX_WAIT=300` (below `MIGRATION_TIMEOUT`) | monitor exit 3, `MIGRATION_MAX_WAIT (300) must be >= MIGRATION_TIMEOUT (600).` |
| Shipped `wbalancer.conf.example` with only the 3 required lines filled in | dry-run exit 0, monitor exit 1 (WARNING, matching the lab's current imbalance), default thresholds and waits |

The cap itself is covered by unit tests (a migration reporting progress every 6 s stops being waited for at exactly 1800 s). The progress-aware wait was verified against real Nova in rc5.

## 1.0.0rc7 (2026-11-11)

### Changed

- **Decision thresholds are now config keys instead of code constants.**
  - Keys: `RAM_HOTSPOT_THRESHOLD`, `RAM_CRITICAL_THRESHOLD`, `CPU_HOTSPOT_THRESHOLD`, `PRESSURE_THRESHOLD`, `TARGET_AVG_MAD`, `MIN_IMPROVEMENT_THRESHOLD`, `PLACEMENT_PRESSURE_THRESHOLD`.
  - Defaults are unchanged: 80 / 95 / 70 / 80 / 3 / 0.15 / 90.
  - Values are validated at start-up: they must be between 0 and 100, and RAM critical must be above RAM hotspot. On an invalid value the command exits 1 (3 for `--monitor-only`) without doing anything.
  - Drain caps follow the hotspot and critical thresholds. `MAKE_ROOM_MIN_IMPROVEMENT` defaults to 2 × `MIN_IMPROVEMENT_THRESHOLD`.
  - Each run logs the thresholds in use (`--verbose`), plus an INFO line when pressure mode cannot trigger.
  - Setting `PRESSURE_THRESHOLD` below a hotspot threshold makes pressure mode reachable.
  - The old `wloadbalancer.py` values can be restored from the config (see the [migration guide](docs/migrating-from-wloadbalancer.md)).
- **The cloud no longer has to be named `openstack`.**
  - Choose it with `OS_CLOUD` (config or environment) or `--os-cloud`, for both `wbalancer` and `rolling-maint`; `rolling-maint` passes it on to its drain step.
  - With an openrc environment, use `--os-cloud envvars`.
  - The default is still `openstack`.
- The `Placement: VCPU>=…%` report line shows the configured threshold.

### Verified (lab: 3 computes, 47 VMs)

| Check | Result |
|---|---|
| Linux test suite against the installed wheel | 361 passed |
| Production config with no new keys | Thresholds 80/95/70/80/3/0.15/90; same dry-run and monitor results as rc6 |
| `PRESSURE_THRESHOLD=15` (lab average RAM 17.6%, no hotspot) | `Mode: PRESSURE (cluster average above 15.0%)` |
| `RAM_CRITICAL_THRESHOLD=70` (below the RAM hotspot) | monitor exit 3, dry-run exit 1, clear configuration error |
| `--os-cloud no-such-cloud` | monitor exit 3, `Cloud no-such-cloud was not found.` |
| A `clouds.yaml` holding only a cloud named `lab-alias` | `--os-cloud lab-alias` and `OS_CLOUD=lab-alias` both work; the default name fails with exit 3 |
| `rolling-maint --os-cloud lab-alias`, drain-reboot dry-run, 30 VMs | READY; both the `openstack` CLI and the drain child used `lab-alias` |

Planner and migration logic are otherwise unchanged from rc6.

## 1.0.0rc6 (2026-11-11)

### Changed

- **`openstacksdk` and `python-openstackclient` are no longer dependencies.** Install them yourself to match your cloud. The tested baseline is the OpenStack **2026.1** upper-constraints (openstacksdk 4.10.0, python-openstackclient 9.0.1). A copy of those constraints ships in [constraints/openstack-2026.1.txt](constraints/openstack-2026.1.txt):
  `pip install -c constraints/openstack-2026.1.txt openstacksdk python-openstackclient`
- **`requirements.lock.txt` locks only the tool's own dependencies.** They are resolved against the 2026.1 constraints and hashed (`click` 8.3.1, `requests` 2.32.5, `urllib3` 2.6.3, `PyYAML` 6.0.3, …), so installing the clients with the same constraints changes nothing.
- **Start-up checks** ([dependencies.py](src/compute_maintenance/dependencies.py), [cli/entry.py](src/compute_maintenance/cli/entry.py)):
  - `openstacksdk` missing: the command stops with install instructions, exit 1 (exit 3 for `--monitor-only`).
  - No `openstack` CLI available: `rolling-maint` stops before touching any host.
  - Client versions older than the baseline: a warning only.
- **`rolling-maint` now runs the `openstack` CLI that sits next to its own Python first;** `PATH` is only a fallback. The run log records which CLI path and version were used.
  - Before this change, `sudo`/cron reset `PATH`, so an unrelated system package could be used.
  - Observed on the lab: `/usr/bin/openstack` 6.6.0 ignores `server list --locked` and returned all 30 servers of a host where none was locked.
- The uv `dev` dependency group pins both clients to the baseline for the test suite only. It is not part of the package metadata.
- MIT [LICENSE](LICENSE) and a [migration guide](docs/migrating-from-wloadbalancer.md) for `wloadbalancer.py` users.

### Verified (lab, fresh virtualenv installed exactly as documented)

| Check | Result |
|---|---|
| Tool installed without clients | `pip check` clean. `wbalancer --dry-run` / `--monitor-only` / `rolling-maint` exit 1 / 3 / 1 and print install instructions |
| Only `openstacksdk` installed, `rolling-maint` run via `sudo` | Falls back to the system CLI 6.6.0 and warns that it is older than the baseline |
| Clients installed with the 2026.1 constraints | openstacksdk 4.10.0, OSC 9.0.1; none of the tool's dependencies changed; `pip check` clean |
| `rolling-maint` via `sudo` with an untouched `PATH` | Uses the virtualenv's CLI 9.0.1 |
| Linux test suite against the installed wheel | 340 passed |
| `rolling-maint drain-reboot --dry-run`, host with 30 VMs | READY in 57.1 s, versus 76.9 s for rc5, which had to confirm every VM's lock state with the 6.6.0 CLI (one run each) |

Planner and migration logic are unchanged from rc5. rc6 was verified with dry-runs only.

## 1.0.0rc5 (2026-11-10)

### Changed

- **Drain no longer waits a full metric window (360 s) after every VM.** Instead it adds the load of VMs that just arrived to their destination host.
  - RAM uses a fresh raw host sample; host-side RSS is the fallback. CPU uses a short decay model.
  - Measured error of the RAM estimate: within ±0.03 percentage points, versus about 2.6 points without compensation.
  - Balance runs (e.g. a 10-minute cron) keep the full wait.
- **Progress-aware migration timeout.** The tool always waits 600 s, then keeps waiting as long as Nova reports progress for the migration or for the one queued ahead of it on the same host. It gives up after 300 s without progress, or after 4 h in total.
- **The migration journal reconciles from `/os-migrations`.**
  - A failed or aborted migration is recorded, with event, e-mail and cooldown, before the journal is closed.
  - The next run closes pending entries as `success`, `failed`, `not_started` or `vm_deleted` without resubmitting.
- **Planner changes:**
  - Locked VMs, VMs with a running task, and VMs whose flavor cannot be modelled (CPU pinning, PCI, `resources:*`) are never moved.
  - `rolling-maint` stops before disabling a host that still has locked ACTIVE VMs.
  - Server groups are checked as Nova does: destination only, honouring `max_server_per_host`. Soft (anti-)affinity is respected. Availability zones are respected.
  - The destination must not become a hotspot after a move.
  - The make-room search budget is scoped to the aggregate.
- **Robustness:**
  - A data problem in one aggregate no longer stops the others; the exit code still reports it.
  - OpenStack reads are retried.
  - `PROMETHEUS_QUERY_URL` is required, and non-finite numbers in the config are rejected.
- **Profiles:**
  - `kernel-update` also reboots when the newest kernel in `/boot` is not the running one.
  - The upgrade check accepts only packages from the approved plan.
  - Before reboot, `rolling-maint` refuses if QEMU guests are still running.

### Verified (lab: 3 computes, 47 VMs, Ceph RBD)

- **Rolling maintenance:** two drain + reboot canaries (19 VMs each, about 70 s per VM) and one drain-only run (16 VMs). All VMs stayed ACTIVE; no moves outside the plan.
- **Fault injection on real Nova:** an aborted migration, a stalled migration (timeout, then reconcile to `success`), a migration queued behind another, and a locked VM.

## 1.0.0rc1–rc4

Internal pre-releases of the v1 rewrite.

## 0.x: `wloadbalancer.py`

The original single-file balancer. It is still in the repository for existing users; see the [migration guide](docs/migrating-from-wloadbalancer.md).
