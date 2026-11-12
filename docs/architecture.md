# Architecture

English | [Tiếng Việt](architecture.vi.md)

This document describes the final architecture of the `compute-maintenance` package (commands `wbalancer` and `rolling-maint`) as of 1.0.0rc13. It is written for developers and AI agents who maintain or extend the codebase: it states where code belongs, the invariants that must never break, and how to add things the intended way. What the algorithm *decides* is specified in [algorithm.md](algorithm.md); this document covers how the code is *organised*.

## 1. Purpose and operating premise

One Linux coordinator host runs two CLI tools against an OpenStack cloud: `wbalancer` keeps the compute hosts of an aggregate balanced and keeps soft-anti-affinity groups spread, one live migration per aggregate per run; `rolling-maint` takes compute hosts through disable → drain → maintain → reboot → restore, one host at a time. Both are designed for cron plus Icinga, not for daemons.

Deliberate constraints, do not "fix" them:

- **Single coordinator.** All locks are local `fcntl` locks (`/var/lock/wbalancer-*.lock`). There is no distributed coordination and none is wanted.
- **One move per aggregate per run.** Convergence is spread over cron ticks; this bounds blast radius and keeps every run explainable.
- **Fail closed.** Missing or implausible input data (Prometheus series, Placement records, group reads) aborts the decision, never degrades it silently.
- **The tool never worsens a policy.** Hard policies follow Nova filter semantics; soft policies are gatekept (a move may not increase a group's penalty) and, since rc11, actively healed.

## 2. Layering

```
cli/            argument parsing, wiring, exit codes        (imports everything below)
presentation/   human-readable report text                  (imports application, domain)
application/    orchestration: runs, rounds, state, journal (imports domain, adapters)
adapters/       all I/O: OpenStack SDK/REST, Prometheus,
                files, locks, ssh, mail                     (imports domain, config)
domain/         pure decision logic, no I/O at all          (imports config only)
config.py       thresholds, keys, parsing; paths.py: file locations
```

Rules that keep this healthy:

1. `domain/` must stay import-clean: no SDK, no requests, no file I/O, no clock reads beyond what is passed in. Everything in `domain/` is unit-testable with plain dicts.
2. `application/` owns sequencing and state; it never talks HTTP itself — it calls `adapters/`.
3. `adapters/` never decides; it fetches, parses fail-closed, and returns neutral structures.
4. Data crossing a layer boundary is a dataclass: an evaluated aggregate is a `PlanContext` (`domain/plan_context.py`), read by attribute; new code must not add stringly-typed dict state (see §9).
5. Dependency direction is one-way, downwards in the list above. If an import wants to go the other way, the design is wrong.

Module map (largest first): `application/migrations.py` plan execution, revalidation, progress monitoring, journal glue; `config.py` keys, parsing, `RuntimeSettings`; `application/snapshots.py` cluster snapshot, arrival compensation, builds the `PlanContext`; `adapters/openstack.py`; `presentation/reports.py`; `application/balancing.py` balance/drain rounds; `application/stages.py` rolling-maint stage actions; `domain/drain.py` fail-closed drain packing; `adapters/prometheus.py`; `application/recovery.py` journal reconcile and settle; `application/maintenance.py` rolling-maint runner; package `domain/balance/` (rc12 split of the former `balance.py`, all names re-exported): `makeroom.py` MAKE_ROOM, `filters.py` candidate eligibility + destination filter chain + hotspot guards, `heal.py` soft-anti-affinity heal, `scoring.py` direct-move search and tie-breaks, `modes.py` mode ladder, baseline, thresholds, `classify.py` detail status and reason; `domain/plan_context.py` `PlanContext` / `PlanLabels`; `domain/severity.py` severity and exit-code mapping (single source).

### 2.1 Where to change what

| You want to change… | Open |
|---|---|
| A threshold, timeout or config key | `config.py` (key group, `RuntimeSettings` field if live; + `wbalancer.conf.example`, user-guide 4.2) |
| Mode ladder, baseline, improvement thresholds | `domain/balance/modes.py` |
| Candidate eligibility, destination filter chain, hotspot guards | `domain/balance/filters.py` |
| Direct-move scoring, tie-breaks | `domain/balance/scoring.py` |
| MAKE_ROOM two-step plan | `domain/balance/makeroom.py` |
| Heal selection and its MAD gate | `domain/balance/heal.py` |
| Detail status and reason of a snapshot | `domain/balance/classify.py` |
| Severity, Icinga exit code, combining exit codes | `domain/severity.py` |
| What an evaluated aggregate carries across layers | `domain/plan_context.py` (built in `application/snapshots.py`) |
| Server-group / AZ / capacity-fit rules | `domain/policies.py` |
| Drain packing (orderings, destination choice) | `domain/drain.py` |
| Migration execution, revalidation, progress timeout | `application/migrations.py` |
| Snapshot contents, metric plausibility, arrival compensation | `application/snapshots.py` (parsing: `adapters/prometheus.py`) |
| Balance/drain round flow, run-level exit codes | `application/balancing.py` |
| `--monitor-only` output | `application/monitoring.py` (merge rule: `domain/severity.py`) |
| Journal reconcile, settle, UNKNOWN handling | `application/recovery.py` |
| A rolling-maint stage's behaviour | `application/stages.py` (sequencing/checkpoints: `application/maintenance.py`) |
| An OpenStack API call | `adapters/openstack.py` |
| ssh/scp invocation of a compute | `adapters/commands.py` |
| Human-readable report wording | `presentation/reports.py` |
| JSONL event fields | emitters in `application/balancing.py` / `migrations.py` |
| A CLI flag | `cli/balance.py` or `cli/maintain.py` |

## 3. Anatomy of a balance run

`cli/balance.py` wires one run: coordinator/balancer lock (apply mode only; `--dry-run` and `--monitor-only` take no lock), journal only in apply mode, then per aggregate:

1. **Snapshot** (`application/snapshots.py`): Nova servers + hypervisors, Placement inventories/usages via raw REST, Prometheus node and libvirt series. Builds the `PlanContext`: `collected_at`, `exclude_unready_hosts`, per-host CPU/RAM%, per-resource baseline MADs, plan and verdict. Any missing host metric ⇒ fail closed for that aggregate (a `PlanContext` with only `error`).
2. **Mode ladder** (`domain/balance/modes.py`, heal rung `heal.py`, constants `MODE_*` in `config.py`): `ram_critical` (>95%) → `ram_hotspot` (>80%) → `cpu_hotspot` (>70%) → `soft_aa_heal` (broken soft-anti-affinity group, rc11) → `mad` balancing (per-resource MAD > `TARGET_AVG_MAD`), plus `pressure` for high cluster mean. Heal never blocks the rung below: no feasible heal ⇒ fall through to MAD.
3. **Plan**: one `DIRECT` move, one two-step `MAKE_ROOM` plan, or one `SOFT_AA_HEAL` move. Every candidate passes the same filter chain: Placement capacity with headroom, no new hotspot, availability zone, server-group policies, per-VM cooldowns, `max_server_per_host`.
4. **Revalidate** (`application/migrations.py`): fresh snapshot with the *same* `exclude_unready_hosts` flag as the plan; snapshot older than 120 s ⇒ reject and re-plan next run; heal moves re-pass the per-resource MAD gate.
5. **Execute**: `journal.begin()` is persisted **before** `live_migrate_server` is submitted — this ordering is an invariant with a dedicated test. Progress is watched with a merged window: stop after `MIGRATION_TIMEOUT` (600 s) without progress, hard cap `MIGRATION_MAX_WAIT` (1800 s).
6. **Settle**: balance waits a 360 s barrier; drain instead compensates arrivals using a measured raw RAM sample (RSS fallback, then decay model).
7. **Record**: outcome to the journal, a JSONL event to `/var/log/wbalancer/migrations.jsonl` (labels of the snapshot the plan was made from; `mode`/`plan_type` identify heal moves; groups without a heal move emit `soft_aa_heal_unresolvable` (structural) or `soft_aa_heal_deferred` (temporary)), cooldown for the moved VM, Nagios exit code.

An interrupted run is reconciled on the next run from the write-ahead journal (`application/recovery.py`): per-instance `/os-migrations` plus a `global_max_id` floor classify each pending entry as success / failed / not_started / vm_deleted / in_progress; UNKNOWN blocks new submits and is never auto-retried.

## 4. Invariants (never break these)

1. **Journal before submit.** No live migration may be requested before its journal entry is durable.
2. **Fail closed** on any missing/implausible metric, Placement record, or server-group read.
3. **Atomic persistence**: every state file is written `mkstemp` → `fsync` → `os.replace` → directory fsync (`adapters/persistence.py`). Never write state any other way.
4. **Freshness gate**: no submit from a snapshot older than 120 s. A read may be reused only inside one snapshot or one revalidation pass (rc13: one hypervisor map, one read of the step's VM), never across a settle barrier or a refresh; a cluster cache is prefetched only for a caller that reads it (`snapshots.prefetched_cluster_cache_is_used`). Keystone project names (display only) are resolved once per run.
5. **Gatekeeper**: no balance/make-room/heal move may increase any soft group's penalty; hard policies follow Nova filter semantics (destination-based, per `domain/policies.py`).
6. **Heal MAD gate** (per resource, CPU and RAM separately): post-move MAD ≤ `TARGET_AVG_MAD` if the baseline was at or below it, else baseline + `SOFT_AA_HEAL_MAD_TOLERANCE`. Heal can never newly unbalance an aggregate.
7. **Exit codes are a contract**: monitor 0/1/2/3 (Nagios) combined with `worst_exit_code` ranking CRITICAL > WARNING > UNKNOWN > OK, all defined once in `domain/severity.py`; drain 0/1/2. Changing them breaks operators' Icinga.
8. **One move per aggregate per run**; per-VM cooldowns apply to every executed or rejected move.
9. **The AZ check is deliberately conservative**: all cross-AZ moves are blocked, even for VMs not pinned to an AZ.
10. **Identity is resolved once**: cloud selection is `--os-cloud` > env `OS_CLOUD` > config-file `OS_CLOUD` > `openstack`, through `config.resolve_os_cloud()` only. Never resolve it ad hoc; the drain child always receives an explicit `--os-cloud`.

## 5. State on disk (`paths.py` is the single source of truth)

| Path | Contents |
|---|---|
| `/etc/wbalancer/wbalancer.conf` | operator config (`KEY=value`); env `WLB_CONFIG_PATH` replaces this default |
| `/var/lib/wbalancer/vm-cooldown.json` | per-VM cooldowns |
| `/var/lib/wbalancer/operations/migration-journal.json` | write-ahead migration journal |
| `/var/lib/wbalancer/maintenance/` | rolling-maint checkpoints |
| `/var/log/wbalancer/migrations.jsonl` | one JSON event per decision/migration |
| `/var/log/wbalancer/maintenance/` | rolling-maint logs and results |
| `/var/lock/wbalancer-balancer.lock`, `-maint.lock` | local coordinator locks |

A populated legacy `/var/lib/wbalancer-v1` with an empty new dir refuses apply (`legacy_state_message`) so state is migrated consciously.

## 6. Configuration system

Keys live in `config.py` in three groups: `THRESHOLD_KEYS` (`RAM_HOTSPOT_THRESHOLD`, `RAM_CRITICAL_THRESHOLD`, `CPU_HOTSPOT_THRESHOLD`, `PRESSURE_THRESHOLD`, `TARGET_AVG_MAD`, `MIN_IMPROVEMENT_THRESHOLD`, `PLACEMENT_PRESSURE_THRESHOLD`), `SOFT_AA_HEAL_KEYS` (`HEAL_SOFT_ANTI_AFFINITY`, `SOFT_AA_HEAL_MAD_TOLERANCE`, `SOFT_AA_TIEBREAK_MARGIN`), `TIMEOUT_KEYS` (`MIGRATION_TIMEOUT`, `MIGRATION_MAX_WAIT`). Values come from the config file and/or environment, are validated, then applied process-wide by the CLI via `config.apply_thresholds` / `apply_soft_aa_heal` / `apply_migration_timeouts`, which replace the live `RuntimeSettings`.

**Canonical layer (rc12)**: the live values are one frozen `RuntimeSettings` object; code reads it at call time, `_config.runtime().ram_hotspot_threshold`. The upper-case constants (`RAM_HOTSPOT_THRESHOLD` …, see `RUNTIME_CONSTANTS`) are only a compatibility layer: every `apply_*` rewrites them, and assigning one (older code, tests via `monkeypatch`) is forwarded into `runtime()`, so the two never disagree. The trap that remains: `from compute_maintenance.config import X` copies the value at import time and silently ignores operator config. Never import config values by name (`tests/test_rc12_refactor.py` enforces it for `src/`), and do not add new readers of the constants.

`wbalancer.conf.example` must list every real key and nothing else — a test enforces that it parses and validates.

## 7. rolling-maint stage machine

`application/maintenance.py` runs one host at a time through `application/stages.py`: `preflight → pre_drain → disable → drain → verify → baseline → maintenance → reboot → health → post_boot → restore → post_restore`, with durable checkpoints (resume by re-running the same command), checkpoint identity validation, and quarantine (host left disabled) when post-restore fails. The drain is a child `wbalancer --drain-host` process that inherits the coordinator lock FD (`pass_fds`, verified by dev/ino) and always receives `--os-cloud`. SSH host-key policy is whatever the coordinator's OS ssh configuration says — the tool passes no `StrictHostKeyChecking` options (rc10 removed the flag).

## 8. Testing bars (what "done" means here)

- The suite is offline and hermetic by design (no cloud, no network, never the host's root-only `/etc/wbalancer/wbalancer.conf`: the root `conftest.py` points `WLB_CONFIG_PATH` at a temp path) and must stay green: on Linux everything runs; on Windows the POSIX-lock and apt tests skip. Release verification runs the suite on Linux at least once.
- A behavior fix ships with a test that fails on the old behavior (rc10/rc11 practice: mutation checks — temporarily revert the change, confirm a test fails, restore byte-identical).
- A risky refactor is verified by differential sampling against the previous build (generate N random scenarios, compare decisions; rc11 used 1500 with 0 unintended diffs, rc12 1500 per configuration plus byte-identical report text and events). `ruff check src tests` stays at zero.
- Docs ship in EN/VI pairs with matching structure; CLI behavior changes update `user-guide`, `operations`, `algorithm` and `wbalancer.conf.example` in the same change.
- Icinga/exit-code semantics and ladder-order changes always get explicit tests.

## 9. How to extend (follow these recipes)

**Add a config key**: constant + entry in the right `*_KEYS` group in `config.py`, parsing + validation + apply function wiring (a process-wide live value gets a `RuntimeSettings` field read through `runtime()`), `wbalancer.conf.example` comment, `user-guide` 4.2 row (EN+VI), tests for default/parse/validate/apply.

**Add a balancing objective**: implement pure logic in `domain/` (a new module in `domain/balance/`, re-exported by its `__init__`; keep modules well under ~800 lines), insert its rung explicitly in the ladder with a documented position rationale, reuse the existing destination filter chain (`filters.check_direct_destination`) rather than writing new checks, emit a distinct `mode` in events, map its severity in `domain/severity.py`, and document it in `algorithm.md` with a worked numeric example. Default-on changes of behavior need a disable key.

**Add an adapter**: new module in `adapters/`, fail-closed parsing, no decisions, retries via the existing helpers; inject it from `application/`, never import it from `domain/`.

**Known debt (remaining after rc12, do not grow it)**: inside `PlanContext` the nested results `soft_aa_heal`, `load_status` and `placement_summary`, and the `compute_baseline_metrics` result, are still string-keyed dicts; the `RuntimeSettings` mirror constants and the `PlanContext` item access exist only for compatibility (new code uses attributes and `runtime()`). rc12 removed the untyped state dict, the `config.py` globals mutation, the single 1 300-line `balance.py` and the duplicated severity mapping.

## 10. Reference history

Independent review and fix reports (rc10: M1–M4; rc11: heal design, gate formula evolution, lab canary) live outside the repo in the operator's workspace; the distilled outcomes are in [CHANGELOG.md](../CHANGELOG.md). The design precedent for healing is VMware DRS constraint correction; the gatekeeper semantics mirror Nova's scheduler filters.
