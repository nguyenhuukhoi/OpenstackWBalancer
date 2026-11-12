# Creating a maintenance profile

English | [Tiếng Việt](profiles.vi.md)

The [dummy](../profiles/dummy.yaml) profile lets you test the integration without changing the host: its local `pre_drain` returns rc3, so the runner records `SKIPPED` before disable/drain, with no migration and no reboot. The runner still needs a valid cloud/inventory for preflight and still writes local logs/checkpoints. Dummy does not validate the full maintenance workflow. Use `--profile profiles/dummy.yaml` with the usual coordinator/aggregate/host options; do not switch the profile of a pending run to dummy.

Each task is a directory containing `profile.yaml` and `scripts/`. The runner takes care of the lock, checkpoints, disable, drain, reboot, health and restore; the profile describes the task-specific maintenance. Adding a task that uses the existing hooks needs no Python changes.

## Getting started

From the repository root, copy the whole template:

```powershell
Copy-Item -Recurse profile-workspace/template profile-workspace/my-maintenance
```

Linux:

```bash
cp -R profile-workspace/template profile-workspace/my-maintenance
```

Change `name`, `description`, the reboot policy and the content of the four scripts. The template deliberately stops with `PROFILE_NOT_IMPLEMENTED`; it is not a profile that runs as is. Pick a recognisable `name`, e.g. `update-multipath-config`; use a stable hook `id` and a short descriptive hook `name` so the logs are easy to read.

Check it offline from the repository root:

```bash
uv run python tools/profile_check.py profile-workspace/my-maintenance/profile.yaml
```

The checker uses the runner's own loader: it validates the schema, duplicate keys/IDs, script files, path containment, types and timings, and prints the hook plan and the settings before CLI overrides. It does not run scripts, SSH, call APIs or disable hosts. Exit 0 = schema valid and no template markers left; exit 2 = error or placeholders remain. It does not prove that the shell logic is correct or that the task is safe.

## Choosing a hook

| Hook | When | Suitable content |
|---|---|---|
| `pre_drain` | Host still enabled, before disable/drain | Check prerequisites, versions, the approved plan; decide whether there is work to do |
| `maintenance` | Host disabled/drained, baseline recorded | Change config, update packages, apply the approved change |
| `post_boot` | After the health gate, before restore; still runs when the reboot is skipped | Verify the version/config/expected result |
| `post_restore` | After enable | Acceptance checks that need the host to serve for real; a failure quarantines the host |

Put checks that can run before enable in `post_boot` whenever possible. `post_restore` runs while the host is enabled and may receive new VMs; recovery checks the inventory and does not automatically drain/reboot new workload.

## Contract to keep

| Item | Rule |
|---|---|
| Regular hook | rc0 = success; non-zero = failure |
| `pre_drain` only | rc3 = no maintenance needed, skip the host; never use it to hide an error |
| `reboot_required_cmd` | rc0 = reboot needed; rc1 = not needed; any other code = UNKNOWN |
| Recovery `check` | rc0 = completed and verified; rc10 = proven safe to run again; any other code = UNKNOWN |
| `run: ssh` | Default; runs on the compute with the runner's SSH identity |
| `run: local` | Runs on the coordinator and needs bash; it does not run on the compute |
| `idempotency: reconcile` | Requires `check`; suits operations with side effects whose result can be verified |
| `idempotency: idempotent` | Only if re-running is truly safe; without a check it does not by itself allow replaying a timeout whose outcome is UNKNOWN |
| `idempotency: never` | Default; a previous result that was not verified is never replayed automatically |

The reconciliation check is only called when the hook has a journal entry from a previous run; it is not a default precheck on the first run. A hook already recorded as `done` is skipped on resume. Check the target resource directly; a plain marker file does not prove that the update/config succeeded.

Scripts are loaded as command text and sent to the shell; they are not copied to the host as a directory. Do not use relative paths, `$0` or `source ./helper.sh` assuming you are inside the profile directory. Write self-contained scripts, or use dependencies already present at absolute paths on the target. YAML `script` and `check` must stay inside the directory that holds the profile, even after symlinks are resolved.

The `{host}`, `{address}` and `{profile}` placeholders are shell-quoted by the runner; use each as a single token without extra quotes, e.g. `printf '%s\n' {host}`. Other shell variables follow the normal Bash rules. `capture` currently supports only `expected_kernel` and `package_plan`; stdout must contain only the value to store, diagnostics go to stderr. Later hooks read `WLB_EXPECTED_KERNEL` / `WLB_PACKAGE_PLAN`. Do not assume arbitrary variables can be captured.

Use `set -euo pipefail`, but still check the exit codes of important commands explicitly. Do not write secrets into the YAML or the output. The runner currently collects hook output and logs only the last 20 lines after the command finishes; it does not stream progress line by line yet. Long waits need a suitable timeout and a recovery check.

## Deploying to the coordinator

Copy the whole reviewed directory to a dedicated path, e.g. `/etc/wbalancer/profiles/my-maintenance/`, and keep the files owned by the administrative owner. Installation and configuration: see [user-guide.md](user-guide.md).

```bash
rolling-maint --profile /etc/wbalancer/profiles/my-maintenance/profile.yaml \
  -a YOUR_AGGREGATE --host-address COMPUTE=IP --ssh-key /path/to/key \
  --prom-url http://PROMETHEUS:9090 --notify off --dry-run COMPUTE
```

The runner's `--dry-run` may run `pre_drain` hooks: only put read-only hooks there. To check the files only, use `profile_check.py`. Validate on one canary host you are allowed to use before the rollout; after review, drop `--dry-run` and use `-y` for the approved execution.

Do not edit the YAML or scripts during a pending run: the run identity includes the loaded content. Keep the profile/artifact/settings unchanged to resume. To switch profiles, finish or reconcile the old run first. Do not delete checkpoints to force a re-run.

Real examples: [drain-reboot](../profiles/drain-reboot.yaml) has no maintenance hook; [kernel-update](../profiles/kernel-update.yaml) has an approved package plan, reconciliation and kernel verification. The package upgrade profile has not yet been validated by a real upgrade; try it on a lab first.
