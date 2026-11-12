import compute_maintenance.domain.models as _models
from compute_maintenance.application.diagnostics import (
    build_migration_monitor_result,
    format_migration_monitor_detail,
    snapshot_action_ids,
)

"""execution.migrate — extracted domain responsibilities."""
import logging
import time
from datetime import datetime
from types import SimpleNamespace

import compute_maintenance.adapters.cooldown as _execution_cooldown
import compute_maintenance.adapters.notifications as _notify
import compute_maintenance.adapters.openstack as _sources_openstack
import compute_maintenance.application.snapshots as _snapshot
import compute_maintenance.config as _config
import compute_maintenance.domain.balance as _planning_balance
import compute_maintenance.domain.errors as _errors
import compute_maintenance.domain.plan_context as _plan_context
import compute_maintenance.domain.policies as _planning_constraints
import compute_maintenance.presentation.reports as _report


def reset_error_server_to_active(conn, server_id, vm_name, expected_source_host=None):
    try:
        server = conn.compute.get_server(server_id)
    except Exception as e:  # noqa: BLE001 - state check after a failure is best effort (logged)
        logging.warning(
            "Failed to check VM %s (%s) state after migration failure: %s",
            vm_name,
            server_id,
            e,
        )
        return False

    status = get_server_status(server)
    if status != "ERROR":
        logging.debug(
            "VM %s (%s) state after migration failure is %s; no reset needed.",
            vm_name,
            server_id,
            status or "UNKNOWN",
        )
        return False

    current_host = get_server_current_host(server)
    task_state = _models.get_resource_field(server, "task_state", "OS-EXT-STS:task_state")
    power_state = _models.get_resource_field(server, "power_state", "OS-EXT-STS:power_state")
    if expected_source_host and current_host != expected_source_host:
        logging.warning(
            "Refusing to reset VM %s (%s) from ERROR: current host %r is not "
            "the expected source %r.",
            vm_name,
            server_id,
            current_host,
            expected_source_host,
        )
        return False
    if task_state not in (None, "", "None"):
        logging.warning(
            "Refusing to reset VM %s (%s) from ERROR while task_state=%r.",
            vm_name,
            server_id,
            task_state,
        )
        return False
    if str(power_state).lower() not in ("1", "running"):
        logging.warning(
            "Refusing to reset VM %s (%s) from ERROR because power_state=%r "
            "does not prove the guest is running.",
            vm_name,
            server_id,
            power_state,
        )
        return False

    try:
        conn.compute.reset_server_state(server, state="active")
        logging.warning(
            "VM %s (%s) was in ERROR after migration failure; reset state to ACTIVE.",
            vm_name,
            server_id,
        )
        return True
    except Exception as e:  # noqa: BLE001 - the reset is best effort (logged)
        logging.warning(
            "Failed to reset VM %s (%s) state from ERROR to ACTIVE: %s",
            vm_name,
            server_id,
            e,
        )
        return False


from compute_maintenance.domain.server import get_server_current_host, get_server_status


def get_migration_identifier(migration):
    """Return the migration identifier regardless of resource shape."""
    if isinstance(migration, dict):
        return migration.get("id") or migration.get("uuid")
    return getattr(migration, "id", None) or getattr(migration, "uuid", None)


def get_migration_status(migration):
    """Return the migration status regardless of resource shape."""
    if migration is None:
        return None
    return getattr(migration, "status", None) or (
        migration['status'] if isinstance(migration, dict) and 'status' in migration else 'n/a'
    )


def find_tracked_migration(migrations, migration_id):
    """Return only the explicitly tracked migration.

    Falling back to an old ``migrations[-1]`` row creates a false-success race
    while Nova has not exposed the newly submitted record yet.
    """
    if migration_id:
        for migration in migrations:
            if str(get_migration_identifier(migration)) == str(migration_id):
                return migration
    return None


def get_migration_progress(server, migration):
    """Return the best available migration progress percentage."""
    if hasattr(server, 'progress') and server.progress is not None:
        return server.progress
    if hasattr(migration, 'progress') and migration.progress is not None:
        return migration.progress
    if isinstance(migration, dict) and 'progress' in migration and migration['progress'] is not None:
        return migration['progress']
    return None


def map_migration_state(status):
    """Map raw migration status to the existing display state labels."""
    normalized_status = status.lower() if status else ""
    if normalized_status == "running (post-copy)":
        return "IN PROGRESS"
    if normalized_status in _config.PROGRESS_STATUSES:
        return "IN PROGRESS"
    if normalized_status in _config.SUCCESS_STATUSES:
        return "SUCCESS"
    if normalized_status in _config.FAILED_STATUSES:
        return "FAILED"
    return f"UNKNOWN ({status})"


def get_server_migration_result(vm_name, current_host, server_status, src_host=None, dst_host=None, verbose=True):
    """Check server placement/state shortcuts that indicate migration completion or failure."""
    if dst_host and current_host == dst_host and server_status == "ACTIVE":
        if verbose:
            print(f"\nMigration completed: VM {vm_name} is now ACTIVE on {dst_host}.")
        return _config.MIGRATION_RESULT_SUCCESS

    if (not dst_host and src_host and current_host != src_host and current_host not in ("", None, "UNKNOWN")
            and server_status == "ACTIVE"):
        if verbose:
            print(f"\nMigration completed: VM {vm_name} moved from {src_host} to {current_host}.")
        return _config.MIGRATION_RESULT_SUCCESS

    if server_status == "ERROR":
        if verbose:
            print(f"\nMigration failed: VM {vm_name} entered ERROR state.")
        return _config.MIGRATION_RESULT_FAILED

    return None


def update_active_stuck_state(
    vm_name,
    current_host,
    server_status,
    src_host,
    active_stuck_start,
    active_stuck_timeout,
    verbose=True,
):
    """Preserve the current stuck-on-source detection behavior."""
    if src_host and current_host == src_host and server_status == "ACTIVE":
        if active_stuck_start is None:
            return time.monotonic(), None
        if time.monotonic() - active_stuck_start >= active_stuck_timeout:
            if verbose:
                print(
                    f"\nMigration did not start: VM {vm_name} stayed ACTIVE on {src_host} "
                    f"for {active_stuck_timeout} seconds."
                )
            return active_stuck_start, _config.MIGRATION_RESULT_STUCK_ACTIVE
        return active_stuck_start, None

    return None, None


def lookup_global_migration_result(conn, migration_id, verbose=True, server_id=None):
    """Check /os-migrations for a tracked migration result.

    With ``server_id`` only that server's rows are listed (instead of the whole
    cloud's history on every poll).
    """
    if not migration_id:
        return False, None

    rows = (_sources_openstack.server_migration_rows(conn, server_id)
            if server_id else conn.compute.migrations())
    for migration in rows:
        current_id = get_migration_identifier(migration)
        if str(current_id) != str(migration_id):
            continue

        status = get_migration_status(migration)
        normalized_status = status.lower() if status else ""
        if verbose:
            print(f"\nMigration {migration_id} found in global migrations with status: {status}")

        if normalized_status == "running (post-copy)":
            if verbose:
                print(f"Migration {migration_id} is running in post-copy (global record).")
            return True, None
        if normalized_status in _config.SUCCESS_STATUSES:
            if verbose:
                print(
                    f"Migration {migration_id} is complete in the global record; "
                    "waiting for Nova to confirm ACTIVE on the expected destination."
                )
            return True, None
        if normalized_status in _config.FAILED_STATUSES:
            if verbose:
                print(f"Migration {migration_id} failed (global record).")
            return True, _config.MIGRATION_RESULT_FAILED
        if verbose:
            print(f"Migration {migration_id} is in state '{status}' (global record).")
        return True, None

    return False, None


def find_new_global_migration(conn, server_id, global_max_id):
    """Newest live-migration row of ``server_id`` with id > ``global_max_id``."""
    newest = None
    for row in _sources_openstack.server_migration_rows(conn, server_id):
        kind = str(_models.get_resource_field(row, "migration_type") or "live-migration").lower()
        if kind != "live-migration":
            continue
        try:
            row_id = int(get_migration_identifier(row))
        except (TypeError, ValueError):
            continue
        if row_id > global_max_id and (newest is None or row_id > int(get_migration_identifier(newest))):
            newest = row
    return newest


def migration_progress_signature(migration):
    """Fields Nova updates while a live migration makes progress."""
    if migration is None:
        return None
    return tuple(str(_models.get_resource_field(migration, name)) for name in (
        "status", "memory_processed_bytes", "memory_remaining_bytes",
        "disk_processed_bytes", "disk_remaining_bytes", "updated_at"))


def host_migration_activity(conn, host):
    """Signature of running migrations touching ``host`` (the one ahead in the queue)."""
    if not host:
        return None
    rows = conn.compute.migrations(host=host)
    return tuple(sorted(
        (str(get_migration_identifier(r)), str(get_migration_status(r)),
         str(_models.get_resource_field(r, "updated_at")))
        for r in rows
        if str(get_migration_status(r) or "").lower() not in _config.SUCCESS_STATUSES | _config.FAILED_STATUSES
        and str(get_migration_status(r) or "").lower() != "reverted"
    ))


def monitor_migration(
    conn,
    server_id,
    vm_name,
    src_host=None,
    dst_host=None,
    timeout=None,
    verbose=True,
    preexisting_migration_ids=None,
    preexisting_action_request_ids=None,
    global_max_id=None,
    stall_timeout=None,
    max_wait=None,
):
    """Poll until Nova confirms the VM ACTIVE on the expected destination.

    Migration rows that existed before submission are ignored.  A successful
    migration record is useful progress evidence, but it is not the terminal
    condition because Nova/Placement views may still lag behind it.
    """
    if timeout is None:
        timeout = _config.runtime().migration_timeout
    # One no-progress window: the same value is the minimum wait and the stall
    # limit, so "no progress for `timeout` s, from the start or since the last
    # progress" ends the wait. stall_timeout only overrides it in tests.
    timer = _errors.ProgressTimer(
        timeout,
        timeout if stall_timeout is None else stall_timeout,
        max(timeout, _config.runtime().migration_max_wait if max_wait is None else max_wait),
        time.monotonic,
    )
    host_activity = None
    host_activity_at = None

    def progress_expired(mig):
        """Observe progress, then decide; host activity is polled sparingly."""
        nonlocal host_activity, host_activity_at
        now_mono = time.monotonic()
        if host_activity_at is None or now_mono - host_activity_at >= _config.MIGRATION_HOST_ACTIVITY_INTERVAL:
            host_activity_at = now_mono
            try:
                host_activity = host_migration_activity(conn, src_host)
            except Exception:  # noqa: BLE001 - progress evidence is best effort
                host_activity = None
        timer.observe((migration_progress_signature(mig), host_activity))
        reason = timer.expired()
        if reason and verbose:
            print(f"\nMigration monitoring stopped after {timer.elapsed():.0f}s: "
                  f"{'no progress for ' + str(timer.stall) + 's' if reason == 'stalled' else 'maximum wait reached'}.")
        return reason

    active_stuck_timeout = _config.MIGRATION_RECORD_DISCOVERY_TIMEOUT
    preexisting_ids = {str(x) for x in (preexisting_migration_ids or set())}
    orig_host = src_host
    dest_host = dst_host

    last_state = None
    active_stuck_start = None
    migration_id = None

    migration, server = None, None
    def outcome(value):
        return build_migration_monitor_result(value, conn=conn, server_id=server_id,
            migration=migration, server=server, src_host=src_host, dst_host=dst_host,
            preexisting_action_request_ids=preexisting_action_request_ids)

    while True:
        try:
            migrations = list(conn.compute.server_migrations(server_id))

            if migration_id is None:
                new_migrations = [
                    migration
                    for migration in migrations
                    if get_migration_identifier(migration) is not None
                    and str(get_migration_identifier(migration)) not in preexisting_ids
                ]
                migration = new_migrations[-1] if new_migrations else None
                if migration is not None:
                    migration_id = get_migration_identifier(migration)
                    orig_host = _models.get_resource_field(migration, "source_node") or src_host
                    dest_host = _models.get_resource_field(migration, "dest_node") or dst_host
                elif global_max_id is not None:
                    # A migration that fails its pre-checks (or finishes) before
                    # the first poll never shows up in the in-progress-only
                    # per-server list; /os-migrations still has the row.
                    global_row = find_new_global_migration(conn, server_id, global_max_id)
                    if global_row is not None:
                        migration_id = get_migration_identifier(global_row)
                        status = (get_migration_status(global_row) or "").lower()
                        if verbose:
                            print(f"Migration ID: {migration_id} (global record, status {status})")
                        if status in _config.FAILED_STATUSES:
                            migration = global_row
                            server = conn.compute.get_server(server_id)
                            return outcome(_config.MIGRATION_RESULT_FAILED)
            else:
                migration = find_tracked_migration(migrations, migration_id)

            if migration_id is not None and last_state is None and migration is not None and verbose:
                print(f"Migration ID: {migration_id}")
                print(f"Status: {get_migration_status(migration)}")
                print(f"Source Server: {orig_host}")
                print(f"Destination Server: {dest_host}\n")

            mig = migration

            server = conn.compute.get_server(server_id)
            current_host = get_server_current_host(server)
            server_status = get_server_status(server)
            server_result = get_server_migration_result(
                vm_name,
                current_host,
                server_status,
                src_host=src_host,
                dst_host=dst_host,
                verbose=verbose,
            )
            if server_result:
                return outcome(server_result)

            if not mig:
                active_stuck_start, stuck_result = update_active_stuck_state(
                    vm_name,
                    current_host,
                    server_status,
                    src_host,
                    active_stuck_start,
                    active_stuck_timeout,
                    verbose=verbose,
                )
                if stuck_result:
                    return outcome(stuck_result)
            else:
                # A queued/preparing/running record proves Nova accepted the
                # operation; the VM may legitimately stay ACTIVE on source
                # for longer than the discovery grace period.
                active_stuck_start = None

            if not mig:
                _global_found, global_result = lookup_global_migration_result(
                    conn,
                    migration_id,
                    verbose=verbose,
                    server_id=server_id,
                )
                if global_result:
                    return outcome(global_result)
                # Per-server and global migration rows can disappear at
                # different times.  Keep polling the authoritative server
                # placement instead of declaring a false failure immediately.

                if progress_expired(None):
                    return outcome(_config.MIGRATION_RESULT_TIMEOUT)

                time.sleep(_config.POLL_INTERVAL)
                continue

            percent = get_migration_progress(server, mig)
            progress_str = f"{percent}%" if percent is not None else "?"
            status = get_migration_status(mig)
            state = map_migration_state(status)

            state_changed = state != last_state
            if state_changed and verbose:
                print()

            if verbose:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Migration {migration_id} | VM: {vm_name} | "
                    f"State: {state} | Progress: {progress_str}   ",
                    end="\r",
                    flush=True
                )
            last_state = state

            if state == "SUCCESS" and state_changed and verbose:
                print(
                    f"\nMigration {migration_id} completed in the migration API; "
                    f"waiting for VM {vm_name} to become ACTIVE on {dst_host}."
                )
            if state == "FAILED":
                if verbose:
                    print(f"\nMigration {migration_id} failed!")
                return outcome(_config.MIGRATION_RESULT_FAILED)

            if progress_expired(mig):
                return outcome(_config.MIGRATION_RESULT_TIMEOUT)

            time.sleep(_config.POLL_INTERVAL)

        except Exception as e:  # noqa: BLE001 - API errors are retried until the no-progress window expires
            if verbose:
                print(f"\nMonitoring error or migration disappeared (will retry): {e}")
            # API errors are not progress: only the unchanged signature counts.
            if timer.expired():
                return outcome(_config.MIGRATION_RESULT_TIMEOUT)
            time.sleep(_config.POLL_INTERVAL)


def get_server_migration_ids(conn, server_id):
    """Snapshot migration IDs immediately before submitting a new operation."""
    try:
        migrations = list(conn.compute.server_migrations(server_id))
    except Exception as e:  # noqa: BLE001 - any SDK/HTTP error -> OpenStackDataError (fail closed)
        raise _errors.OpenStackDataError(
            f"failed to snapshot existing migrations for server {server_id}: {e}"
        )
    return {
        str(migration_id)
        for migration_id in (get_migration_identifier(m) for m in migrations)
        if migration_id is not None
    }


def is_definitive_submission_rejection(error):
    """Return True only when an HTTP 4xx proves Nova rejected the request."""
    status_code = getattr(error, "status_code", None) or getattr(error, "http_status", None)
    try:
        status_code = int(status_code)
    except (TypeError, ValueError):
        return False
    return 400 <= status_code < 500


def validate_migration_step(
    cfg,
    conn,
    state,
    step,
    enforce_server_groups=True,
    require_runtime_improvement=False,
    prevent_new_hotspot=True,
    enforce_soft_groups=False,
    soft_aa_heal=False,
):
    """Revalidate an exact planned step against fresh Nova + Placement state.

    ``require_runtime_improvement`` controls whether this individual step must
    improve the current weighted MAD. MAKE_ROOM step 1 intentionally sets it
    to False because its purpose is to release Placement capacity for step 2.

    ``prevent_new_hotspot`` is independent: even a preparatory MAKE_ROOM step
    must be rejected if fresh runtime metrics show that the destination would
    cross a CPU/RAM hotspot threshold.

    ``soft_aa_heal`` (rc11) revalidates a heal move: on the fresh topology it
    must still remove a soft-anti-affinity violation, and on fresh metrics
    the CPU MAD and the RAM MAD must each stay within the planner's limit:
    TARGET_AVG_MAD, or baseline + SOFT_AA_HEAL_MAD_TOLERANCE when already
    above it (domain.balance.soft_aa_heal_mad_check).
    """
    try:
        server = conn.compute.get_server(step.vm_id)
    except Exception as e:  # noqa: BLE001 - refuses the step before anything is submitted
        return False, f"failed to fetch VM before migration: {e}"
    if not server:
        return False, "VM no longer exists"

    status = get_server_status(server)
    if status != _config.ACTIVE_VM_STATUS:
        return False, f"VM is not ACTIVE (status={status})"

    task = _models.get_resource_field(server, "task_state", "OS-EXT-STS:task_state")
    if task not in (None, "", "None"):
        return False, f"VM has an active task ({task})"
    if _models.get_resource_field(server, "is_locked", "locked") not in (None, False, "False", "false"):
        return False, "VM is locked"

    current_host = get_server_current_host(server)
    expected_hosts = {step.src_host}
    # One service-host -> hypervisor-name map serves this whole validation: the
    # source alias here, the destination's Placement provider name and the
    # server-group refresh below. The mapping only changes when a compute is
    # re-registered, and all three reads happen within this one pass.
    host_to_hypervisor = _sources_openstack.build_nova_host_to_hypervisor_map(
        conn, [step.src_host, step.dst_host]
    )
    try:
        expected_hosts.add(host_to_hypervisor[step.src_host])
    except Exception:  # noqa: BLE001, S110 - the hypervisor alias is optional; the Nova host name still matches
        pass
    if not current_host:
        return False, "VM current host is unavailable from Nova"
    if current_host not in expected_hosts:
        return False, (
            f"VM moved since planning (expected one of {sorted(expected_hosts)}, "
            f"current={current_host})"
        )

    ready, reason = _sources_openstack.is_nova_compute_service_ready(conn, step.dst_host)
    if not ready:
        return False, f"destination not ready: {reason}"

    details = (state.all_vm_details or {}).get(step.vm_id)
    if not details:
        return False, "VM details missing from refreshed aggregate state"

    host_az = state.host_az
    if host_az is not None and not _planning_balance.availability_zone_allows(
        SimpleNamespace(host_az=host_az), details, step.src_host, step.dst_host,
    ):
        return False, "availability_zone (VM {}, destination {})".format(
            details.get("availability_zone"), host_az.get(step.dst_host))

    destination_hypervisor = host_to_hypervisor.get(step.dst_host)
    fresh_placement = _sources_openstack.fetch_placement_host_resource(
        conn,
        cfg,
        step.dst_host,
        provider_name=destination_hypervisor,
    )
    vcpus = int(details.get("vcpus", step.vcpus) or step.vcpus)
    ram_mb = round(float(details.get("ram_gb", step.vm_ram_gb) or step.vm_ram_gb) * 1024.0)
    fits, fit_reason = _planning_constraints.placement_can_fit(fresh_placement, vcpus, ram_mb)
    if not fits:
        return False, fit_reason

    max_instances = (state.host_max_instances or {}).get(step.dst_host, 0)
    current_count = (state.host_running_vms or {}).get(step.dst_host)
    if max_instances and max_instances > 0:
        if current_count is None:
            return False, "max_instances_count_unavailable"
        if current_count + 1 > max_instances:
            return False, "max_instances"

    if enforce_server_groups:
        try:
            # The VM's own record was read at the top of this validation.
            fresh_vm_hosts, fresh_vm_to_groups = _sources_openstack.get_fresh_server_group_context(
                conn,
                step.vm_id,
                known_hosts=[step.src_host, step.dst_host],
                host_to_hypervisor=host_to_hypervisor,
                known_servers={step.vm_id: server},
            )
        except _errors.OpenStackDataError as e:
            return False, f"server_group_refresh: {e}"
        if not _planning_constraints.is_server_group_move_compliant(
            step.vm_id,
            step.src_host,
            step.dst_host,
            fresh_vm_hosts,
            fresh_vm_to_groups,
        ):
            return False, "server_group"
        if enforce_soft_groups and _planning_constraints.soft_group_move_delta(
            step.vm_id, step.src_host, step.dst_host, fresh_vm_hosts, fresh_vm_to_groups
        ) > 0:
            return False, "soft_server_group"
        if soft_aa_heal and _planning_constraints.soft_anti_affinity_move_reduction(
            step.vm_id, step.src_host, step.dst_host, fresh_vm_hosts, fresh_vm_to_groups
        ) <= 0:
            return False, "soft_aa_heal_no_longer_heals"
    elif soft_aa_heal:
        return False, "soft_aa_heal_needs_server_groups"

    if cfg.placement_final_allocation_candidate_check:
        ok, why = _sources_openstack.placement_allocation_candidate_sanity_check(
            conn, fresh_placement, vcpus, ram_mb
        )
        if not ok:
            return False, f"allocation_candidate_sanity: {why}"

    # Runtime safety and runtime-improvement are deliberately separate.
    # MAKE_ROOM step 1 does not have to improve MAD by itself, but it must still
    # be safe for the buffer host based on fresh metrics immediately before the
    # migration is submitted.
    projection = None
    if prevent_new_hotspot or require_runtime_improvement or soft_aa_heal:
        projection = _planning_balance.project_runtime_move(
            state.host_cpu_pct,
            state.host_ram_pct,
            state.host_cores,
            state.host_ram,
            details,
            step.src_host,
            step.dst_host,
        )
        if projection is None:
            return False, "runtime_capacity_missing"

    if prevent_new_hotspot and (_planning_balance.would_create_new_hotspot(
        state.host_cpu_pct,
        state.host_ram_pct,
        projection["new_host_cpu"],
        projection["new_host_ram"],
        step.dst_host,
    ) or _planning_balance.destination_hot_after(
        projection["new_host_cpu"], projection["new_host_ram"], step.dst_host
    )):
        return False, "would_create_new_hotspot"

    if require_runtime_improvement:
        if state.mode in _config.HOTSPOT_MODES and not _planning_balance.is_hotspot_guardrail_satisfied(
            state.mode,
            projection["new_host_cpu"],
            projection["new_host_ram"],
            step.dst_host,
            state.max_cpu,
            state.max_ram,
        ):
            return False, "hotspot_guardrail"

        mad_cpu = _planning_balance.mean_absolute_deviation(list(projection["new_host_cpu"].values()))
        mad_ram = _planning_balance.mean_absolute_deviation(list(projection["new_host_ram"].values()))
        weighted = mad_cpu * state.cpu_weight + mad_ram * state.ram_weight
        improvement = state.baseline_weighted_mad - weighted
        step_improvement_threshold = (
            state.step_improvement_threshold
            if state.step_improvement_threshold is not None
            else state.improvement_threshold
        )
        if improvement < step_improvement_threshold:
            return False, (
                f"refreshed improvement {improvement:.3f} below threshold "
                f"{step_improvement_threshold:.3f}"
            )

    if soft_aa_heal:
        mad_cpu = _planning_balance.mean_absolute_deviation(list(projection["new_host_cpu"].values()))
        mad_ram = _planning_balance.mean_absolute_deviation(list(projection["new_host_ram"].values()))
        # Same per-resource gate as the planner, on the fresh snapshot's own
        # baseline CPU/RAM MAD (evaluate_aggregate_state records both).
        ok, why = _planning_balance.soft_aa_heal_mad_check(
            state.baseline_cpu_mad, state.baseline_ram_mad, mad_cpu, mad_ram)
        if not ok:
            return False, "refreshed " + why

    return True, "ok"


def validate_fresh_projection(state, step, plan, prevent_new_hotspot):
    details = state.all_vm_details.get(step.vm_id)
    if details is None:
        raise _errors.MigrationExecutionError("approved VM is absent from the fresh snapshot")
    for role, host in (("source", step.src_host), ("destination", step.dst_host)):
        if host not in state.host_cpu_pct:
            # e.g. the host became unready after planning and the refresh,
            # using the plan's exclude_unready_hosts, left it out.
            raise _errors.MigrationExecutionError(
                f"{role} host {host} is no longer part of the fresh snapshot "
                f"(excluded: {(state.excluded_hosts or {}).get(host, 'no usable data')})")
    projected = _planning_balance.project_runtime_move(
        state.host_cpu_pct, state.host_ram_pct, state.host_cores, state.host_ram,
        details, step.src_host, step.dst_host)
    if projected is None or projected["new_host_ram"][step.dst_host] > _config.runtime().drain_absolute_max_host_ram_pct:
        raise _errors.MigrationExecutionError("fresh projection violates absolute RAM ceiling")
    if plan.plan_type == _config.DRAIN_PLAN_TYPE:
        from compute_maintenance.domain.drain import drain_host_caps
        cpu_cap, ram_cap = drain_host_caps(allow_hotspot=not prevent_new_hotspot)
        if (projected["new_host_cpu"][step.dst_host] > cpu_cap
                or projected["new_host_ram"][step.dst_host] > ram_cap):
            raise _errors.MigrationExecutionError("fresh projection violates drain destination caps")


def record_rejected_move(cfg, state, step, plan, step_index, why):
    """Revalidation refused an approved step: report it and keep the planner
    away from this VM for a short while, so a cron run does not re-plan the
    very same move every 10 minutes."""
    logging.warning(
        "Step %d of %s plan rejected during fresh validation for VM %s: %s",
        step_index, plan.plan_type, step.vm_id, why,
    )
    _report.append_migration_event(
        _report.build_migration_event(
            state, "step_revalidation_failed", detail=why, step=step, plan=plan, step_index=step_index,
        )
    )
    store = _execution_cooldown.store_for(cfg)
    store.rejected[step.vm_id] = time.time()
    if not store.save():
        raise _errors.MigrationExecutionError(
            "Rejected-move cooldown could not be persisted to the cooldown state file"
        )


def execute_migration_step(
    cfg,
    conn,
    state,
    step,
    plan,
    step_index,
    enforce_server_groups=True,
    send_all=False,
    send_error=False,
    require_runtime_improvement=False,
    prevent_new_hotspot=True,
):
    """Revalidate and run one approved step.

    Returns True (migrated), False (submitted and failed/timed out; failure
    cooldown recorded) or None (refused before submit; nothing changed in the
    cloud, short rejected-move cooldown recorded). Raises when the outcome of
    a submitted migration cannot be proven (journal retained).
    """
    journal = getattr(cfg, "migration_journal", None)
    if journal is not None:
        # Revalidate the exact approved VM/source/destination; do not silently
        # replace it with a different candidate after a long confirmation prompt.
        # The refresh uses the planner's own exclude_unready_hosts (recorded in
        # the plan state): balance/MAKE_ROOM plan without unready hosts, drain
        # with them. A different rule here would recompute the improvement gate
        # on a different baseline (review M3).
        if state.exclude_unready_hosts is None:
            # A context that does not say how it was planned cannot be
            # revalidated (rc11: the dict raised KeyError here).
            raise _plan_context.PlanContextIncomplete("exclude_unready_hosts")
        refreshed = _snapshot.evaluate_aggregate_state(
            cfg, conn, state.aggregate, state.hosts_in_agg,
            enforce_server_groups=enforce_server_groups, plan_balance=False,
            exclude_unready_hosts=state.exclude_unready_hosts)
        if refreshed.error:
            raise _errors.MigrationExecutionError("fresh snapshot failed: " + refreshed.error)
        if state.step_improvement_threshold is not None:
            refreshed.step_improvement_threshold = state.step_improvement_threshold
        # The refresh plans nothing (plan_balance=False), so its own
        # classification reads "no technical candidate" / CRITICAL. The
        # events of this step must describe the executed plan instead.
        refreshed.event_labels = _report.plan_event_labels(state)
        state = refreshed
        try:
            validate_fresh_projection(state, step, plan, prevent_new_hotspot)
        except _errors.MigrationExecutionError as e:
            record_rejected_move(cfg, state, step, plan, step_index, str(e))
            return None
    vm_id = step.vm_id
    vm_name = (state.all_vm_names or {}).get(vm_id, vm_id)
    vm_detail = (state.all_vm_details or {}).get(vm_id, {})
    project_id = vm_detail.get("project_id")
    project_name = vm_detail.get("project_name")

    ok, why = validate_migration_step(
        cfg,
        conn,
        state,
        step,
        enforce_server_groups=enforce_server_groups,
        require_runtime_improvement=require_runtime_improvement,
        prevent_new_hotspot=prevent_new_hotspot,
        enforce_soft_groups=enforce_server_groups and plan.plan_type != _config.DRAIN_PLAN_TYPE,
        soft_aa_heal=plan.plan_type == _config.SOFT_AA_HEAL_PLAN_TYPE,
    )
    if not ok:
        record_rejected_move(cfg, state, step, plan, step_index, why)
        return None

    try:
        preexisting_migration_ids = get_server_migration_ids(conn, vm_id)
        global_max_id = (
            _sources_openstack.max_migration_id(_sources_openstack.server_migration_rows(conn, vm_id))
            if journal is not None else None
        )
    except Exception as e:  # noqa: BLE001 - any failure makes the submit untrackable
        logging.error("Refusing to submit an untrackable migration: %s", e)
        _report.append_migration_event(
            _report.build_migration_event(
                state,
                "migration_tracking_precheck_failed",
                detail=str(e),
                step=step,
                plan=plan,
                step_index=step_index,
            )
        )
        return None

    action_ids = snapshot_action_ids(conn, vm_id)
    migration_started_at = time.time()
    migration_started_datetime = datetime.now()
    submit_error = None
    if journal is not None:
        if time.time() - state.collected_at > 120:
            record_rejected_move(cfg, state, step, plan, step_index,
                                 "snapshot expired before submit; re-plan on the next round")
            return None
        journal.begin(step, preexisting_migration_ids, global_max_id=global_max_id)
    try:
        conn.compute.live_migrate_server(
            server=vm_id,
            host=step.dst_host,
            block_migration=False,
        )
    except Exception as e:  # noqa: BLE001 - classified below: definitive 4xx rejection or uncertain (monitored)
        submit_error = e

    try:
        if submit_error is not None and is_definitive_submission_rejection(submit_error):
            migration_result = _config.MIGRATION_RESULT_FAILED
            if journal is not None:
                journal.finish("rejected", detail=str(submit_error))
        else:
            if submit_error is not None:
                logging.warning(
                    "Nova migration submission returned an uncertain error for VM %s; "
                    "the request may have been accepted, so monitoring before deciding: %s",
                    vm_id,
                    submit_error,
                )
            migration_result = monitor_migration(
                conn,
                vm_id,
                vm_name,
                src_host=step.src_host,
                dst_host=step.dst_host,
                preexisting_migration_ids=preexisting_migration_ids,
                preexisting_action_request_ids=action_ids,
                global_max_id=global_max_id,
            )
    except Exception as e:  # noqa: BLE001 - any monitor failure is recorded, cooled down and reconciled
        migration_ended_datetime = datetime.now()
        migration_duration_seconds = time.time() - migration_started_at
        if journal is None:
            reset_error_server_to_active(conn, vm_id, vm_name, expected_source_host=step.src_host)
        failure_detail = str(e)
        if submit_error is not None:
            failure_detail = f"submit_error={submit_error}; monitor_error={e}"
        if send_error or send_all:
            _notify.send_migration_alert_email(
                cfg,
                state.aggregate,
                vm_id,
                vm_name,
                step.src_host,
                step.dst_host,
                "submit_or_monitor_failed",
                detail=failure_detail,
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                project_name=project_name,
                project_id=project_id,
            )
        _report.append_migration_event(
            _report.build_migration_event(
                state,
                "submit_or_monitor_failed",
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                detail=failure_detail,
                step=step,
                plan=plan,
                step_index=step_index,
            )
        )
        _execution_cooldown.store_for(cfg).failed[vm_id] = time.time()
        if not _execution_cooldown.store_for(cfg).save():
            raise _errors.MigrationExecutionError(
                "Migration outcome could not be persisted to the cooldown state file"
            )
        if journal is not None and journal.state.data.get("pending"):
            journal.reconcile(conn)  # proven outcome closes it; otherwise raises
        return False

    migration_ended_datetime = datetime.now()
    migration_duration_seconds = time.time() - migration_started_at
    now = time.time()

    if migration_result == _config.MIGRATION_RESULT_SUCCESS:
        if journal is not None:
            journal.finish("success")
        _execution_cooldown.store_for(cfg).moved[vm_id] = now
        _execution_cooldown.store_for(cfg).failed.pop(vm_id, None)
        _execution_cooldown.store_for(cfg).rejected.pop(vm_id, None)
        if not _execution_cooldown.store_for(cfg).save():
            _report.append_migration_event(
                _report.build_migration_event(
                    state,
                    "cooldown_persist_failed_after_success",
                    detail="migration succeeded but cooldown persistence failed",
                    step=step,
                    plan=plan,
                    step_index=step_index,
                )
            )
            raise _errors.MigrationExecutionError(
                "Migration succeeded but cooldown state could not be persisted"
            )
        _report.append_migration_event(
            _report.build_migration_event(
                state,
                "migration_success",
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                detail=(
                    f"monitor_migration returned {migration_result}"
                    + (f" after uncertain submit error: {submit_error}" if submit_error else "")
                ),
                step=step,
                plan=plan,
                step_index=step_index,
            )
        )
        if send_all:
            _notify.send_migration_alert_email(
                cfg,
                state.aggregate,
                vm_id,
                vm_name,
                step.src_host,
                step.dst_host,
                "migration_success",
                detail=(
                    f"plan={plan.plan_type}, step={step_index}/{len(plan.steps)}, "
                    f"monitor_migration returned {migration_result}"
                ),
                duration_seconds=migration_duration_seconds,
                started_at=migration_started_datetime,
                ended_at=migration_ended_datetime,
                project_name=project_name,
                project_id=project_id,
            )
        return True

    if journal is None:
        reset_error_server_to_active(conn, vm_id, vm_name, expected_source_host=step.src_host)
    if migration_result == _config.MIGRATION_RESULT_STUCK_ACTIVE:
        alert_result = "stuck_active"
    elif migration_result == _config.MIGRATION_RESULT_TIMEOUT:
        alert_result = "migration_timeout"
    else:
        alert_result = "migration_failed"

    failure_detail = format_migration_monitor_detail(migration_result, submit_error=submit_error,
        monitor_called=not (submit_error is not None and is_definitive_submission_rejection(submit_error)))
    # Record the failure BEFORE touching the journal: if the outcome cannot be
    # proven, reconcile raises, and the operator must still get the event,
    # the alert and the cooldown entry (previously all three were lost).
    if send_error or send_all:
        _notify.send_migration_alert_email(
            cfg,
            state.aggregate,
            vm_id,
            vm_name,
            step.src_host,
            step.dst_host,
            alert_result,
            detail=failure_detail,
            duration_seconds=migration_duration_seconds,
            started_at=migration_started_datetime,
            ended_at=migration_ended_datetime,
            project_name=project_name,
            project_id=project_id,
        )
    _report.append_migration_event(
        _report.build_migration_event(
            state,
            alert_result,
            duration_seconds=migration_duration_seconds,
            started_at=migration_started_datetime,
            ended_at=migration_ended_datetime,
            detail=failure_detail,
            step=step,
            plan=plan,
            step_index=step_index,
        )
    )
    _execution_cooldown.store_for(cfg).failed[vm_id] = now
    if not _execution_cooldown.store_for(cfg).save():
        raise _errors.MigrationExecutionError(
            "Failed migration outcome could not be persisted to the cooldown state file"
        )
    if journal is not None and journal.state.data.get("pending"):
        # Proven terminal failure / never-started closes the intent; a still
        # running or unprovable migration raises and keeps it (no new submit).
        outcome = journal.reconcile(conn)
        logging.warning("Migration of VM %s ended as %s (%s).", vm_id, outcome, alert_result)
    return False


def wait_for_make_room_step(
    cfg,
    conn,
    agg_name,
    hosts_in_agg,
    step,
    enforce_server_groups=True,
    exclude_vm_patterns=None,
    *,
    exclude_unready_hosts,
):
    """Refresh and validate MAKE_ROOM step 2 until Placement converges.

    Only stale-capacity failures are retried.  Any topology, VM status,
    server-group, readiness, hotspot, or improvement change aborts at once.
    ``exclude_unready_hosts`` is the plan's own value (its state's
    ``exclude_unready_hosts``), so the refreshed baseline follows the planner.
    The refresh plans nothing (``plan_balance=False``): validation reads only
    the snapshot, and the step's events keep the plan's own labels.
    """
    deadline = time.monotonic() + cfg.placement_sync_timeout
    attempt = 0

    while True:
        attempt += 1
        # Step 2 runs in apply mode: the settle barrier inside the refresh
        # discards any cache collected before it (a journal-less library
        # caller is the only one that still reads it).
        refreshed_cache = None
        if _snapshot.prefetched_cluster_cache_is_used(cfg):
            refreshed_cache = _snapshot.build_cluster_cache(
                conn,
                cfg,
                hosts_in_agg,
                enforce_server_groups=enforce_server_groups,
            )
        refreshed_state = _snapshot.evaluate_aggregate_state(
            cfg=cfg,
            conn=conn,
            agg_name=agg_name,
            hosts_in_agg=hosts_in_agg,
            enforce_server_groups=enforce_server_groups,
            cluster_cache=refreshed_cache,
            exclude_vm_patterns=exclude_vm_patterns,
            exclude_unready_hosts=exclude_unready_hosts,
            plan_balance=False,
        )
        if refreshed_state.error:
            return refreshed_state, False, f"refresh evaluation failed: {refreshed_state.error}"

        ok, why = validate_migration_step(
            cfg,
            conn,
            refreshed_state,
            step,
            enforce_server_groups=enforce_server_groups,
            require_runtime_improvement=True,
            prevent_new_hotspot=True,
        )
        if ok:
            if attempt > 1:
                logging.info(
                    "Placement converged for MAKE_ROOM step 2 after %d validation attempts.",
                    attempt,
                )
            return refreshed_state, True, "ok"

        retryable = why in ("placement_vcpu", "placement_memory")
        remaining = deadline - time.monotonic()
        if not retryable or remaining <= 0:
            return refreshed_state, False, why

        delay = min(cfg.placement_sync_interval, remaining)
        logging.info(
            "MAKE_ROOM step 2 still blocked by stale %s (attempt %d); "
            "refreshing again in %.1f seconds.",
            why,
            attempt,
            delay,
        )
        time.sleep(delay)
