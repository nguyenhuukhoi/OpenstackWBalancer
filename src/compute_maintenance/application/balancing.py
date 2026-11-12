"""execution.workflow — extracted domain responsibilities."""
import logging
import time
from datetime import datetime

import compute_maintenance.adapters.openstack as _sources_openstack
import compute_maintenance.application.migrations as _execution_migrate
import compute_maintenance.application.snapshots as _snapshot
import compute_maintenance.config as _config
import compute_maintenance.domain.drain as _planning_drain
import compute_maintenance.domain.errors as _errors
import compute_maintenance.domain.plan_context as _plan_context
import compute_maintenance.domain.severity as _severity
import compute_maintenance.presentation.reports as _report


def auto_balance_aggregate(
    cfg,
    conn,
    agg_name,
    hosts_in_agg,
    dry_run=False,
    enforce_server_groups=True,
    assume_yes=False,
    send_all=False,
    send_error=False,
    cluster_cache=None,
    exclude_vm_patterns=None,
):
    print(f"\n[Aggregate: {agg_name}] Placement-aware balancing evaluation.\n")

    if enforce_server_groups:
        logging.debug("Server group checks ENABLED (default behavior).")
    else:
        logging.warning(
            "Server group checks DISABLED (--no-server-groups). Affinity/anti-affinity will be ignored!"
        )

    state = _snapshot.evaluate_aggregate_state(
        cfg=cfg,
        conn=conn,
        agg_name=agg_name,
        hosts_in_agg=hosts_in_agg,
        enforce_server_groups=enforce_server_groups,
        cluster_cache=cluster_cache,
        exclude_vm_patterns=exclude_vm_patterns,
        exclude_unready_hosts=True,
    )

    if state.error:
        raise _errors.OpenStackDataError(
            f"Aggregate {agg_name} evaluation failed: {state.error}"
        )

    _report.print_baseline_table(state)
    _report.print_state_summary(state)
    if state.soft_aa_heal:
        _report.record_soft_aa_heal_decision(state)

    if not state.has_valid_move:
        return False

    plan = state.best_plan
    _report.print_move_details(state, migration_round=1)

    if dry_run:
        _report.append_migration_event(
            _report.build_migration_event(
                state,
                "dry_run",
                detail=(
                    f"valid {plan.plan_type} plan proposed; dry-run mode did not execute "
                    f"{len(plan.steps)} migration step(s)"
                ),
                plan=plan,
            )
        )
        return False

    if not assume_yes:
        summary = "; ".join(
            f"{state.all_vm_names.get(step.vm_id, step.vm_id)} "
            f"{step.src_host}->{step.dst_host}"
            for step in plan.steps
        )
        try:
            answer = input(
                f"Execute {plan.plan_type.upper()} plan ({summary})? [y/N]: "
            ).strip().lower()
        except EOFError:  # no terminal: never assume yes
            answer = ""
        if answer not in ("y", "yes"):
            print("Migration plan cancelled.")
            _report.append_migration_event(
                _report.build_migration_event(
                    state,
                    "cancelled",
                    detail="operator declined migration plan prompt",
                    plan=plan,
                )
            )
            return False

    # DIRECT: one step, but still revalidate live Nova/Placement state immediately
    # before submitting the migration. A soft-anti-affinity heal (rc11) is one
    # step too and takes the same path; its revalidation checks "still heals"
    # and the heal MAD gate instead of the improvement threshold.
    if plan.plan_type in ("direct", _config.SOFT_AA_HEAL_PLAN_TYPE):
        label = "DIRECT" if plan.plan_type == "direct" else plan.plan_type.upper()
        ok = _execution_migrate.execute_migration_step(
            cfg,
            conn,
            state,
            plan.steps[0],
            plan,
            1,
            enforce_server_groups=enforce_server_groups,
            send_all=send_all,
            send_error=send_error,
            require_runtime_improvement=plan.plan_type == "direct",
            prevent_new_hotspot=True,
        )
        if ok is None:
            logging.warning("%s move of VM %s refused at revalidation; nothing migrated.",
                            label, plan.steps[0].vm_id)
            return False
        if not ok:
            raise _errors.MigrationExecutionError(
                f"{label} migration failed for VM {plan.steps[0].vm_id}"
            )
        return True

    # MAKE_ROOM step 1 is allowed to be neutral/slightly worse by itself because
    # its purpose is to release Placement allocation for step 2.
    first = plan.steps[0]
    ok = _execution_migrate.execute_migration_step(
        cfg,
        conn,
        state,
        first,
        plan,
        1,
        enforce_server_groups=enforce_server_groups,
        send_all=send_all,
        send_error=send_error,
        require_runtime_improvement=False,
        prevent_new_hotspot=True,
    )
    if ok is None:
        logging.warning("MAKE_ROOM step 1 (VM %s) refused at revalidation; nothing migrated.",
                        first.vm_id)
        return False
    if not ok:
        raise _errors.MigrationExecutionError(
            f"MAKE_ROOM step 1 failed for VM {first.vm_id}"
        )
    # Critical safety rule: never execute step 2 from stale simulation. Refresh
    # all inputs and tolerate only bounded Placement eventual consistency.
    second = plan.steps[1]
    refreshed_state, ok, why = _execution_migrate.wait_for_make_room_step(
        cfg,
        conn,
        agg_name,
        hosts_in_agg,
        second,
        enforce_server_groups=enforce_server_groups,
        exclude_vm_patterns=exclude_vm_patterns,
        exclude_unready_hosts=state.exclude_unready_hosts,
    )
    # Step 2's events describe this MAKE_ROOM plan, not whatever the
    # post-step-1 snapshot would plan next.
    refreshed_state.event_labels = _report.plan_event_labels(state)
    if not ok:
        logging.warning(
            "MAKE_ROOM step 1 succeeded but step 2 is no longer safe/valid: %s",
            why,
        )
        _report.append_migration_event(
            _report.build_migration_event(
                refreshed_state,
                "make_room_aborted_after_refresh",
                detail=why,
                step=second,
                plan=plan,
                step_index=2,
            )
        )
        raise _errors.MigrationExecutionError(
            f"MAKE_ROOM step 1 succeeded but step 2 was aborted: {why}"
        )

    ok = _execution_migrate.execute_migration_step(
        cfg,
        conn,
        refreshed_state,
        second,
        plan,
        2,
        enforce_server_groups=enforce_server_groups,
        send_all=send_all,
        send_error=send_error,
        require_runtime_improvement=True,
        prevent_new_hotspot=True,
    )
    if ok is None:
        # Step 1 already moved a VM (by design neutral on its own); the second
        # move was refused before submit, so nothing else changed.
        logging.warning("MAKE_ROOM step 2 (VM %s) refused at revalidation after step 1.",
                        second.vm_id)
        return True
    if not ok:
        raise _errors.MigrationExecutionError(
            f"MAKE_ROOM step 2 failed for VM {second.vm_id}"
        )
    return True


def balance_by_aggregate(
    cfg,
    dry_run=False,
    aggregate_names=None,
    enforce_server_groups=True,
    assume_yes=False,
    send_all=False,
    send_error=False,
    exclude_vm_patterns=None,
):
    conn = _sources_openstack.get_openstack_connection(getattr(cfg, "os_cloud", _config.DEFAULT_OS_CLOUD))
    aggregate_map = _sources_openstack.get_aggregate_map(
        conn,
        aggregate_names=aggregate_names,
        show_available_on_missing=dry_run,
    )

    # Fetch cluster-wide data once and reuse it for every aggregate — but only
    # when there is no journal: apply mode waits on the settle barrier inside
    # evaluate_aggregate_state and discards any cache collected before it, so
    # prefetching there would be collected and thrown away on every run.
    all_hosts = _sources_openstack.build_selected_host_list(aggregate_map)
    journal = getattr(cfg, "migration_journal", None)
    cluster_cache = None
    if journal is None:
        cluster_cache = _snapshot.build_cluster_cache(
            conn, cfg, all_hosts, enforce_server_groups=enforce_server_groups
        )

    failed_aggregates = []
    for agg_name, hosts in aggregate_map.items():
        if len(hosts) < 2:
            logging.info("Aggregate %s: less than 2 hosts, skipping.", agg_name)
            continue

        try:
            migrated = auto_balance_aggregate(
                cfg=cfg,
                conn=conn,
                agg_name=agg_name,
                hosts_in_agg=hosts,
                dry_run=dry_run,
                enforce_server_groups=enforce_server_groups,
                assume_yes=assume_yes,
                send_all=send_all,
                send_error=send_error,
                cluster_cache=cluster_cache,
                exclude_vm_patterns=exclude_vm_patterns,
            )
        except (_errors.OpenStackDataError, _errors.PrometheusError, _errors.PlacementError) as e:
            # A data problem of one aggregate (a host without metrics, a
            # missing Placement provider, ...) must not stop the others.
            # Migration/journal errors still stop the whole run.
            logging.error("Aggregate %s skipped: %s", agg_name, e)
            failed_aggregates.append(agg_name)
            continue

        # A successful migration changes host capacity and VM placement. Because
        # aggregates may share hosts, refresh the cluster cache so subsequent
        # aggregates evaluate against the post-migration state. With a journal
        # the refresh would be discarded at the settle barrier like the prefetch.
        if migrated and journal is None:
            cluster_cache = _snapshot.build_cluster_cache(
                conn, cfg, all_hosts, enforce_server_groups=enforce_server_groups
            )

    if failed_aggregates:
        raise _errors.OpenStackDataError(
            "evaluation failed for aggregate(s): " + ", ".join(failed_aggregates)
        )


def evaluate_drain(
    cfg,
    conn,
    agg_name,
    hosts_in_agg,
    drain_host,
    enforce_server_groups=True,
    cluster_cache=None,
    exclude_vm_patterns=None,
    allow_hotspot=False,
):
    """Refresh aggregate state and plan the drain against it."""
    state = _snapshot.evaluate_aggregate_state(
        cfg=cfg,
        conn=conn,
        agg_name=agg_name,
        hosts_in_agg=hosts_in_agg,
        enforce_server_groups=enforce_server_groups,
        cluster_cache=cluster_cache,
        exclude_vm_patterns=exclude_vm_patterns,
        plan_balance=False,
    )
    if state.error:
        raise _errors.OpenStackDataError(
            f"Aggregate {agg_name} evaluation failed: {state.error}"
        )
    if drain_host not in state.hosts_in_agg:
        raise _errors.ConfigError(
            "Host {} is not an aggregate {} member with usable metrics "
            "(members: {})".format(
                drain_host, agg_name, ", ".join(state.hosts_in_agg)
            )
        )

    ctx = _planning_drain.build_drain_context(state, enforce_server_groups=enforce_server_groups)
    plan = _planning_drain.plan_host_drain(
        ctx,
        agg_name,
        drain_host,
        state.cpu_weight,
        state.ram_weight,
        allow_hotspot=allow_hotspot,
    )
    # The drain snapshot plans no balancing move (plan_balance=False), so its
    # balance classification ("no technical candidate", CRITICAL) says nothing
    # about the drain. Drain events (and the migrations of its steps) carry
    # these labels instead; the drain outcome itself is in "result".
    state.event_labels = _plan_context.PlanLabels(
        mode=_config.DRAIN_PLAN_TYPE,
        detail_status=_severity.drain_detail_status(plan.feasible),
        severity=None,
        best_improvement=None,
        improvement_threshold=None,
    )
    return state, plan


def drain_transient_unmovable(plan):
    """Unmovable entries that are expected to clear on their own."""
    return [
        entry for entry in plan.unmovable
        if entry["reason"] in _config.DRAIN_TRANSIENT_UNMOVABLE_REASONS
    ]


def wait_for_drain_metrics_settle(replan, state, plan):
    """Re-plan while VMs on the drained host only *look* unmovable.

    Without this, the migration that just completed would hide the next VM
    behind its own dual-series Prometheus window, the planner would report zero
    remaining steps, and the drain would declare success with VMs still on the
    host. Transient gaps are therefore waited out before that conclusion.
    """
    deadline = time.monotonic() + _config.DRAIN_METRIC_SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        pending = drain_transient_unmovable(plan)
        if not pending:
            break
        logging.info(
            "Drain of %s: %d VM(s) still have incomplete metrics/metadata; waiting "
            "%.0fs before concluding the host is drained (%s).",
            plan.host,
            len(pending),
            _config.DRAIN_METRIC_SETTLE_INTERVAL,
            ", ".join(entry["vm_id"] for entry in pending),
        )
        time.sleep(_config.DRAIN_METRIC_SETTLE_INTERVAL)
        state, plan = replan()
        if plan.steps or not plan.feasible:
            break
    return state, plan


def drain_host_workload(
    cfg,
    conn,
    agg_name,
    hosts_in_agg,
    drain_host,
    dry_run=False,
    enforce_server_groups=True,
    assume_yes=False,
    send_all=False,
    send_error=False,
    cluster_cache=None,
    exclude_vm_patterns=None,
    allow_hotspot=False,
    disable_host=True,
):
    """Evacuate every VM off ``drain_host`` while keeping the rest balanced.

    Fail-closed contract: nothing is migrated and the nova-compute service is
    left untouched unless the whole evacuation is planned as feasible first.
    Each step is then re-planned against freshly fetched Nova/Placement/
    Prometheus state, so no migration is ever submitted from a stale
    simulation.
    """
    print(
        f"\n[Aggregate: {agg_name}] Drain evaluation for host {drain_host}.\n"
    )
    if not enforce_server_groups:
        logging.warning(
            "Server group checks DISABLED (--no-server-groups). Affinity/anti-affinity "
            "will be ignored while draining!"
        )

    state, plan = evaluate_drain(
        cfg,
        conn,
        agg_name,
        hosts_in_agg,
        drain_host,
        enforce_server_groups=enforce_server_groups,
        cluster_cache=cluster_cache,
        exclude_vm_patterns=exclude_vm_patterns,
        allow_hotspot=allow_hotspot,
    )
    _report.print_baseline_table(state)
    _report.print_drain_report(state, plan, allow_hotspot)
    migration_plan = _planning_drain.drain_plan_to_migration_plan(plan)

    if not plan.feasible:
        detail = "; ".join(plan.infeasible_reasons) or "drain planner found no safe packing"
        logging.error("Drain of %s is not possible: %s", drain_host, detail)
        _report.append_migration_event(
            _report.build_drain_event(
                state, plan, "drain_infeasible", detail=detail, migration_plan=migration_plan
            )
        )
        return _config.DRAIN_EXIT_INFEASIBLE

    if not plan.steps:
        detail = (
            f"{drain_host} has no live-migratable VM left"
            if not plan.unmovable
            else f"{drain_host} has only non-live-migratable VM(s) left"
        )
        print(f"Nothing to migrate: {detail}.")
        _report.append_migration_event(
            _report.build_drain_event(state, plan, "drain_noop", detail=detail)
        )
        # Non-zero whenever the host is not actually clean, so an automated
        # maintenance runbook cannot take the host down on a green exit code.
        return _config.DRAIN_EXIT_INFEASIBLE if plan.unmovable else _config.DRAIN_EXIT_OK

    if dry_run:
        _report.append_migration_event(
            _report.build_drain_event(
                state,
                plan,
                "drain_dry_run",
                detail=f"feasible drain plan with {len(plan.steps)} step(s); dry-run did not "
                "migrate or disable the host",
                migration_plan=migration_plan,
            )
        )
        # Exit 0 from a drain dry-run means "this host can be fully emptied",
        # so VMs that would be left behind must still fail the pre-flight.
        return _config.DRAIN_EXIT_INFEASIBLE if plan.unmovable else _config.DRAIN_EXIT_OK

    if not assume_yes:
        try:
            answer = input(
                f"Drain {drain_host} by live-migrating {len(plan.steps)} VM(s)? [y/N]: "
            ).strip().lower()
        except EOFError:  # no terminal: never assume yes
            answer = ""
        if answer not in ("y", "yes"):
            print("Drain cancelled.")
            _report.append_migration_event(
                _report.build_drain_event(
                    state,
                    plan,
                    "drain_cancelled",
                    detail="operator declined the drain prompt",
                    migration_plan=migration_plan,
                )
            )
            # Not drained: a wrapper such as "wbalancer --drain-host X && reboot"
            # must not see success.
            return _config.DRAIN_EXIT_FAILED

    # Stop the scheduler from landing new instances on a host we are emptying.
    # Only re-enabled on abort, and only if this run is the one that disabled it.
    disabled_by_this_run = False
    if disable_host:
        found, already_disabled, disabled_reason = _sources_openstack.get_nova_service_disabled_state(
            conn, drain_host
        )
        if not found:
            logging.error(
                "Cannot disable nova-compute on %s: service not found; aborting before "
                "any migration.",
                drain_host,
            )
            _report.append_migration_event(
                _report.build_drain_event(
                    state,
                    plan,
                    "drain_failed",
                    detail="nova-compute service not found for " + drain_host,
                    migration_plan=migration_plan,
                )
            )
            return _config.DRAIN_EXIT_FAILED
        if already_disabled:
            logging.info(
                "nova-compute on %s is already disabled (reason=%s); leaving it as is.",
                drain_host,
                disabled_reason,
            )
        else:
            ok, detail = _sources_openstack.set_nova_compute_service_enabled(
                conn,
                drain_host,
                False,
                disabled_reason=f"wlbalancer drain {_report.format_migration_timestamp(datetime.now())}",
            )
            if not ok:
                logging.error("%s; aborting before any migration.", detail)
                _report.append_migration_event(
                    _report.build_drain_event(
                        state, plan, "drain_failed", detail=detail,
                        migration_plan=migration_plan,
                    )
                )
                return _config.DRAIN_EXIT_FAILED
            disabled_by_this_run = True
            logging.info("%s", detail)

    def abort(reason, failed_state, failed_plan, failed_migration_plan):
        if disabled_by_this_run:
            ok, detail = _sources_openstack.set_nova_compute_service_enabled(conn, drain_host, True)
            if ok:
                logging.info("Rolled back host disable: %s", detail)
            else:
                logging.error(
                    "Drain aborted AND the host could not be re-enabled: %s. "
                    "Re-enable it manually: openstack compute service set "
                    "--enable %s nova-compute",
                    detail,
                    drain_host,
                )
        _report.append_migration_event(
            _report.build_drain_event(
                failed_state,
                failed_plan,
                "drain_aborted",
                detail=reason,
                migration_plan=failed_migration_plan,
            )
        )
        logging.error("Drain of %s aborted: %s", drain_host, reason)

    total_steps = len(plan.steps)
    max_rounds = total_steps * _config.DRAIN_MAX_ROUND_FACTOR + _config.DRAIN_MAX_ROUND_SLACK
    completed = 0
    # A step refused at revalidation submitted nothing; re-plan from fresh
    # state instead of aborting the drain, a bounded number of times in a row.
    refused_in_a_row = 0

    def replan():
        # Never migrate from a stale simulation: re-fetch Nova, Placement and
        # Prometheus and re-plan the remaining evacuation.
        return evaluate_drain(
            cfg,
            conn,
            agg_name,
            hosts_in_agg,
            drain_host,
            enforce_server_groups=enforce_server_groups,
            exclude_vm_patterns=exclude_vm_patterns,
            allow_hotspot=allow_hotspot,
        )

    for round_index in range(1, max_rounds + 1):
        round_state, round_plan = replan()

        # A VM hidden behind its own post-migration Prometheus window must not
        # be mistaken for one that cannot be migrated at all.
        if not round_plan.steps and drain_transient_unmovable(round_plan):
            round_state, round_plan = wait_for_drain_metrics_settle(
                replan, round_state, round_plan
            )

        round_migration_plan = _planning_drain.drain_plan_to_migration_plan(round_plan)

        if not round_plan.feasible:
            abort(
                "re-plan after {} completed migration(s) is no longer feasible: {}".format(
                    completed, "; ".join(round_plan.infeasible_reasons)
                ),
                round_state,
                round_plan,
                round_migration_plan,
            )
            return _config.DRAIN_EXIT_FAILED

        if not round_plan.steps:
            print(
                f"\nDrain of {drain_host} finished: {completed} migration(s) completed."
            )
            if round_plan.unmovable:
                print(
                    f"{len(round_plan.unmovable)} VM(s) still on the host and NOT live-migratable:"
                )
                for entry in round_plan.unmovable:
                    print(
                        "   {} reason={}".format(
                            _report.format_drain_vm_label(round_state, entry["vm_id"]),
                            entry["reason"],
                        )
                    )
            _report.print_drain_report(round_state, round_plan, allow_hotspot)
            if disable_host:
                print(
                    f"nova-compute on {drain_host} is disabled. Re-enable it after maintenance:\n"
                    f"   openstack compute service set --enable {drain_host} nova-compute"
                )
            _report.append_migration_event(
                _report.build_drain_event(
                    round_state,
                    round_plan,
                    "drain_completed",
                    detail=f"{completed} migration(s) completed; {len(round_plan.unmovable)} non-live-migratable VM(s) "
                    "remain",
                )
            )
            # Same rule as the no-op path: a host that still holds VMs must not
            # report success to an automated maintenance runbook.
            if round_plan.unmovable:
                return _config.DRAIN_EXIT_INFEASIBLE
            return _config.DRAIN_EXIT_OK

        step = round_plan.steps[0]
        print(
            f"\n[Drain round {round_index}] {_report.format_drain_vm_label(round_state, step.vm_id)} -> {step.dst_host} ({len(round_plan.steps)} step(s) left)"
        )
        ok = _execution_migrate.execute_migration_step(
            cfg,
            conn,
            round_state,
            step,
            round_migration_plan,
            completed + 1,
            enforce_server_groups=enforce_server_groups,
            send_all=send_all,
            send_error=send_error,
            # A drain step exists to empty the host, not to improve MAD, so the
            # improvement gate is off. The hotspot guard stays on unless the
            # operator explicitly accepted hotspots for this maintenance.
            require_runtime_improvement=False,
            prevent_new_hotspot=not allow_hotspot,
        )
        if ok is None and refused_in_a_row < _config.DRAIN_MAX_CONSECUTIVE_REFUSALS:
            refused_in_a_row += 1
            logging.warning(
                "Drain step for VM %s refused at revalidation (%d/%d in a row); re-planning.",
                step.vm_id, refused_in_a_row, _config.DRAIN_MAX_CONSECUTIVE_REFUSALS,
            )
            time.sleep(_config.DRAIN_REFUSAL_REPLAN_DELAY)
            continue
        if not ok:
            abort(
                f"migration of VM {step.vm_id} from {step.src_host} to {step.dst_host} failed or was rejected on "
                "revalidation",
                round_state,
                round_plan,
                round_migration_plan,
            )
            return _config.DRAIN_EXIT_FAILED
        refused_in_a_row = 0
        completed += 1

    abort(
        f"drain did not converge within {max_rounds} rounds ({completed} migration(s) completed)",
        state,
        plan,
        migration_plan,
    )
    return _config.DRAIN_EXIT_FAILED


def drain_by_aggregate(
    cfg,
    drain_host,
    dry_run=False,
    aggregate_names=None,
    enforce_server_groups=True,
    assume_yes=False,
    send_all=False,
    send_error=False,
    exclude_vm_patterns=None,
    allow_hotspot=False,
    disable_host=True,
):
    """Resolve the drained host to exactly one aggregate and drain it there."""
    conn = _sources_openstack.get_openstack_connection(getattr(cfg, "os_cloud", _config.DEFAULT_OS_CLOUD))
    aggregate_map = _sources_openstack.get_aggregate_map(
        conn,
        aggregate_names=aggregate_names,
        show_available_on_missing=dry_run,
    )

    owning = {
        name: hosts for name, hosts in aggregate_map.items() if drain_host in hosts
    }
    if not owning:
        raise _errors.ConfigError(
            "Host {} is not a member of the selected aggregate(s): {}".format(
                drain_host, ", ".join(aggregate_map) or "none"
            )
        )
    if len(owning) > 1:
        raise _errors.ConfigError(
            "Host {} belongs to several selected aggregates ({}); re-run with a "
            "single -a/--aggregate so the drain target is unambiguous".format(
                drain_host, ", ".join(sorted(owning))
            )
        )

    agg_name, hosts = next(iter(owning.items()))
    if len(hosts) < 2:
        raise _errors.ConfigError(
            f"Aggregate {agg_name} has only {len(hosts)} host(s); there is nowhere to drain {drain_host} to"
        )

    # Drain runs in the compensate settle mode, which collects its own cache
    # after waiting for running migrations, so a prefetch here would be
    # collected and thrown away (dry-run and apply alike). Only a journal-less
    # barrier caller (library use) still reads it.
    cluster_cache = None
    if _snapshot.prefetched_cluster_cache_is_used(cfg):
        cluster_cache = _snapshot.build_cluster_cache(
            conn, cfg, hosts, enforce_server_groups=enforce_server_groups
        )
    return drain_host_workload(
        cfg=cfg,
        conn=conn,
        agg_name=agg_name,
        hosts_in_agg=hosts,
        drain_host=drain_host,
        dry_run=dry_run,
        enforce_server_groups=enforce_server_groups,
        assume_yes=assume_yes,
        send_all=send_all,
        send_error=send_error,
        cluster_cache=cluster_cache,
        exclude_vm_patterns=exclude_vm_patterns,
        allow_hotspot=allow_hotspot,
        disable_host=disable_host,
    )
