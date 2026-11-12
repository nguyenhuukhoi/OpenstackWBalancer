"""cli.balance — extracted domain responsibilities."""
import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import click

from compute_maintenance.application.recovery import (
    MigrationJournal,
    validate_settle_window,
)

try:
    import fcntl
except ImportError:
    fcntl = None
import compute_maintenance.adapters.cooldown as _execution_cooldown
import compute_maintenance.adapters.locking as _locking
import compute_maintenance.adapters.selection as _selection
import compute_maintenance.application.balancing as _execution_workflow
import compute_maintenance.application.monitoring as _execution_monitor
import compute_maintenance.config as _config
import compute_maintenance.domain.errors as _errors
import compute_maintenance.paths as _paths
import compute_maintenance.presentation.reports as _report


@click.command()
@click.option('--config', default=_config.DEFAULT_CONFIG_FILE, show_default=True,
              envvar=_paths.CONFIG_FILE_ENV,
              help="Path to config file (Prometheus URL/jobs/queries, SMTP, ...); "
                   "env WLB_CONFIG_PATH replaces the default")
@click.option('--verbose', is_flag=True, help="Enable verbose logging")
@click.option('--dry-run', is_flag=True, help="Only show what would be migrated, do not migrate")
@click.option('--monitor-only', is_flag=True, help="Only evaluate cluster balance/hotspot state and return Icinga-compatible exit code")
@click.option('--aggregate', '-a', multiple=True, help="Only operate on these aggregate(s)")
@click.option(
    '--no-server-groups',
    is_flag=True,
    help="Disable server group (affinity/anti-affinity) checks. WARNING: may violate policies."
)
@click.option('-y', '--yes', 'assume_yes', is_flag=True, help="Skip y/n prompt and auto-approve migrations")
@click.option(
    '--use-vm-ram-rss',
    is_flag=True,
    help="Use libvirt_domain_stat_memory_rss_bytes to estimate host-side VM RAM impact",
)
@click.option('--send-all', is_flag=True, help="Send migration email alerts for success and errors")
@click.option(
    '--send-error',
    is_flag=True,
    help="Send migration email alerts for failed, stuck, timeout, or submit/monitor errors",
)
@click.option(
    '--cooldown-file',
    default=_config.COOLDOWN_STATE_FILE,
    show_default=True,
    help="Path to JSON file used to persist VM migration cooldown state",
)
@click.option(
    '--events-file',
    default=_config.MIGRATION_EVENTS_FILE,
    show_default=True,
    help="Path to JSONL file used to append valid migration move events",
)
@click.option(
    '--lock-file',
    default=_config.LOCK_FILE,
    show_default=True,
    help="Path to the single-instance lock file used in apply mode",
)
@click.option(
    '--exclude-file',
    default=None,
    help=(
        "Path to a text file listing instances to exclude from balancing "
        "(one VM UUID, exact name, or name glob per line; '#' starts a comment). "
        "Overrides EXCLUDE_VMS_FILE from the config file."
    ),
)
@click.option(
    '--drain-host',
    default=None,
    help=(
        "MAINTENANCE MODE: live-migrate every VM off this compute host while keeping "
        "the remaining hosts balanced. Fail-closed: if the whole evacuation cannot be "
        "planned safely, nothing is migrated and the shortfall is reported. "
        "Overrides per-VM cooldown and the exclude list for VMs on this host."
    ),
)
@click.option(
    '--drain-allow-hotspot',
    is_flag=True,
    help=(
        "Allow drain destinations to cross the CPU/RAM hotspot thresholds. Only the "
        "absolute RAM safety ceiling is still enforced. Use when finishing the "
        "maintenance matters more than post-drain balance."
    ),
)
@click.option(
    '--drain-no-disable',
    is_flag=True,
    help=(
        "Do not disable the nova-compute service of --drain-host before draining. "
        "By default the service is disabled so the scheduler stops placing new "
        "instances on it, and re-enabled if the drain aborts."
    ),
)
@click.option(
    '--os-cloud',
    default=None,
    help=(
        "Cloud name in clouds.yaml (with an openrc environment: its OS_CLOUD_NAME, "
        "'envvars' by default). Overrides OS_CLOUD from the environment or config; "
        "default 'openstack'."
    ),
)
@click.option("--run-id", default=None, help="parent invocation identity")
@click.option("--result-file", default=None, help="atomic invocation-scoped drain result")
def main(
    config,
    verbose,
    dry_run,
    monitor_only,
    aggregate,
    no_server_groups,
    assume_yes,
    use_vm_ram_rss,
    send_all,
    send_error,
    cooldown_file,
    events_file,
    lock_file,
    exclude_file,
    drain_host,
    drain_allow_hotspot,
    drain_no_disable,
    os_cloud,
    run_id,
    result_file,
):
    from compute_maintenance.adapters.protocol import DrainResultWriter, current_result
    if bool(run_id) != bool(result_file) or (result_file and (not drain_host or len(aggregate) != 1)):
        raise click.UsageError("--run-id/--result-file require each other, --drain-host and exactly one aggregate")
    if os_cloud is not None and not os_cloud.strip():
        raise click.UsageError("--os-cloud must not be empty")
    # One resolution for the connection AND the result identity (--os-cloud >
    # env OS_CLOUD > OS_CLOUD in --config > default): the identity a drain
    # child reports is always the cloud it really migrated on.
    try:
        cloud = _config.resolve_os_cloud(os_cloud, config)
    except (_errors.ConfigError, OSError) as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(_config.ICINGA_UNKNOWN if monitor_only else 1)
    result_writer = DrainResultWriter(result_file, {
        "run_id": run_id, "cloud": cloud, "aggregate": aggregate[0], "host": drain_host,
        "mode": "dry-run" if dry_run else "apply"}) if result_file else None
    result_token = current_result.set(result_writer)
    if monitor_only:
        # Disable INFO logs in monitor mode (keep WARNING/ERROR if any)
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s %(levelname)s: %(message)s'
        )
    else:
        _report.setup_logging(verbose)
    enforce_server_groups = not no_server_groups

    if monitor_only and dry_run:
        logging.warning("--dry-run is ignored when --monitor-only is used.")
    if monitor_only and assume_yes:
        logging.warning("--yes is ignored when --monitor-only is used.")
    if monitor_only and send_all:
        logging.warning("--send-all is ignored when --monitor-only is used.")
    if monitor_only and send_error:
        logging.warning("--send-error is ignored when --monitor-only is used.")
    if drain_host and monitor_only:
        logging.error("--drain-host cannot be combined with --monitor-only.")
        sys.exit(1)
    if not drain_host:
        if drain_allow_hotspot:
            logging.warning("--drain-allow-hotspot is ignored without --drain-host.")
        if drain_no_disable:
            logging.warning("--drain-no-disable is ignored without --drain-host.")

    lock_handle = None
    coordinator = ExitStack()
    event_token = _report.event_path.set(events_file)
    try:

        # State of an earlier version (/var/lib/wbalancer-v1) must not be
        # silently ignored: refuse to apply, only warn for read-only runs.
        legacy = (_paths.legacy_state_message(os.path.dirname(cooldown_file))
                  if cooldown_file == _paths.COOLDOWN_FILE else None)
        if legacy and (monitor_only or dry_run):
            logging.warning("%s.", legacy)
        elif legacy:
            logging.error("Not running: %s.", legacy)
            sys.exit(1)

        # Apply mode mutates the cluster and the cooldown file; guard against
        # overlapping runs. Monitor (read-only) and dry-run do not need it.
        if not monitor_only and not dry_run:
            try:
                coordinator.enter_context(_locking.shared_apply_lock(
                    os.environ.get("WLB_COORDINATOR_LOCK", _paths.COORDINATOR_LOCK),
                    os.environ.get("WLB_COORDINATOR_FD")))
            except _errors.StateError as e:
                # Expected while a rolling maintenance holds the coordinator
                # lock (e.g. the 10-minute balance cron during a drain).
                logging.error("Not running: %s.", e)
                sys.exit(1)
            ok, lock_handle = _locking.acquire_single_instance_lock(lock_file)
            if not ok:
                logging.error(
                    "Another wlbalancer/loadleveller apply run holds the lock (%s); exiting "
                    "to avoid conflicting migrations.",
                    lock_file,
                )
                sys.exit(1)


        cfg = _config.LoadbalancerConfig.load_config(config)
        cfg.os_cloud = cloud
        # Thresholds from the config file become the live values of this run.
        _config.apply_thresholds(cfg.thresholds)
        _config.apply_migration_timeouts(cfg.migration_timeouts)
        _config.apply_soft_aa_heal(cfg.soft_aa_heal)
        logging.info("Cloud %s; thresholds: %s; migration wait: %s; %s", cfg.os_cloud,
                     cfg.thresholds.describe(), cfg.migration_timeouts.describe(),
                     cfg.soft_aa_heal.describe())
        if not cfg.thresholds.pressure_reachable():
            logging.info(
                "PRESSURE_THRESHOLD %g is not below RAM_HOTSPOT_THRESHOLD %g or CPU_HOTSPOT_THRESHOLD %g: "
                "pressure mode cannot trigger (a host above the average is handled as a hotspot first).",
                cfg.thresholds.pressure, cfg.thresholds.ram_hotspot, cfg.thresholds.cpu_hotspot)
        cfg.cooldown_store = _execution_cooldown.CooldownStore(cooldown_file)
        cfg.cooldown_store.load()
        # Balance waits a full metric window after every migration (barrier);
        # drain does not wait and instead adds the load of VMs that arrived
        # within the metric window to their destination (compensation).
        cfg.settle_mode = (_config.SETTLE_MODE_COMPENSATE if drain_host
                           else _config.SETTLE_MODE_BARRIER)
        if not monitor_only and not dry_run:
            settle_seconds = _config.env_int("RUNTIME_SETTLE_SECONDS", 360)
            validate_settle_window(cfg, settle_seconds)
            cfg.migration_journal = MigrationJournal(
                str(Path(cooldown_file).parent / "operations"), settle_seconds)
        if use_vm_ram_rss:
            _config.enable_default_vm_ram_rss_query(cfg)
        aggregate_names = list(aggregate) if aggregate else None
        # CLI wins over the config file so an operator can pin a group of
        # instances for a single run without editing the shared config.
        exclude_vm_patterns = _selection.load_exclude_vm_patterns(
            exclude_file if exclude_file is not None else cfg.exclude_vms_file
        )

        if monitor_only:
            exit_code = _execution_monitor.monitor_by_aggregate(
                cfg,
                aggregate_names=aggregate_names,
                enforce_server_groups=enforce_server_groups,
                exclude_vm_patterns=exclude_vm_patterns,
            )
            sys.exit(exit_code)

        if drain_host:
            exit_code = _execution_workflow.drain_by_aggregate(
                cfg,
                drain_host=drain_host,
                dry_run=dry_run,
                aggregate_names=aggregate_names,
                enforce_server_groups=enforce_server_groups,
                assume_yes=assume_yes,
                send_all=send_all,
                send_error=send_error,
                exclude_vm_patterns=exclude_vm_patterns,
                allow_hotspot=drain_allow_hotspot,
                disable_host=not drain_no_disable,
            )
            sys.exit(exit_code)

        _execution_workflow.balance_by_aggregate(
            cfg,
            dry_run=dry_run,
            aggregate_names=aggregate_names,
            enforce_server_groups=enforce_server_groups,
            assume_yes=assume_yes,
            send_all=send_all,
            send_error=send_error,
            exclude_vm_patterns=exclude_vm_patterns,
        )

    except _errors.PrometheusError as e:
        logging.error("Prometheus query failed: %s", e)
        sys.exit(_config.ICINGA_UNKNOWN if monitor_only else 1)
    except _errors.PlacementError as e:
        logging.error("Placement capacity query failed: %s", e)
        sys.exit(_config.ICINGA_UNKNOWN if monitor_only else 1)
    except _errors.OpenStackDataError as e:
        logging.error("OpenStack state query failed: %s", e)
        sys.exit(_config.ICINGA_UNKNOWN if monitor_only else 1)
    except _errors.MigrationExecutionError as e:
        logging.error("Migration plan did not complete: %s", e)
        sys.exit(1)
    except _errors.ConfigError as e:
        logging.error("Configuration error: %s", e)
        sys.exit(_config.ICINGA_UNKNOWN if monitor_only else 1)
    except KeyError as e:
        logging.error("Missing required environment variable: %s", e)
        sys.exit(_config.ICINGA_UNKNOWN if monitor_only else 1)
    except Exception as e:
        logging.exception("Unexpected error: %s", e)  # noqa: TRY401 - the message text is part of the operator-facing log
        sys.exit(_config.ICINGA_UNKNOWN if monitor_only else 1)
    finally:
        result_failed = False
        if result_writer is not None:
            exc = sys.exc_info()[1]
            exit_code = (exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int)
                         else 1 if exc else 0)
            try:
                result_writer.finish(exit_code)
            except Exception as result_error:  # noqa: BLE001 - an unpersisted result fails the child; the parent rejects it
                logging.error("Cannot persist child control result: %s", result_error)
                result_failed = True
                # Parent rejects a missing result, even if the operation succeeded.
        current_result.reset(result_token)
        _report.event_path.reset(event_token)
        coordinator.close()
        # Release the single-instance lock (the OS also frees it on exit, but be tidy).
        if lock_handle is not None:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                lock_handle.close()
            except OSError:
                pass
        if result_failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
