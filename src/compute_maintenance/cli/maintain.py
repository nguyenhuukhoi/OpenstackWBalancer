"""cli.maint — extracted domain responsibilities."""
import argparse
import os
import shlex
import subprocess
import sys
import time

import compute_maintenance.adapters.commands as _maint_remote
import compute_maintenance.adapters.compute_cli as _sources_compute
import compute_maintenance.adapters.inventory as _maint_inventory
import compute_maintenance.adapters.maintenance_notifications as _maint_notify
import compute_maintenance.adapters.persistence as _maint_checkpoint
import compute_maintenance.application.maintenance as _maint_runner
import compute_maintenance.config as _config
import compute_maintenance.dependencies as _deps
import compute_maintenance.domain.errors as _errors
import compute_maintenance.paths as _paths
import compute_maintenance.presentation.maintenance_reports as _maint_report
import compute_maintenance.profiles.defaults as _maint_settings
import compute_maintenance.profiles.schema as _maint_profile


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hosts", nargs="*",
                   help="Nova compute host names, processed in this order "
                        "(optional with --inventory and/or -a: default = their common hosts)")
    p.add_argument("-H", "--host", dest="host_opts", action="append", nargs="+", default=[],
                   metavar="HOST",
                   help="host(s) to maintain, same as the positional list: -H a,b or -H a b or "
                        "-H a -H b. All hosts of one run must be in the same aggregate")
    p.add_argument("-i", "--inventory",
                   help="kolla-ansible inventory (e.g. /etc/kolla/multinode); hosts, ansible_host, "
                        f"ansible_user and ansible_ssh_private_key_file come from its [{_maint_settings.INVENTORY_GROUP}] group")
    p.add_argument("--profile", required=True, help="maintenance profile (.yaml/.yml or .json)")
    p.add_argument("--wbalancer", default=None, help="optional compatible child script; default is the installed balancer module")
    p.add_argument("--wlb-config", default=_paths.default_config_file(),
                   help="wbalancer --config file (Prometheus URL/jobs, SMTP); default %(default)s "
                        "(env WLB_CONFIG_PATH replaces /etc/wbalancer/wbalancer.conf)")
    p.add_argument("--wlb-events-file", default=_paths.EVENTS_FILE,
                   help="passed to wbalancer --events-file; read back to name the VMs a failed drain left behind")
    p.add_argument("-a", "--aggregate",
                   help="Nova aggregate: passed to wbalancer -a, and without explicit hosts "
                        "selects the hosts to maintain (its members that are also in --inventory, if given)")
    p.add_argument("--drain-allow-hotspot", action="store_true", default=None)
    p.add_argument("--wlb-extra", default=None, help="extra wbalancer args, quoted")
    p.add_argument("--os-cloud", default=None,
                   help="cloud name in clouds.yaml (openrc environment: its OS_CLOUD_NAME, 'envvars' "
                        "by default); always passed to the wbalancer drain child. Default: $OS_CLOUD, "
                        "else OS_CLOUD in --wlb-config, else 'openstack'")
    p.add_argument("--host-address", action="append", default=[], metavar="HOST=IP",
                   help="SSH/metrics address per host (default: inventory ansible_host, else the host name)")
    p.add_argument("--ssh-user", default=None, help="default: inventory ansible_user, else root")
    p.add_argument("--ssh-key", default=None,
                   help=f"default: inventory ansible_ssh_private_key_file, else {_maint_settings.DEFAULT_SSH_KEY} if it exists, else ssh's "
                        "own keys (~/.ssh/id_*, agent, ~/.ssh/config)")
    p.add_argument("--container-engine", choices=("docker", "podman"), default="docker")
    p.add_argument("--required-containers", nargs="*", default=None)
    p.add_argument("--ignore-containers", nargs="*", default=None)
    p.add_argument("--ignore-units", nargs="*", default=None)
    p.add_argument("--prom-url", help="Prometheus base URL; if set, every target of the host must be up")
    p.add_argument("--drain-timeout", type=int, default=None)
    p.add_argument("--boot-timeout", type=int, default=None)
    p.add_argument("--health-timeout", type=int, default=None)
    p.add_argument("--settle", type=int, default=None,
                   help="continuous healthy duration required before restore")
    p.add_argument("--resume-from", choices=_maint_settings.STAGES[1:],
                   help="continue the FIRST host from this stage using its saved state")
    p.add_argument("--dry-run", action="store_true", help="preflight, pre_drain hooks, wbalancer --dry-run")
    p.add_argument("-y", "--yes", "--auto-approve", dest="yes", action="store_true",
                   help="auto-approve: do not ask before touching each host that needs the work")
    p.add_argument("--notify", choices=sorted(_maint_notify.NOTIFY_EVENTS), default="all",
                   help="e-mail alerts, SMTP settings read like wbalancer (ALERT_EMAIL_TO, SMTP_SERVER, ... "
                        "in --wlb-config or env): fail=STOPPED only, summary=+FINISHED, "
                        "all=+STARTED and each host DONE/SKIPPED (default %(default)s); dry-runs never mail")
    p.add_argument("--state-dir", default=_paths.MAINT_STATE_DIR)
    p.add_argument("--log-dir", default=_paths.MAINT_LOG_DIR)
    p.add_argument("--lock-file", default=_paths.COORDINATOR_LOCK)
    args = p.parse_args(argv)
    # Same rule as wbalancer (--os-cloud > env OS_CLOUD > OS_CLOUD in
    # --wlb-config > default); the drain child then gets it as --os-cloud.
    try:
        args.os_cloud = _config.resolve_os_cloud(args.os_cloud, args.wlb_config)
    except (_errors.ConfigError, OSError) as exc:
        p.error(str(exc))
    # "-H a,b", "-H a b", "-H a -H b" and positional "a,b" all mean [a, b].
    raw = list(args.hosts) + [h for group in args.host_opts for h in group]
    args.hosts = [h.strip() for item in raw for h in item.split(",") if h.strip()]
    return args


def resume_command(explicit_hosts, stage, remaining, argv=None):
    """The original command line, continuing the stopped host at ``stage``
    and then the hosts that were not reached yet (listed explicitly, so it
    also works when the list came from --inventory / -a)."""
    argv = list(sys.argv if argv is None else argv)
    out, skip_next = [], False
    for tok in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok == "--resume-from":
            skip_next = True
            continue
        # Hosts (positional or after -H/--host) are re-added at the end.
        parts = [p for p in tok.split(",") if p]
        if tok.startswith(("--resume-from=", "--host=")) or tok in ("-y", "--yes", "--auto-approve", "-H", "--host") \
                or (parts and all(p in explicit_hosts for p in parts)):
            continue
        out.append(tok)
    prog = argv[0] if argv else "rolling_maint.py"
    # Nothing was changed before/at preflight: a plain re-run is the resume.
    resume = ["--resume-from", stage] if stage and stage in _maint_settings.STAGES[1:] else []
    return " ".join(shlex.quote(t) for t in [prog, *out, *resume, "-y", *remaining])


def main(argv=None):
    args = parse_args(argv)
    explicit_hosts = list(args.hosts)
    cmdline = [sys.argv[0]] + list(sys.argv[1:] if argv is None else argv)
    try:
        profile = _maint_profile.load_profile(args.profile)
    except (ValueError, OSError) as e:
        sys.exit(f"invalid profile: {e}")
    try:
        _maint_profile.merge_settings(args, profile)
    except ValueError as exc:
        sys.exit(f"invalid settings: {exc}")
    log_path = _maint_report.setup_logging(args.log_dir)
    _maint_settings.log.info("log file %s", log_path)
    _maint_settings.log.info("profile %s: reboot=%s hooks=%s", profile["name"], profile["reboot"],
             {k: len(v) for k, v in profile["hooks"].items() if v})

    # Dry-runs are interactive checks: never mail for them.
    notifier = _maint_notify.Notifier(args.wlb_config, "off" if args.dry_run else args.notify,
                        "profile {}".format(profile["name"]), log_path, cmdline)
    _maint_settings.log.info("alerts: off for --dry-run" if args.dry_run else notifier.describe())

    # python-openstackclient is installed by the operator: record exactly which
    # CLI this run uses, and stop before touching anything if there is none.
    # State of an earlier version (/var/lib/wbalancer-v1) must not be ignored.
    legacy = (_paths.legacy_state_message(os.path.dirname(args.state_dir))
              if args.state_dir == _paths.MAINT_STATE_DIR else None)
    if legacy and args.dry_run:
        _maint_settings.log.warning("%s", legacy)
    elif legacy:
        _maint_settings.log.error("STOPPED: %s", legacy)
        notifier.send("STOPPED", "state of an earlier version not moved",
                      [f"Error     : {legacy}", "No host was touched."])
        return 1
    cli_path, cli_version, cli_errors, cli_warnings = _deps.check_cli()
    if cli_errors:
        for error in cli_errors:
            _maint_settings.log.error("STOPPED: %s", error)
        _maint_settings.log.error("%s", _deps.install_hint())
        notifier.send("STOPPED", "openstack CLI unavailable",
                      ["Error     : {}".format("; ".join(cli_errors)), "No host was touched."])
        return 1
    _maint_settings.log.info("openstack CLI %s (python-openstackclient %s, openstacksdk %s)",
                             cli_path, cli_version, _deps.installed_version("openstacksdk"))
    for warning in cli_warnings:
        _maint_settings.log.warning("%s", warning)
    cloud = _sources_compute.Cloud(args.os_cloud, cli=cli_path)
    try:
        conn = _maint_inventory.resolve_hosts(args, cloud)
    except (_maint_checkpoint.StateError, subprocess.TimeoutExpired, ValueError) as e:
        _maint_settings.log.error("STOPPED: %s", e)
        notifier.send("STOPPED", "could not resolve the host list",
                      [f"Error     : {e}", "No host was touched."])
        return 1
    _maint_settings.log.info("hosts (in this order, one at a time):")
    for i, h in enumerate(args.hosts, 1):
        _maint_settings.log.info("  %d. %s  %s@%s  key=%s", i, h, conn[h]["user"], conn[h]["address"],
                 conn[h]["key"] or "ssh default")
    runner = _maint_runner.Runner(
        args, profile, cloud,
        lambda h: _maint_remote.Node(h, conn[h]["address"], conn[h]["user"], conn[h]["key"], args.container_engine),
    )
    runner.notifier = notifier
    host_list = "Hosts     : {}".format(" -> ".join(args.hosts))
    started = time.monotonic()
    try:
        if args.dry_run:
            runner.dry_run()
            return 0
        notifier.send("STARTED", "{} host(s), profile {}".format(len(args.hosts), profile["name"]), [host_list])
        done, skipped = runner.run_all()
        not_run = [h for h in args.hosts if h not in done and h not in skipped]
        _maint_settings.log.info("DONE: %s | skipped: %s", ", ".join(done) or "none", ", ".join(skipped) or "none")
        notifier.send("FINISHED", "{} done, {} skipped{} in {}s".format(
            len(done), len(skipped), f", {len(not_run)} NOT run (operator said no)" if not_run else "",
            int(time.monotonic() - started)),
            [host_list, "Done      : {}".format(", ".join(done) or "none"),
             "Skipped   : {}".format(", ".join(skipped) or "none"),
             "Not run   : {}".format(", ".join(not_run) or "none")])
        return 0
    except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001 - top-level stop handler: every error stops the run with a resume hint
        host, stage = getattr(e, "host", None), getattr(e, "stage", None)
        if isinstance(e, KeyboardInterrupt):
            reason = "interrupted (Ctrl+C)"
            _maint_settings.log.error("STOPPED: interrupted")
        elif isinstance(e, (_maint_checkpoint.StateError, subprocess.TimeoutExpired)):
            reason = str(e)
            _maint_settings.log.error("STOPPED: %s", e)
        else:
            reason = f"unexpected error: {e!r}"
            _maint_settings.log.exception("STOPPED on unexpected error: %s", e)
        lines = [f"Error     : {reason}", host_list]
        if host:
            disabled = getattr(e, "left_disabled", False)
            # A plain re-run continues the stopped host by itself (auto resume),
            # so no --resume-from: only the hosts not finished yet are listed.
            resume = resume_command(explicit_hosts, None, getattr(e, "remaining", [host]), cmdline)
            state_text = "nova-compute observed " + getattr(e, "observed_status", "UNKNOWN").upper()
            give_back = f"openstack compute service set --enable {host} nova-compute"
            _maint_settings.log.error("host %s left at stage '%s' (%s).\n  Fix the cause, then re-run (it continues there):  %s%s",
                      host, stage, state_text, resume,
                      "\n  Or give it back by hand:     " + give_back if disabled else "")
            lines += [f"Host      : {host}", f"Stage     : {stage}",
                      f"State     : {state_text}", "",
                      f"Resume    : {resume}"]
            if disabled:
                lines.append(f"Give back : {give_back}")
        if not args.dry_run:
            notifier.send("STOPPED", f"{host} at {stage}" if host else "before any host",
                          lines + ["", "Last log lines:"] + _maint_report.tail_file(log_path, 25))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
