"""maint.stages — extracted domain responsibilities."""
import contextlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import compute_maintenance.adapters.commands as _maint_remote
import compute_maintenance.adapters.persistence as _maint_checkpoint
import compute_maintenance.domain.maintenance_errors as _maint_errors
import compute_maintenance.presentation.maintenance_reports as _maint_report
import compute_maintenance.profiles.defaults as _maint_settings
import compute_maintenance.profiles.schema as _maint_profile
from compute_maintenance.adapters.protocol import read_result


class StageActions:
    def run_hooks(self, stage, node):
        steps = self.profile["hooks"][stage]
        if not steps:
            return
        _maint_settings.log.info("== Hooks %s on %s (%d step(s))", stage, node.name, len(steps))
        for step in steps:
            state = getattr(self, "_active_state", None)
            key = stage + "/" + step["id"]
            journal = dict(state.data.get("hooks", {})) if state else {}
            previous = journal.get(key, {})
            if previous.get("outcome") == "done":
                continue
            if previous:
                check = step.get("check_cmd")
                if check:
                    command = self.hook_command(check, node)
                    result = (node.ssh(command, timeout=step["timeout"], check=False)
                              if step["run"] == "ssh" else
                              _maint_remote.run(["bash", "-c", command], timeout=step["timeout"], check=False))
                    if result.returncode == 0:
                        self.capture_hook_result(step, result, state)
                        journal[key] = dict(previous, outcome="done", reconciled=_maint_remote.now_stamp())
                        state.save(hooks=journal)
                        continue
                    if result.returncode != 10:
                        raise _maint_errors.CommandOutcomeUnknown("hook reconciliation UNKNOWN: " + key)
                    # Contract rc=10 explicitly proves safe to execute again.
                elif previous.get("outcome") != "failed" or step["idempotency"] != "idempotent":
                    raise _maint_errors.CommandOutcomeUnknown("hook outcome needs reconciliation before replay: " + key)
            if state:
                journal[key] = {"outcome": "pending", "started": _maint_remote.now_stamp()}
                state.save(hooks=journal)
            try:
                result = self.run_step(stage, step, node)
            except _maint_errors.CommandOutcomeUnknown:
                # Preserve pending intent; no blind retry after transport loss.
                raise
            except _maint_errors.StepFailed:
                if state:
                    journal[key]["outcome"] = "failed"
                    state.save(hooks=journal)
                raise
            if state:
                self.capture_hook_result(step, result, state)
                journal[key]["outcome"] = "done"
                state.save(hooks=journal)

    def capture_hook_result(self, step, result, state):
        if state and step.get("capture"):
            value = result.stdout.strip()
            if not value or len(value) > 65536:
                raise _maint_errors.StepFailed("invalid or oversized captured hook result")
            state.save(**{step["capture"]: value})

    def hook_command(self, template, node):
        cmd = _maint_profile.render(template, node, self.profile)
        state = getattr(self, "_active_state", None)
        if state:
            for key in ("expected_kernel", "package_plan"):
                value = state.data.get(key)
                if value is not None:
                    env_name = "WLB_" + key.upper()
                    cmd = env_name + "=" + shlex.quote(value) + "; export " + env_name + ";\n" + cmd
        return cmd

    def run_step(self, stage, step, node):
        cmd = self.hook_command(step["cmd"], node)
        for attempt in range(step["retries"] + 1):
            _maint_settings.log.info("  [%s] %s (%s, attempt %d/%d)", stage, step["name"], step["run"], attempt + 1, step["retries"] + 1)
            _maint_settings.log.debug("    cmd: %s", cmd)
            try:
                if step["run"] == "ssh":
                    p = node.ssh(cmd, timeout=step["timeout"], check=False)
                else:
                    p = _maint_remote.run(["bash", "-c", cmd], timeout=step["timeout"], check=False)
            except subprocess.TimeoutExpired:
                raise _maint_errors.CommandOutcomeUnknown(
                    "hook '{}' timed out; remote completion UNKNOWN".format(step["id"])) from None
            else:
                rc, out = p.returncode, (p.stdout + p.stderr).strip()
                if step["run"] == "ssh" and rc == 255:
                    raise _maint_errors.CommandOutcomeUnknown("SSH transport failed; hook outcome UNKNOWN")
            for line in out.splitlines()[-20:]:
                _maint_settings.log.info("    | %s", line)
            if rc == 0:
                return p
            if stage == "pre_drain" and rc == _maint_profile.SKIP_EXIT_CODE:
                raise _maint_errors.SkipHost("hook '{}' says {} does not need this maintenance".format(step["name"], node.name))
            if step["idempotency"] != "idempotent":
                break
            if attempt < step["retries"]:
                _maint_settings.log.warning("    rc=%s, retrying in %ss", rc, step["retry_delay"])
                self.sleep(step["retry_delay"])
        raise _maint_errors.StepFailed("hook '{}' ({}) failed rc={}".format(step["name"], stage, rc))

    def preflight(self, hosts, recovering_host=None):
        _maint_settings.log.info("== Preflight on %s", ", ".join(hosts))
        problems, known, by_aggregate = [], set(), {}
        self._aggregates = None  # re-read membership before every host
        selected = set(hosts)
        for host in hosts:
            try:
                aggregate = self.aggregate_for(host)
                selected.update(self._aggregates[aggregate])
            except _maint_errors.StepFailed as exc:
                problems.append(str(exc))
        for row in self.cloud.nova_services():
            host = row.get("Host")
            if host not in selected:
                continue
            known.add(host)
            if row.get("State") != "up" and host != recovering_host:
                problems.append("nova-compute on {} is {}".format(host, row.get("State")))
            if row.get("Status") != "enabled":
                if host in hosts:
                    try:
                        node = self.node_factory(host)
                        state = _maint_checkpoint.HostState(self.args.state_dir, host)
                        state.load()
                        state.validate_identity(self.checkpoint_identity(node))
                        self.require_owned(node, state)
                        if self.live_owner(host):
                            raise _maint_errors.StepFailed("another live process owns " + host)
                    except _maint_checkpoint.StateError as exc:
                        problems.append(str(exc))
                else:
                    _maint_settings.log.warning("  %s is disabled -> one destination less for drains", host)
        for host in hosts:
            if host not in known:
                problems.append(f"{host} has no nova-compute service")
                continue
            dead = self.cloud.agents_dead(host) if host != recovering_host else []
            if dead:
                problems.append("{} network agent(s) not alive: {}".format(host, ", ".join(dead)))
            try:
                agg = self.aggregate_for(host)
                by_aggregate.setdefault(agg, []).append(host)
                _maint_settings.log.info("  %s drains within aggregate %s", host, agg)
            except _maint_errors.StepFailed as e:
                problems.append(str(e))
        if len(by_aggregate) > 1:
            # One run = one aggregate: hosts of one aggregate absorb each
            # other's VMs in turn, so their order and capacity are planned
            # together. Mixing aggregates in one run is almost always a typo.
            problems.append("hosts span several aggregates ({}); run one aggregate per run".format(
                "; ".join("{}: {}".format(a, ", ".join(h)) for a, h in sorted(by_aggregate.items()))))
        # Query Nova by host, rather than block on unrelated tenants/aggregates.
        migrating = [server for host in sorted(selected) for server in self.cloud.servers_on(host)
                     if self.server_status(server) == "MIGRATING"]
        if migrating:
            problems.append(f"{len(migrating)} server(s) already MIGRATING")
        if problems:
            raise _maint_errors.StepFailed("preflight failed:\n  - " + "\n  - ".join(problems))
        _maint_settings.log.info("  preflight OK")

    def disable(self, node, state):
        svc = self.cloud.nova_service(node.name)
        if svc.get("Status") == "disabled":
            self.require_owned(node, state)
            return
        if svc.get("Status") != "enabled":
            raise _maint_errors.StepFailed("unknown Nova service status")
        self.refuse_error_vms(node.name)
        self.refuse_locked_vms(node.name)
        reason = "rolling-maint v1:" + state.data["run_id"]
        state.save(disabled_reason=reason)  # intent must survive an RPC/process crash
        self.cloud.set_nova_enabled(node.name, False, reason)
        self.require_owned(node, state)
        _maint_settings.log.info("  %s nova-compute disabled (reason=%s)", node.name, reason)

    def refuse_error_vms(self, host):
        """wbalancer never moves a VM in ERROR, so it would stay on the host
        through the maintenance -- and its guest may still be running (Nova
        state stuck in ERROR after a failed operation). Stop BEFORE the host
        is disabled and let the operator decide."""
        errors = [s for s in self.cloud.servers_on(host) if s.get("Status") == "ERROR"]
        if not errors:
            return
        lines = []
        for s in errors:
            try:
                d = self.cloud.server_show(s.get("ID"))
            except _maint_errors.StepFailed:
                d = {}
            power = d.get("OS-EXT-STS:power_state")
            running = str(power) in ("1", "Running", "RUNNING")
            fault = d.get("fault") or {}
            fault = fault.get("message") if isinstance(fault, dict) else fault
            lines.append("  - {} ({}) ERROR, power_state={}{}{}".format(
                s.get("Name"), s.get("ID"), power,
                " (guest still RUNNING: a reboot would take it down)" if running else "",
                f", fault: {fault}" if fault else ""))
            lines.append("      -> check: openstack server show {} -c fault -c OS-EXT-STS:power_state "
                         "-c OS-EXT-STS:task_state; if the guest is fine: openstack server set --state "
                         "active {} (then wbalancer can move it); else fix/stop it, then re-run".format(
                             s.get("ID"), s.get("ID")))
        raise _maint_errors.StepFailed("{} holds {} VM(s) in ERROR that wbalancer will not move; host NOT disabled:\n{}".format(
            host, len(errors), "\n".join(lines)))

    def refuse_locked_vms(self, host):
        """A locked ACTIVE VM cannot be live-migrated, so the drain would stop
        half way with the host already disabled. Stop BEFORE disabling."""
        candidates = [s for s in self.cloud.locked_servers_on(host)
                      if self.server_status(s) == "ACTIVE"]
        # The list filter is only a hint (some clients ignore it); confirm
        # each candidate from Nova's own locked field.
        locked = [s for s in candidates if self.cloud.server_locked(s.get("ID"))]
        if not locked:
            return
        lines = []
        for s in locked:
            lines.append("  - {} ({}) ACTIVE, locked".format(s.get("Name"), s.get("ID")))
            lines.append("      -> ask the owner to unlock it, or as admin: openstack server unlock {} "
                         "(then re-run)".format(s.get("ID")))
        raise _maint_errors.StepFailed("{} holds {} locked running VM(s) that cannot be live-migrated; "
                                       "host NOT disabled:\n{}".format(host, len(locked), "\n".join(lines)))

    def refuse_running_left(self, host, left_servers):
        """Never reboot a host under a RUNNING guest. wbalancer leaves an
        ACTIVE VM behind when it cannot plan it (no Prometheus metrics,
        exporter gap, unreadable flavor, ...): that is fine for SHUTOFF VMs,
        not for one a customer is using."""
        running = [s for s in left_servers if s.get("Status") == "ACTIVE"]
        if not running:
            return
        lines = []
        for s in running:
            lines.append("  - {} ({}) ACTIVE".format(s.get("Name"), s.get("ID")))
            reason = _maint_report.parse_drain_output(getattr(self, "wlb_output", []))["unmovable"].get(s.get("ID"))
            if reason:
                lines.append("      wbalancer: reason={} -> {}".format(
                    reason, _maint_report.explain_unmovable(reason, s.get("ID"))))
            lines.append("      -> fix the metrics and re-run, or move it by hand: openstack "
                         "--os-compute-api-version 2.30 server migrate --live-migration {}".format(s.get("ID")))
        raise _maint_errors.StepFailed("{}: wbalancer would leave {} RUNNING VM(s) on the host; not rebooting under "
                         "them:\n{}".format(host, len(running), "\n".join(lines)))

    @staticmethod
    def server_status(server):
        return str(server.get("Status") or "UNKNOWN").strip().upper()

    def reboot_profile(self):
        """Whether this profile contains a runner-managed reboot policy."""
        return self.profile["reboot"] in ("always", "if_required")

    def refuse_unsafe_reboot_states(self, host, servers, allowed_statuses, phase):
        """Fail closed on VM states that must not be carried through reboot."""
        if not self.reboot_profile():
            return
        unsafe = [s for s in servers if self.server_status(s) not in allowed_statuses]
        if not unsafe:
            return
        lines = ["  - {} ({}) {}".format(s.get("Name"), s.get("ID"), self.server_status(s))
                 for s in unsafe]
        raise _maint_errors.StepFailed(
            "{}: {} VM(s) have a state unsafe for reboot {}. Allowed here: {}:\n{}".format(
                host, len(unsafe), phase, ", ".join(sorted(allowed_statuses)), "\n".join(lines)
            )
        )

    def validate_saved_left(self, node, state, phase):
        """Re-query Nova; validate saved IDs and, for reboot profiles, states."""
        servers = self.cloud.servers_on(node.name)
        allowed_ids = set(state.data.get("left_on_host") or [])
        blocking = [s for s in servers if s.get("ID") not in allowed_ids]
        if blocking:
            names = ", ".join("{}({})".format(s.get("Name"), self.server_status(s)) for s in blocking)
            raise _maint_errors.StepFailed(f"{node.name} {phase} still holds {len(blocking)} unexpected server(s): {names}")
        self.refuse_unsafe_reboot_states(
            node.name, servers, _maint_settings.REBOOT_SAFE_LEFT_VM_STATUSES, phase
        )
        return servers

    def wbalancer_left_behind(self, host, since):
        """wbalancer decides which VMs move. Its exit 2 also means "drain
        feasible for every live-migratable VM, but N VM(s) will remain" (it
        does not move e.g. SHUTOFF VMs). Return those VM ids when that is the
        only reason for the 2 -- no capacity/destination problem and, after a
        real drain, nothing else left on the host -- else None."""
        summary = (getattr(self, "_child_result", None) or {}).get("summary")
        if not summary or summary.get("result") == "drain_infeasible" or not summary.get("drain_feasible", True) \
                or summary.get("drain_blocked_vm_ids") or summary.get("drain_infeasible_reasons"):
            return None
        left = summary.get("drain_unmovable_vm_ids") or []
        if not left:
            return None
        servers = self.cloud.servers_on(host)
        if summary.get("result") != "drain_dry_run" and any(s.get("ID") not in left for s in servers):
            return None  # a real drain must have moved everything it planned
        if any(s.get("Status") == "ERROR" for s in servers if s.get("ID") in left):
            # Host disabled before this run (auto-continue skipped the disable
            # check): an ERROR VM must still stop the run.
            self.refuse_error_vms(host)
        left_servers = [s for s in servers if s.get("ID") in left]
        if self.reboot_profile():
            self.refuse_unsafe_reboot_states(
                host, left_servers, _maint_settings.REBOOT_SAFE_LEFT_VM_STATUSES, "after drain"
            )
        else:
            # Preserve the existing non-reboot profile behavior: only a
            # running ACTIVE guest is forbidden here (ERROR was checked above).
            self.refuse_running_left(host, left_servers)
        _maint_settings.log.warning("  %s: wbalancer leaves %d VM(s) on the host (not live-migratable), continuing: %s",
                    host, len(left), ", ".join("{} {}".format(s.get("Name"), s.get("Status"))
                                               for s in servers if s.get("ID") in left) or ", ".join(left))
        return left

    def wbalancer_cmd(self, host, dry_run):
        a = self.args
        child = [a.wbalancer] if a.wbalancer else ["-m", "compute_maintenance.cli.balance"]
        cmd = [sys.executable, *child, "--config", a.wlb_config, "--drain-host", host,
               "-a", self.aggregate_for(host)]
        # Always the parent's resolved cloud, also for a custom --wbalancer
        # child: without the flag the child would resolve its own cloud (e.g.
        # OS_CLOUD from the config file) while reporting it under our identity.
        # Every child also gets --run-id/--result-file, so it must be v1 anyway.
        cmd += ["--os-cloud", a.os_cloud]
        # Pin the events file so drain_failure_details() reads what wbalancer wrote.
        cmd += ["--events-file", a.wlb_events_file]
        if a.drain_allow_hotspot:
            cmd.append("--drain-allow-hotspot")
        if not dry_run and a.notify != "off":
            # wbalancer mails each failed/stuck/timed-out VM migration itself,
            # with the per-VM detail the runner's STOPPED mail summarises.
            cmd.append("--send-error")
        cmd += shlex.split(a.wlb_extra or "")
        cmd.append("--dry-run" if dry_run else "-y")
        return cmd

    def run_wbalancer(self, host, dry_run):
        cmd = self.wbalancer_cmd(host, dry_run)
        invocation = uuid.uuid4().hex
        result_path = Path(self.args.log_dir) / "results" / (invocation + ".json")
        expected = {"run_id": invocation, "cloud": self.args.os_cloud,
                    "aggregate": self.aggregate_for(host), "host": host, "mode": "dry-run" if dry_run else "apply"}
        cmd += ["--run-id", invocation, "--result-file", str(result_path)]
        self._child_result = None
        _maint_settings.log.info("  $ %s", " ".join(shlex.quote(c) for c in cmd))
        # Stream line by line: a drain takes many minutes and the operator
        # must see each migration as it happens, not one dump at the end.
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        fd = getattr(self, "_coordinator_fd", None)
        if not dry_run:
            if fd is None:
                raise _maint_errors.StepFailed("drain requires the parent coordinator descriptor")
            env["WLB_COORDINATOR_FD"] = str(fd)
            env["WLB_COORDINATOR_LOCK"] = self.args.lock_file
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env,
                             pass_fds=(fd,) if fd is not None else ())
        timed_out = threading.Event()

        def on_timeout():
            timed_out.set()
            p.kill()

        timer = threading.Timer(self.args.drain_timeout, on_timeout)
        timer.start()
        self.wlb_output = []  # kept for drain_failure_details(): reasons are only in stdout
        try:
            for line in p.stdout:
                _maint_settings.log.info("  | %s", line.rstrip())
                self.wlb_output.append(line.rstrip("\n"))
                del self.wlb_output[:-5000]
            rc = p.wait()
        finally:
            timer.cancel()
            if p.poll() is None:  # Ctrl+C / exception while reading
                p.kill()
                p.wait()
        if timed_out.is_set():
            raise subprocess.TimeoutExpired(cmd, self.args.drain_timeout)
        self._child_result = read_result(result_path, expected, rc)
        return rc

    def drain_dry_run(self, node):
        _maint_settings.log.info("== Drain %s (dry-run)", node.name)
        self.refuse_unsafe_reboot_states(
            node.name, self.cloud.servers_on(node.name),
            _maint_settings.REBOOT_PRE_DRAIN_VM_STATUSES, "before drain"
        )
        since = time.time()
        rc = self.run_wbalancer(node.name, True)
        if rc == 2 and self.wbalancer_left_behind(node.name, since) is not None:
            rc = 0
        if rc != 0:
            raise _maint_errors.StepFailed(f"wbalancer dry-run for {node.name} exit {rc} (2 = cannot drain){self.drain_failure_details(node.name, since)}")
        summary = self._child_result.get("summary")
        return summary or {}

    def drain(self, node, state=None):
        _maint_settings.log.info("== Drain %s (execute)", node.name)
        self.refuse_unsafe_reboot_states(
            node.name, self.cloud.servers_on(node.name),
            _maint_settings.REBOOT_PRE_DRAIN_VM_STATUSES, "before drain"
        )
        since = time.time()
        try:
            rc = self.run_wbalancer(node.name, False)
        except subprocess.TimeoutExpired:
            raise _maint_errors.StepFailed(f"wbalancer drain of {node.name} killed after {self.args.drain_timeout}s (--drain-timeout){self.drain_failure_details(node.name, since)}") from None
        left = self.wbalancer_left_behind(node.name, since) if rc == 2 else None
        if left is not None:
            rc = 0
        if state is not None:
            state.save(left_on_host=left or [])  # verify accepts exactly these
        if rc != 0:
            raise _maint_errors.StepFailed(f"wbalancer drain of {node.name} exit {rc} (1 = failed/aborted, 2 = VMs left){self.drain_failure_details(node.name, since)}")

    def drain_failure_details(self, host, since):
        """Which VMs could not be moved, and why -- for the error and the
        STOPPED mail. Best effort: never raises, a lookup failure is noted."""
        lines = []
        names, statuses = {}, {}
        try:
            servers = self.cloud.servers_on(host)
            for s in servers:
                names[s.get("ID")] = s.get("Name")
                statuses[s.get("ID")] = s.get("Status")
            lines.append(f"VMs still on {host}: {len(servers)}")
            lines += ["  - {} ({}) {}".format(s.get("Name"), s.get("ID"), s.get("Status")) for s in servers]
        except Exception as e:  # noqa: BLE001 - reporting must not mask the drain error
            lines.append(f"VMs still on {host}: lookup failed ({e})")

        why = _maint_report.parse_drain_output(getattr(self, "wlb_output", []))
        failed, summary = _maint_report.read_drain_events(self.args.wlb_events_file, host, since)
        if failed:
            lines.append(f"Failed migrations ({self.args.wlb_events_file}, newest last):")
            for ev in failed:
                vm = ev.get("vm_name") or names.get(ev.get("vm_id")) or ev.get("vm_id")
                lines.append("  - {} ({}) -> {}: {}{}".format(
                    vm, ev.get("vm_id"), ev.get("dst_host"), ev.get("result"),
                    " | " + str(ev["detail"]) if ev.get("detail") else ""))
        if summary:
            lines.append("wbalancer drain result: {}{}".format(
                summary.get("result"), " | " + str(summary["detail"]) if summary.get("detail") else ""))
            for key, label in (("drain_infeasible_reasons", "infeasible"),
                               ("drain_unmovable_vm_ids", "unmovable"),
                               ("drain_blocked_vm_ids", "blocked"),
                               ("drain_affinity_conflicts", "affinity conflict")):
                for item in summary.get(key) or []:
                    if isinstance(item, str):
                        name = names.get(item) or why["names"].get(item)
                        text = f"{name} ({item})" if name else item
                    else:
                        text = json.dumps(item, sort_keys=True)
                    if key == "drain_unmovable_vm_ids":
                        reason = why["unmovable"].get(item, "unknown")
                        lines.append("  - not live-migratable: {} reason={}{}".format(
                            text, reason, f", Nova status {statuses[item]}" if statuses.get(item) else ""))
                        lines.append(f"      -> {_maint_report.explain_unmovable(reason, item, statuses.get(item))}")
                    elif key == "drain_blocked_vm_ids":
                        lines.append(f"  - no destination host: {text}")
                        for dst in why["blocked"].get(item, []):
                            lines.append(f"      {dst}")
                    else:
                        lines.append(f"  - {label}: {text}")
        if not failed and not summary:
            lines.append(f"no wbalancer event for {host} since the drain started ({self.args.wlb_events_file})")
        return "\n" + "\n".join(lines)  # main() logs it once with STOPPED

    def verify(self, node, state):
        servers = self.validate_saved_left(node, state, "at verify")
        if servers:
            _maint_settings.log.info("  %s: %d VM(s) left on the host by wbalancer: %s", node.name, len(servers),
                     ", ".join("{}({})".format(s.get("Name"), self.server_status(s)) for s in servers))
        svc = self.cloud.nova_service(node.name)
        if svc.get("Status") != "disabled":
            raise _maint_errors.StepFailed("{} nova-compute is {} after drain, expected disabled".format(node.name, svc.get("Status")))
        _maint_settings.log.info("  %s is disabled and %s", node.name,
                 "holds only what wbalancer left" if servers else "empty")

    def baseline(self, node, state):
        failed = sorted(node.failed_units())
        containers = sorted(n for n in node.containers() if not _maint_remote.is_dynamic_container(n))
        missing = [c for c in self.args.required_containers if c not in containers]
        if missing:
            raise _maint_errors.StepFailed("{} lacks required container(s): {}".format(node.name, ", ".join(missing)))
        agents = self.cloud.agent_snapshot(node.name)
        cinder = self.cloud.cinder_snapshot(node.name)
        prometheus = self.prom_snapshot(node.address) if self.args.prom_url else {}
        if not agents or not all(agents.values()) or not all(cinder.values()) or not all(prometheus.values()):
            raise _maint_errors.StepFailed("baseline has missing/down network, cinder or Prometheus components")
        state.save(baseline={"failed_units": failed, "containers": containers,
                            "agents": sorted(agents), "cinder": sorted(cinder), "prometheus": sorted(prometheus)},
                   baseline_boot_id=node.boot_id())
        _maint_settings.log.info("  baseline: %d containers; pre-existing failed units: %s",
                 len(containers), ", ".join(failed) or "none")

    def reboot_needed(self, node):
        policy = self.profile["reboot"]
        if policy == "always":
            return True, "policy always"
        if policy == "never":
            return False, "policy never"
        cmd = _maint_profile.render(self.profile["reboot_required_cmd"], node, self.profile)
        rc = node.ssh(cmd, check=False).returncode
        if rc not in (0, 1):
            raise _maint_errors.StepFailed(f"reboot requirement UNKNOWN: command failed rc={rc}")
        return rc == 0, f"reboot_required_cmd rc={rc}"

    def reboot(self, node, state):
        needed, why = self.reboot_needed(node)
        if not needed:
            _maint_settings.log.info("== Reboot %s skipped (%s)", node.name, why)
            state.save(rebooted=False)
            return
        # Hooks run between verify and reboot.  Re-check immediately before
        # power cycling so a SHUTOFF/SHELVED VM started in that window stops us.
        self.validate_saved_left(node, state, "immediately before reboot")
        # Nova inventory is not proof: a guest whose record points elsewhere
        # (failed migration cleanup, split-brain) would die with the reboot.
        guests = node.running_guest_processes()
        if guests:
            raise _maint_errors.StepFailed(
                f"{node.name} still runs {guests} QEMU guest process(es) although Nova lists no running VM; "
                "not rebooting. Check 'virsh list --all' in nova_libvirt")
        _maint_settings.log.info("== Reboot %s (%s, %s)", node.name, node.address, why)
        before = node.boot_id()
        state.save(boot_id_before=before, rebooted=True)
        node.reboot()
        started = time.monotonic()

        def came_back():
            try:
                now = node.boot_id()
            except (_maint_errors.StepFailed, subprocess.TimeoutExpired):
                return False, "ssh not reachable yet"
            if now == before:
                return False, "still the old boot (shutdown in progress)"
            return True, now

        new = _maint_remote.wait_until(f"{node.name} to boot with a new boot_id", came_back,
                         self.args.boot_timeout, interval=15)
        took = int(time.monotonic() - started)
        state.save(boot_id_after=new, boot_seconds=took)
        _maint_settings.log.info("  %s is back after %ds, boot_id %s -> %s", node.name, took, before[:8], new[:8])

    def resume_reboot_wait(self, node, state):
        """--resume-from reboot: the reboot was sent, only wait for the new boot."""
        before = state.data.get("boot_id_before")
        if not before:
            _maint_settings.log.info("  no reboot recorded for %s yet: rebooting now", node.name)
            return self.reboot(node, state)
        # Probe only (result unused): unreachable while it reboots, wait below.
        with contextlib.suppress(_maint_errors.StepFailed, subprocess.TimeoutExpired):
            node.boot_id()
        def new_boot():
            current = node.boot_id()
            return current != before, current
        after = _maint_remote.wait_until(f"{node.name} to boot with a new boot_id (no duplicate reboot submit)",
                   new_boot, self.args.boot_timeout, interval=15)
        state.save(boot_id_after=after, reboot_reconciled=True)
        _maint_settings.log.info("  %s booted with a new boot_id", node.name)

    def prom_snapshot(self, address):
        import re
        selector = 'up{instance=~"' + re.escape(address).replace("\\", "\\\\") + ':[0-9]+"}'
        def query(expression):
            url = self.args.prom_url.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": expression})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=10) as response:
                payload = json.load(response)
            if payload.get("status") != "success":
                raise _maint_errors.StepFailed("Prometheus health query failed")
            return payload["data"]["result"]
        rows = query(selector)
        ages = {str(row["metric"].get("job")) + "|" + str(row["metric"].get("instance")): float(row["value"][1])
                for row in query("time() - timestamp(" + selector + ")")}
        result = {}
        for row in rows:
            key = str(row["metric"].get("job")) + "|" + str(row["metric"].get("instance"))
            if key in result:
                raise _maint_errors.StepFailed("duplicate Prometheus health target: " + key)
            result[key] = row["value"][1] == "1" and 0 <= ages.get(key, float("inf")) <= 120
        if not result:
            raise _maint_errors.StepFailed("no Prometheus health targets for " + address)
        return result

    def prom_targets_down(self, address):
        return [key + " down/stale" for key, ok in self.prom_snapshot(address).items() if not ok]

    def health_once(self, node, baseline):
        from compute_maintenance.domain.health import HealthReport
        report = HealthReport()
        report.observe("systemd", lambda: sorted(node.failed_units() - set(baseline["failed_units"]) - set(self.args.ignore_units)))
        def containers_check():
            current = node.containers()
            expected = (set(baseline["containers"]) | set(self.args.required_containers)) - set(self.args.ignore_containers)
            problems = [name + " missing" for name in sorted(expected - set(current))]
            for name, status in current.items():
                if name not in self.args.ignore_containers and (not status.startswith("Up") or "unhealthy" in status or "health: starting" in status):
                    problems.append(name + ": " + status)
            return problems
        report.observe("containers", containers_check)
        report.observe("nova", lambda: [] if self.cloud.nova_service(node.name).get("State") == "up" else ["compute is not up"])
        def expected_set(current, expected):
            return [key + " missing/down" for key in sorted(set(expected) | set(current)) if not current.get(key, False)]
        report.observe("network", lambda: expected_set(self.cloud.agent_snapshot(node.name), baseline.get("agents", [])))
        report.observe("cinder", lambda: expected_set(self.cloud.cinder_snapshot(node.name), baseline.get("cinder", [])))
        if self.args.prom_url:
            report.observe("prometheus", lambda: expected_set(self.prom_snapshot(node.address), baseline.get("prometheus", [])))
        self.last_health_report = report
        return report.ok, report.detail

    def health(self, node, state):
        baseline = state.data.get("baseline")
        if not baseline:
            raise _maint_errors.StepFailed(f"no baseline saved for {node.name}; cannot judge health")
        _maint_settings.log.info("== Health gate %s: continuous healthy window %ss", node.name, self.args.settle)
        clock = getattr(self, "monotonic", time.monotonic)
        deadline = clock() + self.args.health_timeout + self.args.settle
        healthy_since, stable_boot = None, None
        while clock() <= deadline:
            ok, detail = self.health_once(node, baseline)
            try:
                boot = node.boot_id() if ok else None
            except Exception as exc:  # noqa: BLE001 - an unreadable boot id is an unhealthy probe
                ok, detail, boot = False, str(exc), None
            now = clock()
            if not ok:
                healthy_since, stable_boot = None, None
            elif healthy_since is None or stable_boot != boot:
                healthy_since, stable_boot = now, boot
            if ok and now - healthy_since >= self.args.settle:
                state.save(healthy_boot_id=stable_boot, healthy_duration=now - healthy_since)
                return
            _maint_settings.log.info("  health %s: %s", node.name, detail)
            self.sleep(min(10, max(0.1, deadline - now)))
        raise _maint_errors.StepFailed("health never remained stable for the required window")

    def restore(self, node, state):
        svc = self.cloud.nova_service(node.name)
        if svc.get("Status") == "disabled":
            self.require_owned(node, state)
            state.save(restore_intent=True)
            self.cloud.set_nova_enabled(node.name, True)
        elif svc.get("Status") != "enabled":
            raise _maint_errors.StepFailed("unknown Nova service status at restore")
        elif not state.data.get("restore_intent"):
            raise _maint_errors.StepFailed("enabled service has no restore intent from this run")

        def enabled_up():
            s = self.cloud.nova_service(node.name)
            return s.get("Status") == "enabled" and s.get("State") == "up", "{}/{}".format(s.get("Status"), s.get("State"))

        _maint_remote.wait_until(f"{node.name} enabled/up", enabled_up, 120)
        _maint_settings.log.info("  %s back in service", node.name)

    def stage_fn(self, stage, node, state, resumed_at):
        if stage == "preflight":
            return lambda: self.preflight(self.args.hosts)
        if stage in _maint_profile.HOOK_STAGES:
            return lambda: self.run_hooks(stage, node)
        if stage == "reboot" and resumed_at == "reboot":
            return lambda: self.resume_reboot_wait(node, state)
        return {
            "disable": lambda: self.disable(node, state),
            "drain": lambda: self.drain(node, state),
            "verify": lambda: self.verify(node, state),
            "baseline": lambda: self.baseline(node, state),
            "reboot": lambda: self.reboot(node, state),
            "health": lambda: self.health(node, state),
            "restore": lambda: self.restore(node, state),
        }[stage]

    def require_owned(self, node, state):
        svc = self.cloud.nova_service(node.name)
        expected = "rolling-maint v1:" + state.data["run_id"]
        if svc.get("Status") != "disabled" or svc.get("Disabled Reason") != expected:
            raise _maint_errors.StepFailed(f"{node.name} must be disabled by this exact run ({expected})")

    def guard_stage(self, stage, node, state):
        index = _maint_settings.STAGES.index(stage)
        if stage in ("preflight", "pre_drain") and self.cloud.nova_service(node.name).get("Status") != "enabled":
            raise _maint_errors.StepFailed("early stages require an enabled service")
        if stage == "disable" and self.cloud.nova_service(node.name).get("Status") == "disabled":
            self.require_owned(node, state)
        if _maint_settings.STAGES.index("drain") <= index < _maint_settings.STAGES.index("restore"):
            self.require_owned(node, state)
        if _maint_settings.STAGES.index("verify") <= index <= _maint_settings.STAGES.index("restore"):
            self.validate_saved_left(node, state, "before " + stage)
            self.refuse_running_left(node.name, self.cloud.servers_on(node.name))
        if stage in ("maintenance", "reboot", "health", "post_boot", "restore") and not state.data.get("baseline"):
            raise _maint_errors.StepFailed("no baseline for " + stage)
        if stage == "maintenance" and node.boot_id() != state.data.get("baseline_boot_id"):
            raise _maint_errors.StepFailed("boot changed since baseline; reconcile before maintenance")
        if stage in ("post_boot", "restore") and node.boot_id() != state.data.get("healthy_boot_id"):
            raise _maint_errors.StepFailed("boot changed since health gate")
        if stage == "restore":
            if not {"health", "post_boot"}.issubset(state.data.get("completed", [])):
                raise _maint_errors.StepFailed("restore requires completed health and post_boot")
            ok, detail = self.health_once(node, state.data["baseline"])
            if not ok:
                raise _maint_errors.StepFailed("restore health recheck failed: " + detail)
            svc = self.cloud.nova_service(node.name)
            if svc.get("Status") == "disabled":
                self.require_owned(node, state)
            elif svc.get("Status") != "enabled" or not state.data.get("restore_intent"):
                raise _maint_errors.StepFailed("enabled service has no restore intent from this run")
        if stage == "post_restore":
            svc = self.cloud.nova_service(node.name)
            if (svc.get("Status"), svc.get("State")) != ("enabled", "up"):
                raise _maint_errors.StepFailed("post_restore requires enabled/up service")
