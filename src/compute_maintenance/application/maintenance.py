"""maint.runner — extracted domain responsibilities."""
import os
import socket
import time
import uuid

import compute_maintenance.adapters.commands as _maint_remote
import compute_maintenance.adapters.locking as _locking
import compute_maintenance.adapters.maintenance_notifications as _maint_notify
import compute_maintenance.adapters.persistence as _maint_checkpoint
import compute_maintenance.application.stages as _maint_stages
import compute_maintenance.domain.maintenance_errors as _maint_errors
import compute_maintenance.profiles.defaults as _maint_settings
import compute_maintenance.profiles.schema as _maint_profile


class Runner(_maint_stages.StageActions):
    def __init__(self, args, profile, cloud, node_factory):
        self.args = args
        self.profile = profile
        self.cloud = cloud
        self.node_factory = node_factory
        self.sleep = time.sleep
        self.monotonic = time.monotonic
        self._aggregates = None  # cached {name: [hosts]}
        self.notifier = _maint_notify._NullNotifier()
        self._lock_held = False
        self._active_state = None

    def aggregate_for(self, host):
        """The single aggregate ``host`` drains within; StepFailed otherwise.

        With -a it is that aggregate. Without -a it is looked up here, so a
        host in zero or several aggregates fails in preflight -- before it is
        disabled -- instead of inside wbalancer after the disable.
        """
        if self._aggregates is None:
            self._aggregates = self.cloud.aggregate_map()
        if self.args.aggregate:
            owning = {self.args.aggregate: self._aggregates.get(self.args.aggregate)}
            if owning[self.args.aggregate] is None:
                raise _maint_errors.StepFailed(f"aggregate {self.args.aggregate} does not exist")
            if host not in owning[self.args.aggregate]:
                raise _maint_errors.StepFailed(f"{host} is not in aggregate {self.args.aggregate}")
        else:
            owning = {name: hosts for name, hosts in self._aggregates.items() if host in hosts}
            if not owning:
                raise _maint_errors.StepFailed(f"{host} is in no aggregate: wbalancer has nowhere to drain it")
            if len(owning) > 1:
                raise _maint_errors.StepFailed("{} is in several aggregates ({}); pass -a to choose one".format(
                    host, ", ".join(sorted(owning))))
        name, members = next(iter(owning.items()))
        if len(members) < 2:
            raise _maint_errors.StepFailed(f"aggregate {name} has only {host}: nowhere to drain it")
        return name

    def live_owner(self, host):
        """PID of another rolling-maint process on this machine still working
        on ``host`` (from its state file), else None."""
        prev = _maint_checkpoint.HostState(self.args.state_dir, host).load()
        pid = prev.get("pid")
        if prev.get("result") != "running" or not pid or pid == os.getpid() \
                or prev.get("runner") != socket.gethostname():
            return None
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read()
        except OSError:
            return None  # process gone (killed, SSH dropped, ...)
        names = {os.path.basename(arg) for arg in cmdline.split(b"\0")}
        return pid if names.intersection({b"rolling_maint.py", b"rolling-maint",
                                         b"compute_maintenance.cli.maintain"}) else None

    def auto_resume_stage(self, host):
        state = _maint_checkpoint.HostState(self.args.state_dir, host)
        previous = state.load()
        if previous and previous.get("result") not in ("done", "skipped"):
            node = self.node_factory(host)
            state.validate_identity(self.checkpoint_identity(node))
            pending = previous.get("next_stage")
            if pending not in _maint_settings.STAGES:
                raise _maint_errors.StepFailed("invalid pending stage")
            self.guard_stage(pending, node, state)
            return pending
        if self.cloud.nova_service(host).get("Status") != "enabled":
            raise _maint_errors.StepFailed("disabled host has no resumable v1 checkpoint")
        return None

    def checkpoint_identity(self, node):
        self._aggregates = None
        service = self.cloud.nova_service(node.name)
        return {
            "profile_sha256": _maint_checkpoint.digest(self.profile),
            "cloud": self.args.os_cloud,
            "aggregate": self.aggregate_for(node.name),
            "members": sorted(self._aggregates[self.aggregate_for(node.name)]),
            "host": node.name, "address": node.address,
            "ssh_user": node.user, "ssh_key": node.key,
            "service_id": service.get("ID"),
            "settings_sha256": _maint_checkpoint.digest({k: getattr(self.args, k) for k in _maint_profile.DEFAULTS}),
            "health_environment": [self.args.prom_url, self.args.container_engine],
            "config_sha256": _maint_checkpoint.file_digest(self.args.wlb_config),
            "balancer_sha256": (_maint_checkpoint.file_digest(self.args.wbalancer)
                                if self.args.wbalancer else "installed-module"),
            "package_sha256": _maint_checkpoint.package_digest(),
        }

    def process_host(self, host, start_stage="preflight"):
        if not self._lock_held:
            raise _maint_errors.StepFailed("coordinator lock required")
        self._active_state = None
        node = self.node_factory(host)
        state = _maint_checkpoint.HostState(self.args.state_dir, host)
        previous = state.load()
        identity = self.checkpoint_identity(node)
        unfinished = previous and previous.get("result") not in ("done", "skipped")
        if previous and previous.get("schema_version") != _maint_checkpoint.SCHEMA_VERSION:
            raise _maint_checkpoint.StateError("legacy checkpoint cannot be adopted by v1")
        if unfinished:
            state.validate_identity(identity)
            if previous.get("runner") != socket.gethostname() or self.live_owner(host):
                raise _maint_errors.StepFailed("checkpoint belongs to another coordinator or live process")
            pending = previous.get("next_stage")
            if pending not in _maint_settings.STAGES or previous.get("completed") != _maint_settings.STAGES[:_maint_settings.STAGES.index(pending)]:
                raise _maint_errors.StepFailed("invalid checkpoint stage history")
            if start_stage != "preflight" and start_stage != pending:
                raise _maint_errors.StepFailed(f"resume must use pending stage '{pending}'")
            start_stage = pending
            # Validate BEFORE taking ownership or writing any state.
            self.guard_stage(start_stage, node, state)
            self.preflight(self.args.hosts,
                           recovering_host=host if start_stage in ("reboot", "health") else None)
        else:
            if start_stage != "preflight":
                raise _maint_errors.StepFailed("explicit resume requires an unfinished v1 checkpoint")
            if self.cloud.nova_service(host).get("Status") != "enabled":
                raise _maint_errors.StepFailed("fresh run requires enabled host; cannot adopt disabled service")
            self.preflight(self.args.hosts)
            state.data = {"schema_version": _maint_checkpoint.SCHEMA_VERSION, "identity": identity,
                          "run_id": uuid.uuid4().hex, "profile": self.profile["name"],
                          "started": _maint_remote.now_stamp(), "completed": ["preflight"],
                          "next_stage": "pre_drain"}
            start_stage = "pre_drain"
        self._active_state = state
        state.save(result="running", pid=os.getpid(), runner=socket.gethostname())
        timings = {}
        for stage in _maint_settings.STAGES[_maint_settings.STAGES.index(start_stage):]:
            if stage == "disable" or (stage == start_stage and _maint_settings.STAGES.index(stage) > _maint_settings.STAGES.index("disable")):
                self.confirm(host, stage)
            state.save(last_stage=stage, next_stage=stage)
            self.guard_stage(stage, node, state)
            started = time.monotonic()
            self.stage_fn(stage, node, state, start_stage)()
            timings[stage] = int(time.monotonic() - started)
            next_index = _maint_settings.STAGES.index(stage) + 1
            state.save(completed=_maint_settings.STAGES[:next_index],
                       next_stage=_maint_settings.STAGES[next_index] if next_index < len(_maint_settings.STAGES) else None,
                       result="done" if next_index == len(_maint_settings.STAGES) else "running")
        state.save(result="done", finished=_maint_remote.now_stamp(), timings=timings)
        return timings

    def log_dry_run_summary(self, results, error=None):
        """Print one compact, final report for a profile dry-run."""
        hooks = self.profile["hooks"]
        workflow = [
            "disable",
            "drain",
            "verify",
            "baseline",
            "maintenance({})".format(len(hooks["maintenance"])),
            "reboot({})".format(self.profile["reboot"]),
            "health",
            "post_boot({})".format(len(hooks["post_boot"])),
            "restore",
            "post_restore({})".format(len(hooks["post_restore"])),
        ]
        _maint_settings.log.info("================ DRY-RUN SUMMARY ================")
        _maint_settings.log.info("Profile   : %s%s", self.profile["name"],
                 " | " + self.profile["description"] if self.profile["description"] else "")
        _maint_settings.log.info("Checked   : preflight on all hosts; READY hosts also ran %d pre_drain hook(s), "
                 "drain plan and reboot decision", len(hooks["pre_drain"]))
        _maint_settings.log.info("No change : no host disabled, no VM migrated, no maintenance/reboot/restore executed")
        _maint_settings.log.info("Fresh run : %s", " -> ".join(workflow))

        for item in results:
            host, status = item["host"], item["status"]
            if status == "READY":
                summary = item.get("drain") or {}
                steps = summary.get("plan_steps") or []
                count = summary.get("drain_step_count")
                count = len(steps) if count is None else count
                left = summary.get("drain_unmovable_vm_ids") or []
                _maint_settings.log.info("  %-24s READY | drain=%d move(s), left=%d | reboot=%s (%s)",
                         host, count, len(left), item["reboot"], item["reboot_reason"])
                destinations = {}
                for step in steps:
                    dst = step.get("dst_host") or "unknown"
                    destinations[dst] = destinations.get(dst, 0) + 1
                if destinations:
                    _maint_settings.log.info("    destinations: %s", ", ".join(
                        f"{dst}={count}" for dst, count in sorted(destinations.items())))
                if left:
                    _maint_settings.log.info("    left behind: %s", ", ".join(left))
            elif status == "RESUME":
                _maint_settings.log.info("  %-24s RESUME | execute continues at stage '%s'", host, item["stage"])
            else:
                _maint_settings.log.info("  %-24s %s | %s", host, status, item.get("detail") or "")

        counts = {status: sum(1 for item in results if item["status"] == status)
                  for status in ("READY", "SKIPPED", "RESUME", "FAILED")}
        overall = f"FAILED: {error}" if error else "OK"
        _maint_settings.log.info("Overall   : %s | ready=%d skipped=%d resume=%d failed=%d",
                 overall, counts["READY"], counts["SKIPPED"], counts["RESUME"], counts["FAILED"])
        _maint_settings.log.info("=================================================")

    def dry_run(self):
        results = []
        try:
            self.preflight(self.args.hosts)
        except _maint_errors.StepFailed as e:
            self.log_dry_run_summary(results, str(e).splitlines()[0])
            raise
        for host in self.args.hosts:
            try:
                node = self.node_factory(host)
                node.boot_id()
                auto = self.auto_resume_stage(host)
                if auto:
                    _maint_settings.log.info("== %s: left disabled by an earlier run; a real run continues it at stage '%s'",
                             host, auto)
                    results.append({"host": host, "status": "RESUME", "stage": auto})
                    continue
                _maint_settings.log.info("== %s: pre_drain hooks (read-only check of the profile)", host)
                self.run_hooks("pre_drain", node)
                self.refuse_error_vms(host)
                self.refuse_locked_vms(host)
                drain_summary = self.drain_dry_run(node)
                needed, why = self.reboot_needed(node)
                _maint_settings.log.info("  %s: would reboot=%s (%s)", host, needed, why)
                results.append({
                    "host": host,
                    "status": "READY",
                    "drain": drain_summary,
                    "reboot": "yes" if needed else "no",
                    "reboot_reason": why,
                })
            except _maint_errors.SkipHost as e:
                _maint_settings.log.info("  SKIP %s: %s", host, e)
                results.append({"host": host, "status": "SKIPPED", "detail": str(e)})
            except _maint_errors.StepFailed as e:
                results.append({"host": host, "status": "FAILED", "detail": str(e).splitlines()[0]})
                self.log_dry_run_summary(results, str(e).splitlines()[0])
                raise
        self.log_dry_run_summary(results)
        _maint_settings.log.info("DRY-RUN OK (each host evaluated independently against the current state)")

    def run_all(self):
        with _locking.coordinator_lock(self.args.lock_file) as stream:
            self._coordinator_fd = stream.fileno()
            self._lock_held = True
            try:
                return self._run_all()
            finally:
                self._lock_held = False
                self._coordinator_fd = None

    def _run_all(self):
        done, skipped = [], []
        for i, host in enumerate(self.args.hosts, 1):
            self._active_state = None
            start = self.args.resume_from if (i == 1 and self.args.resume_from) else "preflight"
            _maint_settings.log.info("######## [%d/%d] %s  profile=%s  start=%s", i, len(self.args.hosts), host,
                     self.profile["name"], start)
            try:
                timings = self.process_host(host, start)
            except _maint_errors.OperatorStop:
                _maint_settings.log.info("stopped by operator before touching %s", host)
                declined = _maint_checkpoint.HostState(self.args.state_dir, host)
                declined.load()  # keep baseline/boot ids of a host being continued
                declined.save(result="declined")
                break
            except _maint_errors.SkipHost as e:
                _maint_settings.log.info("  SKIP %s: %s", host, e)
                skipped_state = _maint_checkpoint.HostState(self.args.state_dir, host)
                skipped_state.load()
                skipped_state.save(result="skipped", detail=str(e))
                skipped.append(host)
                self.notifier.send("HOST SKIPPED", f"{host} [{i}/{len(self.args.hosts)}]",
                                   [f"Host      : {host}", f"Reason    : {e}",
                                    f"Next      : {self.next_host(i)}"])
                continue
            except BaseException as e:
                state = self._active_state
                if state is not None:
                    if state.data.get("next_stage") == "post_restore" and state.data.get("restore_intent"):
                        # Acceptance gate failed after enable. Quarantine only;
                        # do not redrain/reboot workloads scheduled in that gap.
                        try:
                            service = self.cloud.nova_service(host)
                            if service.get("Status") == "enabled":
                                reason = "rolling-maint v1:" + state.data["run_id"]
                                state.save(quarantine_intent=True)
                                self.cloud.set_nova_enabled(host, False, reason)
                            self.require_owned(self.node_factory(host), state)
                            state.save(quarantined=True, next_stage="health",
                                completed=_maint_settings.STAGES[:_maint_settings.STAGES.index("health")])
                        except Exception as quarantine_error:  # noqa: BLE001 - quarantine bookkeeping must not hide the original stop reason
                            state.save(quarantine_error=str(quarantine_error))
                    state.save(result="stopped", detail=str(e))
                e.host = host
                e.stage = state.data.get("next_stage") if state else "validation"
                try:
                    observed = self.cloud.nova_service(host).get("Status")
                    e.observed_status = observed if observed in ("enabled", "disabled") else "UNKNOWN"
                except Exception:  # noqa: BLE001 - an unreadable service status is reported as UNKNOWN
                    e.observed_status = "UNKNOWN"
                e.left_disabled = e.observed_status == "disabled"
                e.remaining = self.args.hosts[i - 1:]
                raise
            timing_text = ", ".join(f"{k}={v}s" for k, v in timings.items())
            _maint_settings.log.info("  %s DONE  timings: %s", host, timing_text)
            done.append(host)
            self.notifier.send("HOST DONE", f"{host} [{i}/{len(self.args.hosts)}] in {sum(timings.values())}s",
                [f"Host      : {host}", f"Timings   : {timing_text}",
                 f"Next      : {self.next_host(i)}"])
        return done, skipped

    def confirm(self, host, stage):
        if self.args.yes:
            return
        what = "disable, drain and maintain" if stage == "disable" else f"continue at stage '{stage}' on"
        try:
            answer = input("{} needs '{}': {} {} now? [y/N] ".format(
                host, self.profile["name"], what, host)).strip().lower()
        except EOFError:  # no terminal (cron, nohup): never assume yes
            answer = ""
        if answer != "y":
            raise _maint_errors.OperatorStop()

    def next_host(self, i):
        """Name of the host after position i (1-based), for alert text."""
        return self.args.hosts[i] if i < len(self.args.hosts) else "none (last host)"
