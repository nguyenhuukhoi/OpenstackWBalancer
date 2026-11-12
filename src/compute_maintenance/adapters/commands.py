"""maint.remote — extracted domain responsibilities."""
import shlex
import subprocess
import time
from datetime import datetime

import compute_maintenance.domain.maintenance_errors as _maint_errors
import compute_maintenance.profiles.defaults as _maint_settings

DYNAMIC_CONTAINER_PREFIXES = ("neutron_ovn_metadata_agent_haproxy_",)


def run(cmd, timeout=120, check=True, input_text=None, env=None):
    _maint_settings.log.debug("$ %s", " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=input_text, env=env,
                       check=False)   # the return code is checked below
    if check and p.returncode != 0:
        raise _maint_errors.StepFailed(
            "command failed rc={}: {}\n{}".format(
                p.returncode, " ".join(cmd), (p.stderr or p.stdout).strip()
            )
        )
    return p


def wait_until(desc, fn, timeout, interval=10):
    """Poll fn() -> (ok, detail) until ok or timeout. Returns last detail."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            ok, detail = fn()
        except (_maint_errors.StepFailed, subprocess.TimeoutExpired, OSError) as e:
            ok, detail = False, str(e)
        if ok:
            return detail
        if time.monotonic() >= deadline:
            raise _maint_errors.StepFailed(f"timeout after {timeout}s waiting for {desc}: {detail}")
        _maint_settings.log.info("  waiting for %s: %s", desc, detail)
        time.sleep(interval)


def is_dynamic_container(name):
    return name.startswith(DYNAMIC_CONTAINER_PREFIXES)


def now_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Node:
    def __init__(self, name, address, user, key, engine):
        self.name = name
        self.address = address
        self.user = user
        self.key = key
        self.engine = engine

    def ssh(self, command, timeout=30, check=True):
        # Host-key policy (StrictHostKeyChecking, UserKnownHostsFile, ...) is
        # deliberately NOT set here: the coordinator's OpenSSH configuration
        # (~/.ssh/config, /etc/ssh/ssh_config) decides, like for any other ssh
        # the operator runs. With BatchMode=yes an unknown or changed key fails
        # the command (exit 255) unless that configuration says otherwise.
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
        ]
        if self.key:  # else ssh's own defaults: ~/.ssh/id_*, agent, ~/.ssh/config
            cmd += ["-i", self.key]
        cmd += [f"{self.user}@{self.address}", command]
        return run(cmd, timeout=timeout, check=check)

    def boot_id(self):
        import uuid
        value = self.ssh("cat /proc/sys/kernel/random/boot_id").stdout.strip()
        try:
            uuid.UUID(value)
        except (ValueError, AttributeError) as exc:
            raise _maint_errors.StepFailed("invalid/empty boot ID") from exc
        return value

    def failed_units(self):
        out = self.ssh("systemctl --failed --no-legend --plain").stdout
        return {line.split()[0] for line in out.splitlines() if line.strip()}

    def containers(self):
        out = self.ssh(f"{self.engine} ps -a --format '{{{{.Names}}}}\\t{{{{.Status}}}}'").stdout
        result = {}
        for line in out.splitlines():
            if "\t" in line:
                name, status = line.split("\t", 1)
                result[name] = status
        return result

    def running_guest_processes(self):
        """Number of QEMU guest processes on the host (Kolla runs them in the
        host PID namespace). Nova inventory is not proof that the hypervisor
        is empty; this is checked independently right before a reboot."""
        out = self.ssh("ps -eo comm= | grep -cE '^qemu-(system|kvm)' || true").stdout.strip()
        try:
            return int(out.splitlines()[-1]) if out else 0
        except ValueError as exc:
            raise _maint_errors.StepFailed(f"cannot count QEMU processes: {out!r}") from exc

    def reboot(self):
        # A systemd-owned job survives loss of this SSH session. Unknown submit
        # outcomes are reconciled by boot ID, never by submitting a second reboot.
        self.ssh("systemd-run --on-active=2s --unit=compute-maintenance-reboot systemctl reboot")
