"""maint.report — extracted domain responsibilities."""
import json
import logging
import os
import re
import sys
from datetime import datetime

import compute_maintenance.profiles.defaults as _maint_settings

MIGRATION_OK_RESULTS = {"success", "migration_success"}


UNMOVABLE_HELP = {
    "not_active": "VM is {detail}, not ACTIVE: live migration only moves running VMs. Either start it "
                  "(openstack server start {vm}) so the drain can live-migrate it, or move it while off "
                  "(openstack server migrate {vm}; then openstack server migration confirm {vm}); "
                  "then re-run",
    "incomplete_metrics": "Prometheus has no CPU/RAM sample for this VM (just created? libvirt exporter "
                          "down?). Wait a few minutes or fix the exporter, then re-run",
    "incomplete_metadata": "its flavor/metadata could not be read from Nova. Check "
                           "'openstack server show {vm}', then re-run",
    # wbalancer builds VM details from Prometheus samples, so a VM with no
    # libvirt metrics (SHUTOFF/PAUSED/SUSPENDED, exporter gap) lands here.
    "locked": "the VM is locked, and Nova refuses to live-migrate a locked VM. Ask the owner to unlock "
              "it, or as admin: openstack server unlock {vm}; then re-run",
    "task_in_progress": "Nova is still running a task on this VM (snapshot, rebuild, resize, ...). "
                        "Wait for it to finish (openstack server show {vm} -c OS-EXT-STS:task_state), "
                        "then re-run",
    "unsupported_resources": "its flavor needs resources the planner does not model (pinned CPU / PCPU, "
                             "PCI or vGPU devices, resources:* extra specs). Move it by hand with "
                             "openstack server migrate --live-migration {vm}, then re-run",
    "missing_vm_details": "Prometheus has no usable CPU/RAM sample for this VM, so wbalancer cannot plan "
                          "it (libvirt exporter gap, VM created seconds ago, or a guest without balloon "
                          "memory stats with a wbalancer older than the RSS fallback). Check "
                          "libvirt_domain_* for the VM, then re-run",
}


_UUID_AT_END = re.compile(r"\(([0-9a-fA-F-]{36})\)$")


def explain_unmovable(reason, vm_id, nova_status=None):
    code, _, detail = reason.partition(":")
    if nova_status and nova_status != "ACTIVE":
        # The real cause whatever wbalancer's code says (a SHUTOFF VM has no
        # metrics and shows up as missing_vm_details).
        code, detail = "not_active", nova_status
    text = UNMOVABLE_HELP.get(code)
    if not text:
        return f"wbalancer cannot live-migrate it ({reason}); handle it by hand, then re-run"
    return text.format(detail=detail or "not ACTIVE", vm=vm_id)


def parse_drain_output(lines):
    """Pull per-VM reasons out of wbalancer's drain report, which the events
    file does not carry:  '   name (uuid) reason=not_active:SHUTOFF'  and the
    'NO DESTINATION' block ('   name (uuid) (2 vCPU / 4.0 GiB)' followed by
    '      host -> why' lines)."""
    out = {"unmovable": {}, "blocked": {}, "names": {}}
    in_blocked, current = False, None
    for raw in lines:
        line = raw.rstrip()
        if line.lstrip().startswith("NO DESTINATION for"):
            in_blocked, current = True, None
            continue
        if not line.startswith("   "):
            in_blocked = False if line.strip() else in_blocked
            continue
        body = line.strip()
        m = re.match(r"^(.*) reason=(\S+)$", body)
        if m:
            label, reason = m.groups()
            u = _UUID_AT_END.search(label)
            vm = u.group(1) if u else label
            out["unmovable"][vm] = reason
            if u:
                out["names"][vm] = label[:u.start()].strip()
            continue
        if in_blocked:
            m = re.match(r"^(.*) \(\d+ vCPU / [\d.]+ GiB\)$", body)
            if m and not line.startswith("      "):
                u = _UUID_AT_END.search(m.group(1))
                current = u.group(1) if u else m.group(1)
                if u:
                    out["names"][current] = m.group(1)[:u.start()].strip()
                out["blocked"].setdefault(current, [])
            elif current and "->" in body:
                out["blocked"][current].append(body)
    return out


def read_drain_events(path, host, since, max_bytes=4 * 1024 * 1024):
    """From wbalancer's JSONL events file, return (failed VM moves off
    ``host``, last drain summary event for ``host``) recorded at/after
    ``since`` (epoch). Only the tail of the file is read."""
    failed, summary = [], None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError as e:
        _maint_settings.log.warning("cannot read wbalancer events %s: %s", path, e)
        return failed, summary
    lines = chunk.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line is probably cut in half
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict) or (ev.get("event_ts") or 0) < since - 5:
            continue
        # Drain summary events (drain_dry_run/noop/completed/infeasible) also
        # carry vm_id/src_host of a plan step: test drain_host first.
        if ev.get("drain_host") == host:
            summary = ev
        elif ev.get("vm_id") and ev.get("src_host") == host and ev.get("result") not in MIGRATION_OK_RESULTS:
            failed.append(ev)
    return failed, summary


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "rolling-maint-{}.log".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    _maint_settings.log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    _maint_settings.log.addHandler(fh)
    _maint_settings.log.addHandler(sh)
    return path


def tail_file(path, n):
    try:
        with open(path, errors="replace") as f:
            return ["  " + line.rstrip() for line in f.readlines()[-n:]]
    except OSError:
        return []
