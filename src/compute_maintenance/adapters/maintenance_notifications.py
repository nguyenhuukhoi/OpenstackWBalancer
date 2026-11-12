"""maint.notify — extracted domain responsibilities."""
import os
import shlex
import socket
import sys
from email.mime.text import MIMEText

import compute_maintenance.adapters.commands as _maint_remote
import compute_maintenance.profiles.defaults as _maint_settings
import compute_maintenance.profiles.schema as _maint_profile
from compute_maintenance.adapters.mail import send_message

NOTIFY_EVENTS = {
    "off": set(),
    "fail": {"STOPPED"},
    "summary": {"STOPPED", "FINISHED"},
    "all": {"STOPPED", "FINISHED", "STARTED", "HOST DONE", "HOST SKIPPED"},
}


SMTP_KEYS = ("ALERT_EMAIL_TO", "EMAIL_TO", "ALERT_EMAIL_FROM", "EMAIL_FROM", "SMTP_SERVER",
             "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_STARTTLS")


class Notifier:
    """Mail rolling-maint events. Never raises: a broken mail setup must not
    stop or fail a maintenance run, it is only logged."""

    def __init__(self, wlb_config, level, run_label, log_path, cmdline=None):
        self.cmdline = " ".join(shlex.quote(t) for t in (cmdline or sys.argv))
        file_values = _maint_profile.read_env_file(wlb_config) if wlb_config else {}

        def get(key):  # environment wins, like wbalancer's load_dotenv()
            return os.environ.get(key) or file_values.get(key) or ""

        self.events = NOTIFY_EVENTS[level]
        self.to = [x.strip() for x in (get("ALERT_EMAIL_TO") or get("EMAIL_TO")).split(",") if x.strip()]
        self.sender = get("ALERT_EMAIL_FROM") or get("EMAIL_FROM") or "rolling-maint@" + socket.gethostname()
        self.server = get("SMTP_SERVER")
        self.port = int(get("SMTP_PORT") or 25)
        self.user = get("SMTP_USER")
        self.password = get("SMTP_PASSWORD")
        self.starttls = get("SMTP_STARTTLS").strip().lower() in ("1", "true", "yes", "y", "on")
        self.run_label = run_label
        self.log_path = log_path
        self.enabled = bool(self.events and self.to and self.server)

    def describe(self):
        if not self.events:
            return "alerts: off (--notify off)"
        if not self.enabled:
            return "alerts: NOT configured (need ALERT_EMAIL_TO + SMTP_SERVER in --wlb-config or env)"
        return "alerts: {} -> {} via {}:{}".format(
            "/".join(sorted(self.events)), ", ".join(self.to), self.server, self.port)

    def send(self, event, summary, lines=()):
        if not self.enabled or event not in self.events:
            return False
        subject = f"[OpenStack][rolling-maint][{event}] {summary}"
        body = "\n".join([
            "OpenStack rolling maintenance alert",
            "",
            f"Event     : {event}",
            f"Summary   : {summary}",
            f"Run       : {self.run_label}",
            f"Runner    : {socket.gethostname()} ({self.cmdline})",
            f"Log file  : {self.log_path}",
            f"Alert Time: {_maint_remote.now_stamp()}",
            "",
            *lines,
        ])
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.to)
        try:
            send_message(self.server, self.port, self.user, self.password, self.starttls, self.sender, self.to, msg)
        except Exception as e:  # noqa: BLE001 - a mail failure must never stop maintenance (logged)
            _maint_settings.log.warning("alerts: failed to send '%s' mail: %s", event, e)
            return False
        _maint_settings.log.info("alerts: sent '%s' mail to %s", event, ", ".join(self.to))
        return True


class _NullNotifier:
    def send(self, *a, **kw):
        return False
