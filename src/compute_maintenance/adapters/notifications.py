"""notify — extracted domain responsibilities."""
import logging
from datetime import datetime
from email.mime.text import MIMEText

import compute_maintenance.presentation.reports as _report
from compute_maintenance.adapters.mail import send_message


def send_migration_alert_email(
    cfg,
    aggregate,
    vm_id,
    vm_name,
    src_host,
    dst_host,
    result,
    detail=None,
    duration_seconds=None,
    started_at=None,
    ended_at=None,
    project_name=None,
    project_id=None,
):
    if not cfg.alert_email_to or not cfg.smtp_server:
        logging.debug("Migration alert email not configured; skipping email notification.")
        return

    recipients = [x.strip() for x in cfg.alert_email_to.split(",") if x.strip()]
    if not recipients:
        logging.debug("Migration alert email recipient list is empty; skipping email notification.")
        return

    sender = cfg.alert_email_from or "loadleveller@localhost"
    subject = f"[OpenStack][{result}] Migration alert for {vm_name}"
    body = "\n".join(
        [
            "OpenStack workload balancer migration alert",
            "",
            f"Result    : {result}",
            f"Aggregate : {aggregate}",
            f"VM        : {vm_name}",
            f"VM ID     : {vm_id}",
            f"Project   : {project_name or project_id or 'n/a'}",
            f"Project ID: {project_id or 'n/a'}",
            f"Source    : {src_host}",
            f"Target    : {dst_host}",
            f"Route     : {src_host} -> {dst_host}",
            f"Started   : {_report.format_migration_timestamp(started_at)}",
            f"Ended     : {_report.format_migration_timestamp(ended_at)}",
            f"Duration  : {_report.format_migration_duration(duration_seconds)}",
            f"Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Detail    : {detail or 'n/a'}",
        ]
    )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    try:
        send_message(cfg.smtp_server, cfg.smtp_port, cfg.smtp_user, cfg.smtp_password, cfg.smtp_starttls, sender, recipients, msg)

        logging.info("Sent migration alert email for VM %s (%s) to %s.", vm_name, vm_id, cfg.alert_email_to)
    except Exception as e:  # noqa: BLE001 - a mail failure must never stop a balance run (logged)
        logging.warning("Failed to send migration alert email for VM %s (%s): %s", vm_name, vm_id, e)
