"""Shared SMTP transport; callers choose event policy and presentation."""
import smtplib
import ssl


def send_message(server, port, user, password, starttls, sender, recipients, message):
    smtp = smtplib.SMTP(server, port, timeout=30)
    try:
        smtp.ehlo()
        if starttls:
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
        if user:
            smtp.login(user, password or "")
        smtp.sendmail(sender, recipients, message.as_string())
    finally:
        try:
            smtp.quit()
        except Exception:  # noqa: BLE001, S110 - quit() after a sent or failed message is best effort
            pass
