"""Outgoing email. Only used for password resets; complaints are sent by citizens themselves."""

import logging
import smtplib
from email.message import EmailMessage

from backend.settings import Settings

log = logging.getLogger("citypulse.mail")
OUTBOX_MAX = 100


class ConsoleMailer:
    """Dev/test mailer: keeps recent messages in memory. Logs bodies only when `log_body` (never in
    production, because a body can contain a password-reset token)."""

    def __init__(self, log_body: bool = True):
        self.outbox: list[dict] = []
        self.log_body = log_body

    def send(self, to: str, subject: str, body: str) -> None:
        self.outbox.append({"to": to, "subject": subject, "body": body})
        del self.outbox[:-OUTBOX_MAX]
        if self.log_body:
            log.info("email to %s: %s\n%s", to, subject, body)
        else:
            log.info("email to %s suppressed (no SMTP configured): %s", to, subject)


class SmtpMailer:
    def __init__(self, settings: Settings):
        self.s = settings

    def send(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"] = self.s.smtp_from
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        try:
            with smtplib.SMTP(self.s.smtp_host, self.s.smtp_port, timeout=15) as smtp:
                smtp.starttls()
                if self.s.smtp_user:
                    smtp.login(self.s.smtp_user, self.s.smtp_password)
                smtp.send_message(msg)
        except Exception:
            # Never leak delivery failures to the caller (forgot-password must not reveal anything).
            log.exception("failed to send email to %s", to)


def build_mailer(settings: Settings):
    if settings.smtp_host:
        return SmtpMailer(settings)
    return ConsoleMailer(log_body=settings.app_env != "production")
