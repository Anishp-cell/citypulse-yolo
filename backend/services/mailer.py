"""Outgoing email. Only used for password resets; complaints are sent by citizens themselves."""

import logging
import smtplib
from email.message import EmailMessage

from backend.settings import Settings

log = logging.getLogger("citypulse.mail")


class ConsoleMailer:
    """Dev/test mailer: keeps messages in memory and logs them."""

    def __init__(self):
        self.outbox: list[dict] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.outbox.append({"to": to, "subject": subject, "body": body})
        log.info("email to %s: %s\n%s", to, subject, body)


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
    return SmtpMailer(settings) if settings.smtp_host else ConsoleMailer()
