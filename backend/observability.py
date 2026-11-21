"""Logging (plain or JSON lines) and optional Sentry error reporting."""

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"level": record.levelname, "logger": record.name, "msg": record.getMessage(),
                "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S")}
        if getattr(record, "request_id", None):
            data["request_id"] = record.request_id
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def configure(settings) -> None:
    root = logging.getLogger("citypulse")
    if getattr(root, "_cp_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else
                         logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root._cp_configured = True  # type: ignore[attr-defined]

    if settings.sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.app_env, traces_sample_rate=0.05,
                        send_default_pii=False)
