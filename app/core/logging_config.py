"""
Structured logging.

We log task ids, tool names, durations and outcomes - never API keys,
raw user content, or internal prompts. See README "Security & Privacy".
"""
from __future__ import annotations

import logging
import sys

from app.core.config import settings

_REDACT_KEYS = ("api_key", "authorization", "token", "secret")


class RedactFilter(logging.Filter):
    """Best-effort filter that blanks out anything that looks like a secret."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = str(record.getMessage())
        lowered = msg.lower()
        if any(key in lowered for key in _REDACT_KEYS):
            record.msg = "[redacted log line - contained a sensitive key name]"
            record.args = ()
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RedactFilter())

    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
