"""Structured, credential-safe logging for POLARIS."""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
from pathlib import Path

_CONFIGURED = False

# Patterns that must never reach the log stream.
_SECRET_PATTERNS = [
    re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*\S+"),
    re.compile(r"(?i)(api[_-]?key|apikey|token|secret)\s*[=:]\s*\S+"),
    re.compile(r"(?i)://([^:/@\s]+):([^@/\s]+)@"),  # user:pass@host in URLs
]


class SecretRedactingFilter(logging.Filter):
    """Redact anything that looks like a credential from log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        redacted = msg
        for pattern in _SECRET_PATTERNS:
            if pattern.groups >= 2 and "://" in pattern.pattern:
                redacted = pattern.sub(r"://\1:***@", redacted)
            else:
                redacted = pattern.sub(lambda m: m.group(0).split("=")[0].split(":")[0] + "=***", redacted)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(SecretRedactingFilter())
    root.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        fh.addFilter(SecretRedactingFilter())
        root.addHandler(fh)

    # Third-party noise control
    for noisy in ("urllib3", "botocore", "matplotlib", "asyncio", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    if not _CONFIGURED:
        try:
            from app.config import get_settings

            s = get_settings()
            setup_logging(s.log_level, s.log_file)
        except Exception:  # pragma: no cover - config may not be importable yet
            setup_logging("INFO", None)
    return logging.getLogger(f"polaris.{name}" if not name.startswith("polaris") else name)
