"""Logging setup: plain text (default) or JSON lines on stderr (``CHURN_LOG_FORMAT=json``)."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Final

_LOG_LEVEL_ENV = "CHURN_LOG_LEVEL"
_LOG_FORMAT_ENV = "CHURN_LOG_FORMAT"
_PACKAGE_LOGGER = "mlechurn"

_TEXT_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_TEXT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line; merges ``record.structured`` from :func:`log_event`."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            for k, v in structured.items():
                if k not in payload:
                    payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextStructuredFormatter(logging.Formatter):
    """Plain text; appends JSON blob when ``log_event`` set ``record.structured``."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict) and len(structured) > 1:
            detail = {k: v for k, v in structured.items() if k != "event"}
            if detail:
                line = f"{line} | {json.dumps(detail, default=str)}"
        return line


def setup_logging(
    level: int | str | None = None,
    *,
    force: bool = False,
    use_json: bool | None = None,
) -> None:
    """
    Configure root logging once.

    - Level: argument, else ``CHURN_LOG_LEVEL`` (default ``INFO``).
    - Format: ``CHURN_LOG_FORMAT=json`` for JSON lines; anything else = plain text.
    """
    if level is None:
        raw = os.environ.get(_LOG_LEVEL_ENV, "INFO")
        level = getattr(logging, str(raw).upper(), logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if use_json is None:
        use_json = os.environ.get(_LOG_FORMAT_ENV, "text").lower() == "json"

    root = logging.getLogger()
    if root.handlers and not force:
        root.setLevel(level)
        return

    if force and root.handlers:
        for h in root.handlers[:]:
            root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    if use_json:
        handler.setFormatter(JsonLineFormatter())
    else:
        handler.setFormatter(TextStructuredFormatter(_TEXT_FORMAT, datefmt=_TEXT_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Log a named event; fields are attached as ``record.structured`` for JSON/text formatters."""
    logger.log(level, event, extra={"structured": {"event": event, **fields}})


def get_logger(name: str | None = None) -> logging.Logger:
    suffix = name.strip(".") if name else ""
    full = f"{_PACKAGE_LOGGER}.{suffix}" if suffix else _PACKAGE_LOGGER
    return logging.getLogger(full)
