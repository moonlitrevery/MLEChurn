"""Application-wide logging setup (console, env-driven level)."""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DEFAULT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"

_LOG_LEVEL_ENV = "CHURN_LOG_LEVEL"
_PACKAGE_LOGGER = "mlechurn"


def setup_logging(
    level: int | str | None = None,
    *,
    force: bool = False,
) -> None:
    """
    Configure the root logger once (or replace handlers if ``force=True``).

    Level: argument, else ``CHURN_LOG_LEVEL`` (default ``INFO``).
    """
    if level is None:
        raw = os.environ.get(_LOG_LEVEL_ENV, "INFO")
        level = getattr(logging, str(raw).upper(), logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    if root.handlers and not force:
        root.setLevel(level)
        return

    if force and root.handlers:
        for h in root.handlers[:]:
            root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str | None = None) -> logging.Logger:
    """Child logger under the ``mlechurn`` namespace."""
    suffix = name.strip(".") if name else ""
    full = f"{_PACKAGE_LOGGER}.{suffix}" if suffix else _PACKAGE_LOGGER
    return logging.getLogger(full)
