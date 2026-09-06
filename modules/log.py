"""
Shared logging helpers.

A single module-level logger writes to both stdout and a daily file under
logs/. Modules call log_info / log_warning / log_error rather than configuring
their own handlers.

Adapted from the sibling KLSE_Monitor project's modules/log.py -- same shape,
same Windows UTF-8 workaround, different logger name so the two engines can run
in one process (e.g. a combined export script) without stealing each other's
handlers.
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from config import settings

_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("macro_engine")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")

    # Windows consoles default to cp1252; force UTF-8 so em-dashes/arrows in log
    # messages don't turn into mojibake -- or worse, raise UnicodeEncodeError
    # mid-run and abort an otherwise healthy ingestion.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    try:
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        fileh = logging.FileHandler(
            settings.LOG_DIR / f"run_{date.today().isoformat()}.log",
            encoding="utf-8",
        )
        fileh.setFormatter(fmt)
        logger.addHandler(fileh)
    except OSError:
        # File logging is best-effort; stdout always works.
        pass

    _logger = logger
    return logger


def log_info(msg: str) -> None:
    get_logger().info(msg)


def log_warning(msg: str) -> None:
    get_logger().warning(msg)


def log_error(msg: str) -> None:
    get_logger().error(msg)
