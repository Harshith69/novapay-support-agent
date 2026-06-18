"""Logging helpers.

Two concerns:
1. ``get_logger`` — a consistent console + rotating-file logger for humans.
2. ``append_jsonl`` / ``log_event`` — structured, machine-readable event logs
   (fallbacks, hallucinations, errors, interactions) that the dashboard and
   evaluation scripts read back.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from common.config import LOGS_DIR

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console + ``logs/<name>.log``.

    Idempotent: repeated calls return the same handler-free-of-duplicates
    logger, which matters because modules are imported many times.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOGS_DIR / f"{name}.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console)
    _LOGGERS[name] = logger
    return logger


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record per line, creating parent dirs as needed.

    Never raises on a missing directory — structured logs must not crash the
    request path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_event(filename: str, **fields: Any) -> None:
    """Append a timestamped structured event to ``logs/<filename>``."""
    record = {"timestamp": utc_now_iso(), **fields}
    append_jsonl(LOGS_DIR / filename, record)
