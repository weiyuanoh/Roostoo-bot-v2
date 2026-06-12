"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone

from bot.config import LOG_CONSOLE, LOG_DIR, LOG_LEVEL

LOG_DIR.mkdir(parents=True, exist_ok=True)


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, sort_keys=True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    if LOG_CONSOLE:
        console = logging.StreamHandler()
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(console)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / "bot.jsonl",
        when="midnight",
        backupCount=30,
        utc=True,
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    return logger


def log_jsonl(filename: str, payload: dict) -> None:
    """Append a JSON record in the configured log directory."""
    record = {"logged_at": datetime.now(timezone.utc).isoformat(), **payload}
    with (LOG_DIR / filename).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def log_trade(trade_data: dict) -> None:
    log_jsonl("trades.jsonl", trade_data)

