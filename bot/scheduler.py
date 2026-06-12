"""Scheduling helpers for hour-boundary live trading."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone


def next_hour_boundary(delay_seconds: int = 5, now: datetime | None = None) -> float:
    """Return the next UTC hour boundary plus a small data-settlement delay."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    next_hour = current.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return (next_hour + timedelta(seconds=delay_seconds)).timestamp()


def sleep_until(target_ts: float, keep_running: Callable[[], bool] | None = None) -> bool:
    """Sleep until target timestamp; return False if interrupted by keep_running."""
    running = keep_running or (lambda: True)
    while running():
        remaining = target_ts - time.time()
        if remaining <= 0:
            return True
        time.sleep(min(1.0, remaining))
    return False
