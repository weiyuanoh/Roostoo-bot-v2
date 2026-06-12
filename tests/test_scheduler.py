from datetime import datetime, timezone

from bot.scheduler import next_hour_boundary


def test_next_hour_boundary_uses_next_utc_hour_plus_delay():
    now = datetime(2026, 6, 10, 12, 34, 56, tzinfo=timezone.utc)

    target = datetime.fromtimestamp(next_hour_boundary(delay_seconds=5, now=now), tz=timezone.utc)

    assert target == datetime(2026, 6, 10, 13, 0, 5, tzinfo=timezone.utc)


def test_next_hour_boundary_treats_naive_datetime_as_utc():
    now = datetime(2026, 6, 10, 12, 0, 0)

    target = datetime.fromtimestamp(next_hour_boundary(delay_seconds=10, now=now), tz=timezone.utc)

    assert target == datetime(2026, 6, 10, 13, 0, 10, tzinfo=timezone.utc)
