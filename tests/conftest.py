"""
Shared fixtures.

Everything here is synthetic and offline. No test in this suite touches the
network or the production database: the parsers are pure by construction
(modules/sources.py keeps every fetch in a one-line function), so the parsing,
transformation, factor, model and vintage logic can all be exercised against
data built in memory. That is what makes the suite fast enough to run on every
change rather than occasionally.
"""
from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import db  # noqa: E402


@pytest.fixture
def conn():
    c = db.get_db(":memory:")
    yield c
    c.close()


def month_dates(n: int, end: date | None = None, day: int = 1) -> list[str]:
    """n consecutive month-start dates ending at `end`'s month."""
    end = end or date.today()
    out = []
    y, m = end.year, end.month
    for _ in range(n):
        out.append(date(y, m, day).isoformat())
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return list(reversed(out))


def daily_dates(n: int, end: date | None = None) -> list[str]:
    """n consecutive weekdays ending at `end`."""
    end = end or date.today()
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= timedelta(days=1)
    return list(reversed(out))


@pytest.fixture
def monthly_series():
    """120 months of a smooth trend with a dip -- enough for MIN_PERIODS."""
    dates = month_dates(120)
    return [(d, 100.0 + i * 0.5 + math.sin(i / 6.0) * 3.0)
            for i, d in enumerate(dates)]


@pytest.fixture
def daily_series():
    dates = daily_dates(800)
    return [(d, 3.0 + math.sin(i / 40.0) * 0.8) for i, d in enumerate(dates)]


@pytest.fixture
def seeded_conn(conn, monthly_series):
    """A database with a few series populated, for integration-flavoured tests."""
    from config import series as cat
    from modules import ingestion
    for sid in ("PAYEMS", "UNRATE", "INDPRO", "CPIAUCSL"):
        ingestion.upsert_observations(conn, cat.BY_ID[sid], monthly_series)
    return conn
