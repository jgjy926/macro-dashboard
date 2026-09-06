"""
Data-layer QA (spec 23): schema, API failures, missing/duplicate/stale
observations, frequency conversion, date alignment, release-date alignment.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from config import series as cat
from modules import db, ingestion, quality, sources
from tests.conftest import daily_dates, month_dates


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
def test_schema_is_idempotent(conn):
    """init() runs twice without error -- every daily run calls it."""
    db.init(conn)
    db.init(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for required in ("observations", "vintages", "forecasts", "model_runs",
                     "backtest_results", "backtest_audit", "alerts", "api_errors"):
        assert required in tables


def test_catalogue_syncs_to_metadata(conn):
    n = conn.execute("SELECT COUNT(*) FROM series_metadata").fetchone()[0]
    assert n == len(cat.ALL_SERIES)


def test_series_ids_are_unique():
    assert len(cat.BY_ID) == len(cat.ALL_SERIES)


def test_every_series_has_a_valid_category_and_direction():
    valid = set(cat.CATEGORIES) | {"truth", "market"}
    for s in cat.ALL_SERIES:
        assert s.category in valid, s.sid
        assert s.direction in (-1, 0, 1), s.sid
        assert s.freq in ("daily", "weekly", "monthly", "quarterly"), s.sid
        assert s.vintage in ("alfred", "final", "lag"), s.sid
        assert s.role in ("LEADING", "COINCIDENT", "LAGGING", "CONTEXT"), s.sid


def test_inflation_series_carry_no_sign():
    """Spec 8.7 / config.series INFLATION_NOTE: inflation is regime-handled, so
    no inflation series may carry a good/bad direction."""
    for s in cat.by_category("inflation"):
        assert s.direction == 0, f"{s.sid} must be direction=0"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parse_fred_csv_handles_missing_and_crlf():
    text = "observation_date,DGS10\r\n2020-01-01,1.88\r\n2020-01-02,.\r\n2020-01-03,1.90\r\n"
    rows = sources.parse_fred_single(text)
    assert rows == [("2020-01-01", 1.88), ("2020-01-03", 1.90)]


def test_parse_fred_csv_multi_series():
    text = ("observation_date,A,B\n2020-01-01,1,2\n2020-01-02,.,3\n")
    cols = sources.parse_fred_csv(text)
    assert cols["A"] == [("2020-01-01", 1.0)]
    assert cols["B"] == [("2020-01-01", 2.0), ("2020-01-02", 3.0)]


def test_parse_fred_csv_empty_input():
    assert sources.parse_fred_csv("") == {}
    assert sources.parse_fred_single("observation_date,X\n") == []


def test_parse_bls_drops_annual_averages():
    payload = {"Results": {"series": [{"seriesID": "X", "data": [
        {"year": "2024", "period": "M13", "value": "9.9"},   # annual average
        {"year": "2024", "period": "M01", "value": "1.0"},
        {"year": "2024", "period": "M02", "value": "2.0"},
    ]}]}}
    rows = sources.parse_bls(payload)["X"]
    assert rows == [("2024-01-01", 1.0), ("2024-02-01", 2.0)]


def test_parse_yahoo_skips_nulls():
    payload = ('{"chart":{"result":[{"timestamp":[1704067200,1704153600],'
               '"indicators":{"quote":[{"close":[100.5,null]}]}}]}}')
    assert sources.parse_yahoo_chart(payload) == [("2024-01-01", 100.5)]


# ---------------------------------------------------------------------------
# API failures (spec 23: "API failures")
# ---------------------------------------------------------------------------
def test_retry_gives_up_and_raises_fetch_error(conn):
    calls = []

    def always_fails():
        calls.append(1)
        raise TimeoutError("boom")

    with pytest.raises(sources.FetchError):
        sources.retry_with_backoff(always_fails, what="test", conn=conn,
                                   source="TEST", retries=2)
    assert len(calls) == 2
    logged = conn.execute("SELECT COUNT(*) FROM api_errors").fetchone()[0]
    assert logged == 2, "every failed attempt must be recorded, not swallowed"


def test_404_is_not_retried(conn):
    """A 404 means a wrong or retired id -- retrying wastes time on a URL that
    will never work."""
    import urllib.error
    calls = []

    def not_found():
        calls.append(1)
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    with pytest.raises(sources.FetchError):
        sources.retry_with_backoff(not_found, what="t", conn=conn, retries=3)
    assert len(calls) == 1


def test_one_dead_series_does_not_abort_the_run(conn, monkeypatch):
    """Spec 29: never fail silently -- but also never let one failure abort
    every other series."""
    def fake_batch(ids, start=None, conn=None):
        raise sources.FetchError("simulated outage")

    monkeypatch.setattr(sources, "fetch_fred_csv", fake_batch)
    report = ingestion.update(conn, only=["PAYEMS", "UNRATE"])
    assert len(report.results) == 2
    assert all(not r.ok for r in report.results)
    assert all(r.error for r in report.failed), "a failure must carry a reason"


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
def test_duplicate_observations_are_impossible(conn):
    spec = cat.BY_ID["PAYEMS"]
    ingestion.upsert_observations(conn, spec, [("2020-01-01", 100.0)])
    ingestion.upsert_observations(conn, spec, [("2020-01-01", 100.0)])
    n = conn.execute("SELECT COUNT(*) FROM observations WHERE series_id='PAYEMS'").fetchone()[0]
    assert n == 1


def test_revision_is_detected_and_recorded(conn):
    spec = cat.BY_ID["PAYEMS"]
    ingestion.upsert_observations(conn, spec, [("2020-01-01", 100.0)])
    new, rev = ingestion.upsert_observations(conn, spec, [("2020-01-01", 105.0)])
    assert (new, rev) == (0, 1)
    row = conn.execute("SELECT old_value, new_value, pct_change FROM revisions").fetchone()
    assert row["old_value"] == 100.0 and row["new_value"] == 105.0
    assert row["pct_change"] == pytest.approx(0.05)
    current = conn.execute(
        "SELECT value FROM observations WHERE series_id='PAYEMS'").fetchone()[0]
    assert current == 105.0, "the observation itself must hold the revised value"


def test_observations_before_history_start_are_dropped(conn):
    spec = cat.BY_ID["PAYEMS"]
    new, _ = ingestion.upsert_observations(conn, spec, [("1939-01-01", 29923.0),
                                                        ("2020-01-01", 100.0)])
    assert new == 1


# ---------------------------------------------------------------------------
# release-date alignment (spec 23)
# ---------------------------------------------------------------------------
def test_monthly_release_date_follows_the_period_end_not_the_start():
    """FRED stamps a monthly observation at the period START. A March CPI print
    dated 2026-03-01 is released in mid-April -- adding the lag to the start
    date would place it in mid-March and leak a month of information."""
    spec = cat.BY_ID["CPIAUCSL"]           # 13-day lag
    got = ingestion.estimate_release_date("2026-03-01", spec)
    assert got == "2026-04-13", got


def test_quarterly_release_date_uses_the_quarter_end():
    spec = cat.BY_ID["DRTSCILM"]           # quarterly, 30-day lag
    got = ingestion.estimate_release_date("2026-01-01", spec)
    assert got == "2026-04-30", got


def test_daily_release_date_is_effectively_immediate():
    spec = cat.BY_ID["DGS10"]              # daily, 1-day lag
    assert ingestion.estimate_release_date("2026-03-05", spec) == "2026-03-06"


def test_release_date_is_never_before_the_observation():
    for spec in cat.ALL_SERIES:
        obs = "2020-02-01"
        assert ingestion.estimate_release_date(obs, spec) >= obs, spec.sid


# ---------------------------------------------------------------------------
# quality engine
# ---------------------------------------------------------------------------
def test_frequency_mismatch_is_detected(conn):
    """Spec 23's test_monthly_series_not_falsely_daily."""
    spec = cat.BY_ID["PAYEMS"]             # declared monthly
    daily_rows = [(d, 100.0 + i) for i, d in enumerate(daily_dates(120))]
    issues = quality.detect_frequency_mismatch(daily_rows, spec)
    assert issues and "frequency mismatch" in issues[0]


def test_correct_frequency_raises_no_mismatch(conn):
    spec = cat.BY_ID["PAYEMS"]
    rows = [(d, 100.0 + i) for i, d in enumerate(month_dates(60))]
    assert quality.detect_frequency_mismatch(rows, spec) == []


def test_stale_frozen_value_is_detected():
    spec = cat.BY_ID["PAYEMS"]
    rows = [(d, 100.0) for d in month_dates(40)]
    issues = quality.detect_stale_values(rows, spec)
    assert issues and "frozen" in issues[0]


def test_future_dated_observation_is_flagged(conn):
    spec = cat.BY_ID["DGS10"]
    future = (date.today() + timedelta(days=30)).isoformat()
    conn.execute(
        "INSERT INTO observations (series_id, observation_date, value, frequency, source) "
        "VALUES (?,?,?,?,?)", (spec.sid, future, 4.0, spec.freq, "TEST"))
    conn.commit()
    assert quality.detect_timestamp_errors(conn, spec)


def test_outlier_is_flagged_but_not_removed():
    import math
    spec = cat.BY_ID["ICSA"]
    # Real variation in the history: a constant series has zero standard
    # deviation, so no value is any number of sigma away from it and the check
    # correctly declines to flag anything.
    rows = [(d, 200.0 + math.sin(i / 5.0) * 15.0)
            for i, d in enumerate(month_dates(80))]
    rows[-1] = (rows[-1][0], 6_000.0)      # a 2020-style print
    issues = quality.detect_outliers(rows, spec)
    assert issues and "flagged, not removed" in issues[0]
    assert rows[-1][1] == 6_000.0


def test_outlier_check_is_silent_on_a_constant_series():
    """Zero variance means no value is N sigma from the mean; flagging one
    would be a divide-by-zero dressed up as a finding."""
    spec = cat.BY_ID["ICSA"]
    rows = [(d, 200.0) for d in month_dates(80)]
    assert quality.detect_outliers(rows, spec) == []


def test_stale_series_lowers_the_quality_score(conn):
    spec = cat.BY_ID["PAYEMS"]
    old = [(d, 100.0 + i) for i, d in enumerate(
        month_dates(60, end=date.today() - timedelta(days=400)))]
    ingestion.upsert_observations(conn, spec, old)
    q = quality.assess_series(conn, spec)
    assert q.is_stale
    assert q.score < 0.85
    assert any("stale" in i.lower() for i in q.issues)


def test_short_history_series_is_marked_unusable_for_statistics():
    """EXHOSLUSM495S has ~12 months on FRED; a percentile from 12 points would
    be meaningless, so the engine must refuse to compute one."""
    spec = cat.BY_ID["EXHOSLUSM495S"]
    rows = [(d, 4_000_000.0 + i) for i, d in enumerate(month_dates(13))]
    assert not quality.has_sufficient_history(rows, spec)


def test_health_summary_shape(conn):
    q = quality.assess_all(conn, persist=False)
    h = quality.health_summary(q)
    assert 0.0 <= h["health"] <= 1.0
    assert h["n"] == len(cat.ALL_SERIES)
    assert h["high"] + h["medium"] + h["low"] == h["n"]
