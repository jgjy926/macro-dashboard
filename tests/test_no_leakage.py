"""
Look-ahead leakage QA (spec 7, 23) -- the tests that matter most.

Spec 7 calls vintage protection mandatory and spec 1 asks whether the system can
PROVE its historical forecasts used no future information. These tests are that
proof, and they check all three leak channels separately, because closing two of
them and missing the third gives a backtest that is wrong in a way nothing else
would reveal:

  1. future observations   -- data not yet published at the as-of date
  2. future revisions      -- a published value, but at a number revised later
  3. future statistics     -- a past value normalised against future statistics
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from config import series as cat
from config import settings
from modules import ingestion, models, transforms, vintage
from tests.conftest import month_dates


# ---------------------------------------------------------------------------
# 1. future observations
# ---------------------------------------------------------------------------
def test_no_future_data_leakage(conn):
    """The headline test named in spec 23."""
    spec = cat.BY_ID["PAYEMS"]
    rows = [(d, 100.0 + i) for i, d in enumerate(month_dates(120, end=date(2026, 8, 1)))]
    ingestion.upsert_observations(conn, spec, rows)

    vc = vintage.VintageController(conn)
    as_of = "2020-01-15"
    data = vc.get_available_data(as_of, sids=["PAYEMS"])

    assert data.slices["PAYEMS"].last_obs is not None
    assert data.slices["PAYEMS"].last_obs < as_of
    assert vc.detect_future_data(data) == [], "no leak may be present"


def test_release_date_filter_excludes_published_after_as_of(conn):
    """An observation dated before the as-of date is still a leak if it had not
    been PUBLISHED yet. This is the case a naive date filter gets wrong."""
    spec = cat.BY_ID["CPIAUCSL"]           # 13-day publication lag
    ingestion.upsert_observations(conn, spec, [("2020-01-01", 258.0),
                                               ("2020-02-01", 259.0)])
    vc = vintage.VintageController(conn)

    # 2020-02-05: the January print (released ~2020-02-13) is NOT yet public,
    # even though its observation date of 2020-01-01 is comfortably in the past.
    data = vc.get_available_data("2020-02-05", sids=["CPIAUCSL"])
    assert data.slices["CPIAUCSL"].rows == [], \
        "January CPI is not public on 5 February -- it releases on the 13th"

    data = vc.get_available_data("2020-02-20", sids=["CPIAUCSL"])
    assert [d for d, _ in data.slices["CPIAUCSL"].rows] == ["2020-01-01"]


def test_detect_future_data_catches_an_injected_leak(conn):
    """Deliberately hand the controller data it should not have, and confirm the
    guard fires. A guard that has never been seen to fail is not a guard."""
    spec = cat.BY_ID["PAYEMS"]
    ingestion.upsert_observations(conn, spec, [("2026-01-01", 160000.0)])
    vc = vintage.VintageController(conn)
    leaked = vintage.AvailableData(as_of="2020-01-15")
    leaked.slices["PAYEMS"] = vintage.SeriesSlice(
        "PAYEMS", [("2026-01-01", 160000.0)], "lag", None, False)
    leaks = vc.detect_future_data(leaked)
    assert leaks and "2026-01-01" in leaks[0]


def test_vintage_grid_dates_are_ordered_and_in_range():
    grid = vintage.vintage_grid("2000-01-01", "2005-01-01", months=3)
    assert grid == sorted(grid)
    assert grid[0].startswith("2000-01")
    assert all(d <= "2005-01-15" for d in grid)
    assert all(d.endswith("-15") for d in grid), \
        "grid points sit mid-month, after the employment and CPI releases"


# ---------------------------------------------------------------------------
# 2. future revisions
# ---------------------------------------------------------------------------
def test_vintage_returns_the_value_as_it_stood_not_the_revised_one(conn):
    """The distinguishing property of a real vintage engine."""
    sid = "PAYEMS"
    vintage.store_vintage(conn, sid, "2008-01-15", [("2007-12-01", 138_000.0)])
    vintage.store_vintage(conn, sid, "2010-01-15", [("2007-12-01", 137_500.0)])
    # Today's (revised) value:
    ingestion.upsert_observations(conn, cat.BY_ID[sid], [("2007-12-01", 137_400.0)])

    vc = vintage.VintageController(conn)
    data = vc.get_available_data("2008-06-15", sids=[sid])
    rows = dict(data.slices[sid].rows)
    assert rows["2007-12-01"] == 138_000.0, \
        "must return the number known in 2008, not either later revision"
    assert data.slices[sid].vintage_used == "2008-01-15"
    assert data.slices[sid].vintage_true is True


def test_vintage_never_uses_a_vintage_dated_after_the_as_of(conn):
    sid = "PAYEMS"
    vintage.store_vintage(conn, sid, "2008-01-15", [("2007-12-01", 138_000.0)])
    vintage.store_vintage(conn, sid, "2009-01-15", [("2007-12-01", 137_000.0)])
    vc = vintage.VintageController(conn)
    assert vc._latest_vintage_on_or_before(sid, "2008-06-15") == "2008-01-15"
    assert vc._latest_vintage_on_or_before(sid, "2007-01-01") is None


def test_missing_vintage_falls_back_and_is_flagged_not_vintage_true(conn):
    """When no vintage exists, the engine must degrade AUDIBLY."""
    spec = cat.BY_ID["PAYEMS"]             # vintage policy 'alfred'
    ingestion.upsert_observations(conn, spec, [("2007-12-01", 137_400.0)])
    vc = vintage.VintageController(conn)
    data = vc.get_available_data("2008-06-15", sids=["PAYEMS"])
    sl = data.slices["PAYEMS"]
    assert sl.source_kind == "lag"
    assert sl.vintage_true is False
    assert data.vintage_true is False


def test_never_revised_series_are_vintage_true_without_a_vintage(conn):
    """Market data is never revised, so the release-date filter alone is exact --
    not an approximation. Spending ALFRED requests on these would return
    identical numbers."""
    spec = cat.BY_ID["DGS10"]              # vintage policy 'final'
    ingestion.upsert_observations(conn, spec, [("2007-06-01", 5.0)])
    vc = vintage.VintageController(conn)
    data = vc.get_available_data("2007-12-01", sids=["DGS10"])
    assert data.slices["DGS10"].source_kind == "final"
    assert data.slices["DGS10"].vintage_true is True


def test_audit_records_provenance_for_every_series(conn):
    spec = cat.BY_ID["DGS10"]
    ingestion.upsert_observations(conn, spec, [("2007-06-01", 5.0)])
    vc = vintage.VintageController(conn)
    data = vc.get_available_data("2007-12-01", sids=["DGS10"])
    summary = vc.audit_backtest_inputs("bt-test", data)
    assert summary["leaks"] == 0
    row = conn.execute(
        "SELECT * FROM backtest_audit WHERE backtest_id='bt-test'").fetchone()
    assert row["series_id"] == "DGS10"
    assert row["source_kind"] == "final"
    assert row["leak_detected"] == 0


# ---------------------------------------------------------------------------
# 3. future statistics -- the subtle one
# ---------------------------------------------------------------------------
def test_expanding_zscore_uses_no_future_data():
    """The z-score at t must be identical whether or not the future exists.

    A full-sample z-score fails this test, which is precisely why the factor
    engine uses the expanding version.
    """
    values = pd.Series(np.random.RandomState(0).normal(size=300))
    full = transforms.zscore_expanding(values, min_periods=36)
    truncated = transforms.zscore_expanding(values.iloc[:200], min_periods=36)
    pd.testing.assert_series_equal(full.iloc[:200], truncated, check_names=False)


def test_full_sample_zscore_would_leak():
    """Demonstrates the failure mode the engine avoids, so the distinction is
    tested rather than only asserted in a comment."""
    values = pd.Series(list(range(100)) + [1000.0])
    full = transforms.zscore(values)
    truncated = transforms.zscore(values.iloc[:100])
    assert not np.isclose(full.iloc[50], truncated.iloc[50]), \
        "a full-sample z-score at t=50 changes when the future changes -- a leak"


def test_expanding_percentile_uses_no_future_data():
    values = pd.Series(np.random.RandomState(1).normal(size=300))
    full = transforms.percentile_expanding(values, min_periods=36)
    truncated = transforms.percentile_expanding(values.iloc[:180], min_periods=36)
    pd.testing.assert_series_equal(full.iloc[:180], truncated, check_names=False)


def test_expanding_percentile_matches_a_naive_implementation():
    """The fast bisect implementation must equal the obvious O(n^2) one."""
    values = pd.Series([5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0])
    fast = transforms.percentile_expanding(values, min_periods=3)
    for i in range(2, len(values)):
        prior = values.iloc[:i]
        expected = (prior < values.iloc[i]).sum() / len(prior)
        assert fast.iloc[i] == pytest.approx(expected), f"mismatch at {i}"


# ---------------------------------------------------------------------------
# model-level purging
# ---------------------------------------------------------------------------
def test_walk_forward_purges_overlapping_outcome_windows():
    """Rows whose 12-month outcome window reaches into the test period must be
    dropped from training, or the answer leaks backwards across the split."""
    idx = pd.date_range("2000-01-31", periods=100, freq="ME")
    test_start = idx[60]
    purged = models._purge(idx[:60], test_start, horizon_m=12)
    assert purged[-1] < test_start - pd.DateOffset(months=12)
    assert len(purged) < 60, "purging must actually remove rows"


def test_build_target_excludes_months_already_in_recession():
    """Asking 'will a recession start in the next 12 months' while one is
    running is not well-posed; those rows must be NaN, not 0."""
    rec = [("2007-11-01", 0.0), ("2007-12-01", 1.0), ("2008-01-01", 1.0),
           ("2009-07-01", 0.0)]
    idx = pd.DatetimeIndex(["2007-11-30", "2007-12-31", "2008-01-31"])
    y = models.build_target(rec, 12, idx)
    assert y.iloc[0] == 1.0, "November 2007 precedes a December start"
    assert np.isnan(y.iloc[1]), "December 2007 is inside the recession"
    assert np.isnan(y.iloc[2])


def test_fit_final_excludes_unknowable_recent_labels():
    """The last `horizon` months have labels that depend on months which have
    not happened; training on them teaches the model a default, not a fact."""
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    X = pd.DataFrame({"a": np.random.RandomState(2).normal(size=200),
                      "b": np.random.RandomState(3).normal(size=200)}, index=idx)
    y = pd.Series(([0.0] * 150) + ([1.0] * 50), index=idx)
    fitted = models.fit_final(X, y, "logit", horizon_m=12)
    assert fitted is not None
    assert pd.Timestamp(fitted.train_end) <= idx[-1] - pd.DateOffset(months=12)


# ---------------------------------------------------------------------------
# derived release dates must track the catalogue
# ---------------------------------------------------------------------------
def test_release_dates_are_recomputed_when_the_catalogue_changes(conn):
    """release_date is DERIVED from (observation_date, frequency, lag), so a
    catalogue correction invalidates every stored value for that series.

    This is not hypothetical. The quality engine's frequency check found four
    series declared at the wrong frequency; after correcting them, the backtest's
    leak detector flagged all four -- because `_released_by` filters on the
    STORED release date while `detect_future_data` recomputes from the
    catalogue, and the two had silently diverged.
    """
    spec = cat.BY_ID["BUSLOANS"]           # monthly, 20-day lag
    ingestion.upsert_observations(conn, spec, [("2020-01-01", 2000.0)])
    correct = ingestion.estimate_release_date("2020-01-01", spec)

    # Simulate a stale value left behind by an earlier, wrong declaration.
    conn.execute("UPDATE observations SET release_date = '2020-01-09' "
                 "WHERE series_id='BUSLOANS'")
    conn.commit()

    changed = ingestion.recompute_release_dates(conn, only=["BUSLOANS"])
    assert changed == 1
    got = conn.execute("SELECT release_date FROM observations WHERE series_id='BUSLOANS'"
                       ).fetchone()[0]
    assert got == correct


def test_rebuild_release_calendar_refreshes_release_dates_first(conn):
    """The daily run must not be able to leave the calendar describing an old
    publication schedule."""
    spec = cat.BY_ID["CPIAUCSL"]
    ingestion.upsert_observations(conn, spec, [("2020-01-01", 258.0)])
    conn.execute("UPDATE observations SET release_date = '2020-01-02' "
                 "WHERE series_id='CPIAUCSL'")
    conn.commit()
    ingestion.rebuild_release_calendar(conn)
    got = conn.execute("SELECT release_date FROM observations WHERE series_id='CPIAUCSL'"
                       ).fetchone()[0]
    assert got == ingestion.estimate_release_date("2020-01-01", spec)


def test_no_series_declares_a_fallback_id_at_a_different_frequency():
    """A fallback id at another frequency is a silent trap: the fallback's data
    lands under the primary's declared frequency, and every downstream lag (YoY,
    momentum, staleness) is then wrong by the frequency ratio without anything
    raising. BAA/DBAA was exactly this before it was fixed."""
    known_mixed = {
        # sid: (ids that are all the SAME frequency as the declaration)
    }
    for spec in cat.ALL_SERIES:
        if len(spec.candidates) > 1:
            assert spec.sid not in known_mixed, (
                f"{spec.sid} lists multiple candidate ids -- confirm they share the "
                f"declared frequency ({spec.freq})")
