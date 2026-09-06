"""
Vintage and look-ahead protection (spec 7) -- MANDATORY.

The one job of this module: when the engine is asked what it knew on 2007-07-01,
it must answer with 2007-07-01's numbers, not today's. Two distinct kinds of
future information have to be blocked, and they are different problems:

  1. FUTURE OBSERVATIONS. Q2 2007 GDP had not been published on 2007-07-01. This
     is a release-date filter and applies to every series.
  2. FUTURE REVISIONS. Q1 2007 GDP HAD been published -- but at a number that was
     later revised. Using today's revised value is look-ahead bias even though
     the observation date is safely in the past. This needs a genuine vintage.

Three tiers, chosen per series by the catalogue's `vintage` field, because
paying for tier A everywhere would cost tens of thousands of HTTP calls to
retrieve values that provably never changed:

  A. vintage="alfred"  True historical vintages pulled from ALFRED, stored in
                       `vintages`. Both problems solved exactly. Used for the
                       heavily-revised series (GDP-adjacent, payrolls, IP,
                       housing starts, CPI) -- 23 series.
  B. vintage="final"   Market prices, yields, spreads, diffusion indices. These
                       are never revised, so the value we hold today IS the
                       value that was on the screen that day. Only problem 1
                       applies, and a release-date filter solves it exactly.
                       This is not an approximation.
  C. vintage="lag"     Revised, but ALFRED coverage is thin or the series is
                       peripheral. Problem 1 is handled by the estimated release
                       date; problem 2 is NOT. Every result computed from a
                       tier-C series is flagged vintage_true=0 and the audit
                       table records it, so a backtest can never quietly present
                       lag-adjusted numbers as vintage-true.

`is_vintage_true()` reports which tier a given as-of date actually achieved, and
modules/backtest.py stamps that on every stored row.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta

from config import series as cat
from config import settings
from modules import sources
from modules.log import log_info, log_warning


@dataclass
class SeriesSlice:
    """One series as it was knowable on a given as-of date."""
    sid: str
    rows: list[tuple[str, float]]
    source_kind: str            # alfred | final | lag
    vintage_used: str | None    # the ALFRED vintage date, when tier A
    vintage_true: bool          # True if revisions are correctly reproduced

    @property
    def last_obs(self) -> str | None:
        return self.rows[-1][0] if self.rows else None


@dataclass
class AvailableData:
    as_of: str
    slices: dict[str, SeriesSlice] = field(default_factory=dict)

    @property
    def vintage_true(self) -> bool:
        """True only if EVERY series in the slice reproduces revisions correctly."""
        return all(s.vintage_true for s in self.slices.values())

    def coverage(self) -> float:
        return sum(1 for s in self.slices.values() if s.rows) / max(1, len(self.slices))

    def rows(self, sid: str) -> list[tuple[str, float]]:
        s = self.slices.get(sid)
        return s.rows if s else []


class VintageController:
    """Gatekeeper between the raw cache and anything that computes a forecast."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._vintage_dates: dict[str, list[str]] = {}

    # -- tier A: real ALFRED vintages ------------------------------------
    def available_vintages(self, sid: str) -> list[str]:
        if sid not in self._vintage_dates:
            self._vintage_dates[sid] = [
                r[0] for r in self.conn.execute(
                    "SELECT vintage_date FROM vintage_coverage WHERE series_id=? AND obs_count > 0 "
                    "ORDER BY vintage_date", (sid,))]
        return self._vintage_dates[sid]

    def get_vintage(self, series_id: str, vintage_date: str) -> list[tuple[str, float]]:
        """The exact stored vintage. Empty list if we never pulled that date."""
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT observation_date, value FROM vintages "
            "WHERE series_id=? AND vintage_date=? ORDER BY observation_date",
            (series_id, vintage_date))]

    def _latest_vintage_on_or_before(self, sid: str, as_of: str) -> str | None:
        """The newest vintage we hold that a forecaster on `as_of` could have seen.

        Strictly on-or-before: a vintage dated the day AFTER as_of contains a
        release that had not happened yet, which is exactly the leak this module
        exists to prevent.
        """
        candidates = [v for v in self.available_vintages(sid) if v <= as_of]
        return candidates[-1] if candidates else None

    # -- tiers B and C: release-date filtering ---------------------------
    def _released_by(self, sid: str, as_of: str) -> list[tuple[str, float]]:
        """Observations whose release_date is on or before as_of.

        For tier B this is exact. For tier C the release date is an estimate
        (observation period end + expected_lag_days), which is why tier C is
        never marked vintage_true.
        """
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT observation_date, value FROM observations "
            "WHERE series_id=? AND value IS NOT NULL AND release_date <= ? "
            "ORDER BY observation_date", (sid, as_of))]

    # -- the main entry point --------------------------------------------
    def get_available_data(self, as_of_date: str, sids: list[str] | None = None
                           ) -> AvailableData:
        """Everything the engine could legitimately have known on as_of_date."""
        out = AvailableData(as_of=as_of_date)
        specs = [cat.BY_ID[s] for s in sids] if sids else cat.ALL_SERIES

        for spec in specs:
            if spec.vintage == "alfred":
                vd = self._latest_vintage_on_or_before(spec.sid, as_of_date)
                if vd:
                    rows = self.get_vintage(spec.sid, vd)
                    # The stored vintage is the series as ALFRED had it on `vd`.
                    # Anything published between vd and as_of is genuinely absent
                    # -- which understates knowledge slightly but never overstates
                    # it, so the bias is conservative and in the safe direction.
                    out.slices[spec.sid] = SeriesSlice(spec.sid, rows, "alfred", vd, True)
                    continue
                # No vintage pulled for this date: fall back to the release-date
                # filter and be honest that revisions are not reproduced.
                out.slices[spec.sid] = SeriesSlice(
                    spec.sid, self._released_by(spec.sid, as_of_date), "lag", None, False)
                continue

            rows = self._released_by(spec.sid, as_of_date)
            true_ = spec.vintage == "final"
            out.slices[spec.sid] = SeriesSlice(spec.sid, rows, spec.vintage, None, true_)

        return out

    # -- guards -----------------------------------------------------------
    def enforce_release_date(self, rows: list[tuple[str, float]], spec: cat.Series,
                             as_of: str) -> list[tuple[str, float]]:
        """Drop any row whose estimated release post-dates as_of.

        A second line of defence applied to data that arrives from outside
        get_available_data (e.g. an ALFRED vintage that ALFRED itself stamped
        generously). Cheap, and it makes the leak test in tests/ meaningful.
        """
        from modules.ingestion import estimate_release_date
        return [(d, v) for d, v in rows if estimate_release_date(d, spec) <= as_of]

    def detect_future_data(self, data: AvailableData) -> list[str]:
        """Return a leak description for every series holding data it should not.

        Empty list == no leak. This is the assertion that spec 23's
        test_no_future_data_leakage checks, run in production rather than only
        in the test suite.
        """
        leaks = []
        as_of = data.as_of
        for sid, sl in data.slices.items():
            spec = cat.BY_ID.get(sid)
            if not spec or not sl.rows:
                continue
            if sl.last_obs and sl.last_obs > as_of:
                leaks.append(f"{sid}: observation {sl.last_obs} is after as-of {as_of}")
                continue
            # An observation dated before as_of can still be a leak if it could
            # not have been PUBLISHED yet -- the subtle case that separates a
            # real vintage engine from a naive date filter.
            from modules.ingestion import estimate_release_date
            if sl.source_kind != "alfred":
                latest_release = estimate_release_date(sl.last_obs, spec)
                if latest_release > as_of:
                    leaks.append(
                        f"{sid}: observation {sl.last_obs} would not publish until "
                        f"{latest_release}, after as-of {as_of}")
        return leaks

    def audit_backtest_inputs(self, backtest_id: str, data: AvailableData) -> dict:
        """Persist per-series provenance for one backtest as-of date (spec 7:
        'Every backtest run must log the data vintage used')."""
        leaks = set()
        for msg in self.detect_future_data(data):
            leaks.add(msg.split(":")[0])

        rows = []
        for sid, sl in data.slices.items():
            spec = cat.BY_ID.get(sid)
            latest_release = None
            if sl.last_obs and spec:
                from modules.ingestion import estimate_release_date
                latest_release = estimate_release_date(sl.last_obs, spec)
            rows.append((backtest_id, data.as_of, sid, sl.vintage_used, sl.source_kind,
                         sl.last_obs, latest_release, 1 if sid in leaks else 0))
        self.conn.executemany(
            "INSERT OR REPLACE INTO backtest_audit (backtest_id, as_of_date, series_id, "
            "vintage_used, source_kind, last_obs_used, latest_release, leak_detected) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()

        kinds: dict[str, int] = {}
        for sl in data.slices.values():
            kinds[sl.source_kind] = kinds.get(sl.source_kind, 0) + 1
        return {"as_of": data.as_of, "series": len(rows), "leaks": len(leaks),
                "by_kind": kinds, "vintage_true": data.vintage_true}


# ---------------------------------------------------------------------------
# building the vintage store
# ---------------------------------------------------------------------------
def vintage_grid(start: str, end: str, months: int = 3) -> list[str]:
    """The as-of dates the backtest will ask about, and therefore the vintages
    worth downloading.

    Quarterly by default: recession probability is a slow-moving quantity, and a
    monthly grid would triple the request count to resolve movement finer than
    the models can express. Dates land on the 15th, after the month's employment
    and CPI releases, so a grid point sits at a realistic decision moment rather
    than in the dead zone at the start of a month.
    """
    out: list[str] = []
    d = date.fromisoformat(start).replace(day=15)
    end_d = date.fromisoformat(end)
    while d <= end_d:
        out.append(d.isoformat())
        m = d.month + months
        d = d.replace(year=d.year + (m - 1) // 12, month=(m - 1) % 12 + 1)
    return out


def store_vintage(conn: sqlite3.Connection, sid: str, vintage_date: str,
                  rows: list[tuple[str, float]]) -> int:
    conn.executemany(
        "INSERT OR REPLACE INTO vintages (series_id, vintage_date, observation_date, value) "
        "VALUES (?,?,?,?)", [(sid, vintage_date, d, v) for d, v in rows])
    conn.execute(
        "INSERT OR REPLACE INTO vintage_coverage (series_id, vintage_date, obs_count) "
        "VALUES (?,?,?)", (sid, vintage_date, len(rows)))
    conn.commit()
    return len(rows)


def backfill_vintages(conn: sqlite3.Connection, *, start: str | None = None,
                      end: str | None = None, months: int = 3,
                      sids: list[str] | None = None, limit: int | None = None) -> dict:
    """Download the ALFRED vintage grid. Resumable: pairs already recorded in
    vintage_coverage are skipped, so an interrupted backfill continues where it
    stopped rather than re-downloading from the beginning.
    """
    start = start or settings.BACKTEST_START
    end = end or date.today().isoformat()
    specs = ([cat.BY_ID[s] for s in sids] if sids else cat.alfred_vintage_series())
    grid = vintage_grid(start, end, months)

    have = {(r[0], r[1]) for r in conn.execute(
        "SELECT series_id, vintage_date FROM vintage_coverage")}
    todo = [(spec, vd) for spec in specs for vd in grid if (spec.sid, vd) not in have]
    if limit:
        todo = todo[:limit]

    log_info(f"[vintage] {len(specs)} series x {len(grid)} vintages -> {len(todo)} to fetch")
    ok = empty = failed = 0
    for i, (spec, vd) in enumerate(todo, 1):
        try:
            cols = sources.parse_fred_csv(
                sources.fetch_alfred_vintage(spec.candidates[0], vd,
                                             start=settings.HISTORY_START, conn=conn))
            rows = next(iter(cols.values()), [])
        except sources.FetchError as e:
            # A 404 here is PERMANENT, not transient: ALFRED simply holds no
            # vintage for that series before FRED began archiving it (typically
            # 1997-2000, later for newer series). Record it as an empty vintage
            # so a resumed backfill skips it instead of re-spending a request on
            # every future run -- for a 1985 start that is over a thousand calls
            # per run that would otherwise be burnt rediscovering the same gap.
            if "HTTP 404" in str(e):
                store_vintage(conn, spec.sid, vd, [])
                empty += 1
            else:
                failed += 1
                log_warning(f"[vintage] {spec.sid}@{vd}: {e}")
            continue
        # Record even an empty result: it means ALFRED has no vintage that far
        # back (the series did not exist yet), and without the record we would
        # re-request it on every future backfill.
        store_vintage(conn, spec.sid, vd, rows)
        if rows:
            ok += 1
        else:
            empty += 1
        if i % 50 == 0:
            log_info(f"[vintage] {i}/{len(todo)} ({ok} ok, {empty} empty, {failed} failed)")

    return {"requested": len(todo), "ok": ok, "empty": empty, "failed": failed,
            "grid_points": len(grid), "series": len(specs)}


def coverage_report(conn: sqlite3.Connection) -> dict:
    """What the vintage store actually holds -- shown on the data-quality page so
    the tier each series achieved is visible rather than assumed."""
    rows = conn.execute(
        "SELECT series_id, COUNT(*) n, MIN(vintage_date) lo, MAX(vintage_date) hi, "
        "SUM(CASE WHEN obs_count > 0 THEN 1 ELSE 0 END) nonempty "
        "FROM vintage_coverage GROUP BY series_id ORDER BY series_id").fetchall()
    total_obs = conn.execute("SELECT COUNT(*) FROM vintages").fetchone()[0]
    return {
        "series": [dict(r) for r in rows],
        "total_vintage_rows": total_obs,
        "tier_a_series": len(cat.alfred_vintage_series()),
        "tier_a_covered": len(rows),
    }
