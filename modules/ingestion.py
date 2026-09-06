"""
Ingestion -- pull public data into the local cache incrementally (spec 4, 5).

Three properties this module exists to guarantee:

1. NATIVE FREQUENCY IS PRESERVED. Nothing here resamples. A monthly series is
   stored on month-start dates, a weekly one on its own week-ending dates. Spec 4
   is explicit ("Do NOT force every series into daily frequency") and it matters:
   forward-filling CPI to daily would let a z-score built on 250 identical values
   masquerade as 250 independent observations.

2. INCREMENTAL. After the first backfill, a daily run asks FRED only for
   observations at or after each series' current last_obs (minus a re-check
   window, so revisions to recent prints are still seen). Spec 5: "Never download
   the entire historical dataset on every run."

3. FAILURES ARE VISIBLE. Each series is fetched inside its own try/except, so one
   retired FRED id cannot abort the run; what failed lands in api_errors and in
   the returned IngestReport, which the run log and dashboard both surface.

Batching: FRED's fredgraph.csv accepts comma-separated ids, so series are grouped
by (frequency, start-date) and pulled a batch at a time -- 81 series in ~10 calls.
Series with fallback candidate ids are fetched individually, because a batch that
contains one dead id returns a CSV with that column simply absent and we would
not know which id to retry.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from config import series as cat
from config import settings
from modules import sources
from modules.log import log_error, log_info, log_warning

# How far back to re-request on an incremental run. Statistical agencies revise
# the last few prints routinely (payrolls revises two months as a matter of
# course, GDP three quarters), so a window shorter than this would let revisions
# accumulate unseen in the cache.
RECHECK_DAYS = {"daily": 10, "weekly": 45, "monthly": 200, "quarterly": 500}

# A revision smaller than this is float noise or a rounding change, not news.
REVISION_EPS = 1e-9
REVISION_MATERIAL_PCT = 0.001   # 0.1% -- below this we store but do not alert


@dataclass
class SeriesResult:
    sid: str
    ok: bool
    resolved_id: str = ""
    rows_new: int = 0
    rows_revised: int = 0
    last_obs: str | None = None
    error: str = ""


@dataclass
class IngestReport:
    started_at: str
    results: list[SeriesResult] = field(default_factory=list)

    @property
    def ok(self) -> list[SeriesResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[SeriesResult]:
        return [r for r in self.results if not r.ok]

    @property
    def rows_new(self) -> int:
        return sum(r.rows_new for r in self.results)

    @property
    def rows_revised(self) -> int:
        return sum(r.rows_revised for r in self.results)

    def summary(self) -> str:
        return (f"{len(self.ok)}/{len(self.results)} series OK, "
                f"{self.rows_new} new obs, {self.rows_revised} revisions"
                + (f", FAILED: {', '.join(r.sid for r in self.failed)}" if self.failed else ""))


# ---------------------------------------------------------------------------
# release dates
# ---------------------------------------------------------------------------
def estimate_release_date(observation_date: str, spec: cat.Series) -> str:
    """When a value for `observation_date` most likely became public.

    For monthly/quarterly series the observation date is the START of the period
    (FRED convention), so the release cannot precede the period's END -- a March
    CPI print stamped 2026-03-01 is released in mid-April, not mid-March. Getting
    this wrong by a period is the single easiest way to introduce look-ahead
    bias, so the period end is computed explicitly rather than by adding a lag to
    the start date.
    """
    d = date.fromisoformat(observation_date)
    if spec.freq == "monthly":
        period_end = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    elif spec.freq == "quarterly":
        m = d.month + 2
        y = d.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        period_end = (date(y, m, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    elif spec.freq == "weekly":
        period_end = d + timedelta(days=6)
    else:
        period_end = d
    return (period_end + timedelta(days=spec.expected_lag_days)).isoformat()


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def upsert_observations(conn: sqlite3.Connection, spec: cat.Series,
                        rows: list[tuple[str, float]]) -> tuple[int, int]:
    """Insert new observations and detect revisions to existing ones.

    Returns (new_rows, revised_rows). A revision is recorded in `revisions` with
    both values so the dashboard can show what changed and the quality engine can
    flag a series that is being revised unusually hard.
    """
    if not rows:
        return 0, 0
    existing = {r["observation_date"]: r["value"] for r in conn.execute(
        "SELECT observation_date, value FROM observations WHERE series_id = ?", (spec.sid,))}

    new_count = rev_count = 0
    for obs_date, value in rows:
        if obs_date < settings.HISTORY_START:
            continue
        prior = existing.get(obs_date)
        if prior is None:
            new_count += 1
        elif abs(prior - value) > REVISION_EPS:
            rev_count += 1
            pct = (value - prior) / abs(prior) if prior else None
            conn.execute(
                "INSERT OR REPLACE INTO revisions "
                "(series_id, observation_date, old_value, new_value, pct_change) VALUES (?,?,?,?,?)",
                (spec.sid, obs_date, prior, value, pct))
        else:
            continue  # unchanged: nothing to write

        conn.execute(
            """
            INSERT INTO observations
                (series_id, observation_date, value, release_date, release_estimated,
                 frequency, source, fetched_at)
            VALUES (?,?,?,?,1,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(series_id, observation_date) DO UPDATE SET
                value=excluded.value, fetched_at=CURRENT_TIMESTAMP
            """,
            (spec.sid, obs_date, value, estimate_release_date(obs_date, spec),
             spec.freq, spec.source))
    conn.commit()
    return new_count, rev_count


def _mark_metadata(conn: sqlite3.Connection, sid: str, resolved: str, ok: bool) -> None:
    stats = conn.execute(
        "SELECT MIN(observation_date), MAX(observation_date), COUNT(*) "
        "FROM observations WHERE series_id = ?", (sid,)).fetchone()
    conn.execute(
        """
        UPDATE series_metadata SET
            resolved_id = COALESCE(NULLIF(?, ''), resolved_id),
            first_obs = ?, last_obs = ?, obs_count = ?,
            last_fetch_attempt = CURRENT_TIMESTAMP,
            last_fetch_ok = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE last_fetch_ok END
        WHERE series_id = ?
        """,
        (resolved, stats[0], stats[1], stats[2], 1 if ok else 0, sid))
    conn.commit()


# ---------------------------------------------------------------------------
# reading back
# ---------------------------------------------------------------------------
def last_observation(conn: sqlite3.Connection, sid: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(observation_date) FROM observations WHERE series_id = ?", (sid,)).fetchone()
    return row[0] if row and row[0] else None


def _incremental_start(conn: sqlite3.Connection, spec: cat.Series, full: bool) -> str:
    if full:
        return settings.HISTORY_START
    last = last_observation(conn, spec.sid)
    if not last:
        return settings.HISTORY_START
    back = RECHECK_DAYS.get(spec.freq, 60)
    return max(settings.HISTORY_START,
               (date.fromisoformat(last) - timedelta(days=back)).isoformat())


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def _fetch_one(conn, spec: cat.Series, start: str) -> SeriesResult:
    """Fetch a single series, trying each candidate id in order.

    Candidates exist because FRED renames and retires ids over the years (the
    LBMA gold fixings and the STLFSI generations are both in the catalogue). The
    first candidate that returns rows wins and is remembered as resolved_id.
    """
    errors = []
    for candidate in spec.candidates:
        try:
            rows = sources.parse_fred_single(
                sources.fetch_fred_csv(candidate, start=start, conn=conn))
        except sources.FetchError as e:
            errors.append(f"{candidate}: {e}")
            continue
        if rows:
            new, rev = upsert_observations(conn, spec, rows)
            _mark_metadata(conn, spec.sid, candidate, ok=True)
            return SeriesResult(spec.sid, True, candidate, new, rev, rows[-1][0])
        errors.append(f"{candidate}: empty")

    # Gold is the one series with a proven non-FRED fallback.
    if spec.sid == "GOLD":
        try:
            rows = sources.parse_yahoo_chart(sources.fetch_yahoo_chart("GC=F", conn=conn))
            if rows:
                new, rev = upsert_observations(conn, spec, rows)
                _mark_metadata(conn, spec.sid, "GC=F(yahoo)", ok=True)
                log_info(f"[ingest] GOLD via Yahoo GC=F ({len(rows)} obs) -- FRED ids retired")
                return SeriesResult(spec.sid, True, "GC=F(yahoo)", new, rev, rows[-1][0])
        except Exception as e:
            errors.append(f"yahoo GC=F: {e}")

    _mark_metadata(conn, spec.sid, "", ok=False)
    return SeriesResult(spec.sid, False, error="; ".join(errors))


def _fetch_batch(conn, specs: list[cat.Series], start: str) -> list[SeriesResult]:
    """Fetch several single-candidate series in one FRED call.

    If the batch call itself fails we fall back to fetching each series
    individually rather than declaring all of them dead -- a batch can fail for
    reasons (URL length, one bad id) that do not apply to the members.
    """
    ids = [s.sid for s in specs]
    try:
        cols = sources.parse_fred_csv(sources.fetch_fred_csv(ids, start=start, conn=conn))
    except sources.FetchError as e:
        log_warning(f"[ingest] batch of {len(ids)} failed ({e}); retrying individually")
        return [_fetch_one(conn, s, start) for s in specs]

    out = []
    for spec in specs:
        rows = cols.get(spec.sid) or []
        if not rows:
            # Present in our catalogue, absent from FRED's response -- retry
            # alone so the error message names the series, not the batch.
            out.append(_fetch_one(conn, spec, start))
            continue
        new, rev = upsert_observations(conn, spec, rows)
        _mark_metadata(conn, spec.sid, spec.sid, ok=True)
        out.append(SeriesResult(spec.sid, True, spec.sid, new, rev, rows[-1][0]))
    return out


def update(conn: sqlite3.Connection, *, full: bool = False,
           only: list[str] | None = None, batch_size: int = 12) -> IngestReport:
    """Run an ingestion pass.

    full=True re-pulls history from settings.HISTORY_START (the one-time
    backfill); the default incremental pass asks only for recent observations
    plus a revision re-check window.
    """
    report = IngestReport(started_at=datetime.now().isoformat(timespec="seconds"))
    specs = [s for s in cat.ALL_SERIES if not only or s.sid in only]

    # Group series that share a start date so they can share an HTTP call.
    # Multi-candidate series are excluded from batching (see _fetch_one).
    batches: dict[str, list[cat.Series]] = {}
    singles: list[cat.Series] = []
    for spec in specs:
        if len(spec.candidates) > 1:
            singles.append(spec)
        else:
            batches.setdefault(_incremental_start(conn, spec, full), []).append(spec)

    for start, group in batches.items():
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            report.results.extend(_fetch_batch(conn, chunk, start))

    for spec in singles:
        report.results.extend([_fetch_one(conn, spec, _incremental_start(conn, spec, full))])

    for r in report.failed:
        log_error(f"[ingest] {r.sid}: FAILED -- {r.error}")
    log_info(f"[ingest] {report.summary()}")
    return report


# ---------------------------------------------------------------------------
# release calendar
# ---------------------------------------------------------------------------
def recompute_release_dates(conn: sqlite3.Connection, only: list[str] | None = None) -> int:
    """Recompute observations.release_date from the CURRENT catalogue.

    release_date is derived data: it is a pure function of (observation_date,
    frequency, expected_lag_days). So whenever a series' declared frequency or
    lag changes in config/series.py, every stored release date for it becomes
    inconsistent with the catalogue -- and the vintage controller starts
    disagreeing with itself, because `_released_by` filters on the STORED value
    while `detect_future_data` recomputes from the catalogue.

    That is not hypothetical: the quality engine's frequency check caught four
    series declared at the wrong frequency (BUSLOANS, BAA, DTWEXBGS, DCOILWTICO),
    and after correcting them the backtest's leak detector immediately flagged
    all four -- correctly. This function is the fix, and it runs on every
    ingestion pass so the two can never drift apart again.
    """
    specs = [s for s in cat.ALL_SERIES if not only or s.sid in only]
    updated = 0
    for spec in specs:
        rows = conn.execute(
            "SELECT observation_date, release_date FROM observations WHERE series_id = ?",
            (spec.sid,)).fetchall()
        changes = [(estimate_release_date(r[0], spec), spec.sid, r[0])
                   for r in rows if estimate_release_date(r[0], spec) != r[1]]
        if changes:
            conn.executemany(
                "UPDATE observations SET release_date = ? WHERE series_id = ? "
                "AND observation_date = ?", changes)
            updated += len(changes)
            log_info(f"[ingest] recomputed {len(changes)} release dates for {spec.sid} "
                     f"({spec.freq}, {spec.expected_lag_days}d lag)")
    if updated:
        conn.commit()
    return updated


def rebuild_release_calendar(conn: sqlite3.Connection) -> int:
    """Materialise `releases` from observations + revisions.

    Two kinds of row land here: the first publication of an observation (from
    observations.release_date) and each later revision (from revisions, dated by
    when we detected it). The result is the engine's answer to "what did the
    world learn, and when" -- which is what the vintage controller filters on for
    series ALFRED does not cover.
    """
    # Release dates are derived from the catalogue, so refresh them first --
    # otherwise a catalogue edit silently leaves the calendar describing the old
    # publication schedule.
    recompute_release_dates(conn)
    conn.execute("DELETE FROM releases")
    conn.execute(
        """
        INSERT OR REPLACE INTO releases
            (series_id, release_date, observation_date, value, is_revision, prior_value)
        SELECT series_id, release_date, observation_date, value, 0, NULL
        FROM observations WHERE release_date IS NOT NULL
        """)
    conn.execute(
        """
        INSERT OR REPLACE INTO releases
            (series_id, release_date, observation_date, value, is_revision, prior_value)
        SELECT series_id, DATE(detected_at), observation_date, new_value, 1, old_value
        FROM revisions
        """)
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM releases").fetchone()[0]


def recent_revisions(conn: sqlite3.Connection, days: int = 30) -> list[dict]:
    """Material revisions in the last `days` -- what the dashboard's data page
    shows and what triggers a 'data revised' alert."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT series_id, observation_date, old_value, new_value, pct_change, detected_at
        FROM revisions WHERE DATE(detected_at) >= ?
        ORDER BY ABS(COALESCE(pct_change, 0)) DESC
        """, (cutoff,)).fetchall()
    return [dict(r) for r in rows
            if r["pct_change"] is None or abs(r["pct_change"]) >= REVISION_MATERIAL_PCT]
