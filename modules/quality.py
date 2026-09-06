"""
Data Quality Engine (spec 6).

The governing rule from the spec is a single sentence: "A bad data source must
never silently produce a normal forecast." Everything here exists to make that
true. The engine grades every series HIGH/MEDIUM/LOW, and modules/confidence.py
reads those grades directly -- so when data degrades, the published confidence
falls and the dashboard's health panel says why, rather than the forecast
quietly continuing on stale numbers.

Each check is a small pure-ish function taking rows and returning issues, so a
new check is one function plus one line in assess().
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from config import series as cat
from config import settings

# A z-score beyond this against the series' own history is flagged as a possible
# outlier. 6 sigma, not 3: real macro data has fat tails (April 2020 claims were
# a genuine ~20 sigma print), and a threshold that fires on every recession would
# train the operator to ignore it.
OUTLIER_Z = 6.0

# Minimum observations before percentile/z-score statistics mean anything. Below
# this the transform engine returns None rather than a number computed from a
# sample too small to describe a distribution.
MIN_HISTORY = {"daily": 500, "weekly": 104, "monthly": 60, "quarterly": 20}

GRADE_THRESHOLDS = [(0.85, "HIGH"), (0.60, "MEDIUM")]   # else LOW


@dataclass
class SeriesQuality:
    sid: str
    grade: str
    score: float
    last_observation: str | None
    last_release: str | None
    observation_age: int | None       # days since last observation
    missing_pct: float
    is_stale: bool
    revision_status: str
    issues: list[str] = field(default_factory=list)

    def to_row(self) -> tuple:
        return (self.sid, datetime.now().isoformat(timespec="seconds"), self.grade,
                self.score, self.last_observation, self.last_release,
                self.observation_age, self.missing_pct, 1 if self.is_stale else 0,
                self.revision_status, json.dumps(self.issues))


# ---------------------------------------------------------------------------
# expected cadence
# ---------------------------------------------------------------------------
_PERIOD_DAYS = {"daily": 1, "weekly": 7, "monthly": 31, "quarterly": 92}


def expected_gap_days(spec: cat.Series) -> int:
    """Days that may pass between the newest observation DATE and today before
    the series is late. It is the publication lag plus one period, because on
    any given day the most recent period may simply not have ended yet."""
    return spec.expected_lag_days + _PERIOD_DAYS.get(spec.freq, 31)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def validate_schema(conn: sqlite3.Connection, spec: cat.Series) -> list[str]:
    """Catch rows that violate what the catalogue promises: NULL values, dates
    that are not ISO, and observations from the future."""
    issues = []
    bad = conn.execute(
        "SELECT COUNT(*) FROM observations WHERE series_id=? AND value IS NULL", (spec.sid,)
    ).fetchone()[0]
    if bad:
        issues.append(f"{bad} NULL values")
    malformed = conn.execute(
        "SELECT COUNT(*) FROM observations WHERE series_id=? AND "
        "(LENGTH(observation_date) != 10 OR observation_date NOT LIKE '____-__-__')",
        (spec.sid,)).fetchone()[0]
    if malformed:
        issues.append(f"{malformed} malformed dates")
    return issues


def detect_timestamp_errors(conn: sqlite3.Connection, spec: cat.Series) -> list[str]:
    """An observation dated after today is either a source error or a parsing
    bug on our side. Either way it must never reach the model, which would treat
    it as the newest reading."""
    today = date.today().isoformat()
    n = conn.execute(
        "SELECT COUNT(*) FROM observations WHERE series_id=? AND observation_date > ?",
        (spec.sid, today)).fetchone()[0]
    return [f"{n} observations dated in the future"] if n else []


def detect_duplicates(conn: sqlite3.Connection, spec: cat.Series) -> list[str]:
    """The PRIMARY KEY makes true duplicates impossible; this check exists to
    catch the case where that constraint is ever relaxed or bypassed."""
    n = conn.execute(
        "SELECT COUNT(*) FROM (SELECT observation_date FROM observations "
        "WHERE series_id=? GROUP BY observation_date HAVING COUNT(*) > 1)",
        (spec.sid,)).fetchone()[0]
    return [f"{n} duplicate observation dates"] if n else []


def detect_frequency_mismatch(rows: list[tuple[str, float]], spec: cat.Series) -> list[str]:
    """Verify the observed spacing matches the declared frequency.

    This is the check that catches spec 23's `test_monthly_series_not_falsely_daily`
    in production: if a monthly series ever arrives forward-filled to daily, its
    median gap collapses to 1 day and every z-score built on it silently becomes
    30x over-sampled.
    """
    if len(rows) < 12:
        return []
    dates = [date.fromisoformat(d) for d, _ in rows[-60:]]
    gaps = [(b - a).days for a, b in zip(dates, dates[1:]) if (b - a).days > 0]
    if not gaps:
        return []
    median_gap = statistics.median(gaps)
    expected = _PERIOD_DAYS.get(spec.freq, 31)
    # Generous bounds: months are 28-31 days, weekly series skip holidays, daily
    # series skip weekends (median gap 1, occasionally 3).
    lo, hi = expected * 0.4, expected * 2.2
    if not (lo <= median_gap <= hi):
        return [f"frequency mismatch: declared {spec.freq} but median gap is {median_gap:.0f}d"]
    return []


def detect_missing_values(rows: list[tuple[str, float]], spec: cat.Series) -> tuple[float, list[str]]:
    """Share of expected periods with no observation, over the last 3 years.

    Weekends and holidays make a naive count wrong for daily series, so the
    expected count for daily data uses 252 business days a year rather than 365.
    """
    if not rows:
        return 1.0, ["no observations"]
    cutoff = (date.today() - timedelta(days=365 * 3)).isoformat()
    recent = [r for r in rows if r[0] >= cutoff]
    if not recent:
        return 1.0, ["no observations in the last 3 years"]
    span_days = (date.fromisoformat(recent[-1][0]) - date.fromisoformat(recent[0][0])).days or 1
    per_year = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}[spec.freq]
    expected = max(1, round(span_days / 365.25 * per_year))
    missing = max(0.0, 1.0 - len(recent) / expected)
    issues = [f"{missing:.0%} of expected observations missing"] if missing > 0.15 else []
    return missing, issues


def detect_stale_values(rows: list[tuple[str, float]], spec: cat.Series) -> list[str]:
    """A series repeating one identical value far longer than it should.

    Distinguished from a legitimately flat series: policy rates sit at 0.00 for
    years by design, so direction=0 CONTEXT series are exempt. For everything
    else, a run of identical prints longer than a year of periods means the feed
    has frozen -- the classic silent-failure mode where an upstream cache serves
    yesterday's number forever.
    """
    if spec.direction == 0 or len(rows) < 24:
        return []
    last = rows[-1][1]
    run = 0
    for _, v in reversed(rows):
        if v != last:
            break
        run += 1
    per_year = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}[spec.freq]
    if run > per_year:
        return [f"value frozen at {last} for {run} consecutive observations"]
    return []


def detect_outliers(rows: list[tuple[str, float]], spec: cat.Series) -> list[str]:
    """Flag the newest observation if it is beyond OUTLIER_Z of its own history.

    Deliberately a flag, not a filter. The 2020 collapse was a genuine outlier
    and dropping it would have taught the model that recessions do not happen.
    """
    if len(rows) < 60:
        return []
    values = [v for _, v in rows]
    hist, latest = values[:-1], values[-1]
    mu = statistics.fmean(hist)
    sd = statistics.pstdev(hist)
    if sd == 0:
        return []
    z = (latest - mu) / sd
    if abs(z) > OUTLIER_Z:
        return [f"latest value {latest:g} is {z:+.1f} sigma vs history (flagged, not removed)"]
    return []


def detect_revision(conn: sqlite3.Connection, spec: cat.Series) -> tuple[str, list[str]]:
    """Summarise recent revision activity for the series."""
    cutoff = (date.today() - timedelta(days=90)).isoformat()
    rows = conn.execute(
        "SELECT pct_change FROM revisions WHERE series_id=? AND DATE(detected_at) >= ?",
        (spec.sid, cutoff)).fetchall()
    if not rows:
        return "none in 90d", []
    pcts = [abs(r[0]) for r in rows if r[0] is not None]
    biggest = max(pcts) if pcts else 0.0
    status = f"{len(rows)} revisions in 90d (max {biggest:.1%})"
    issues = [f"heavy revisions: {status}"] if biggest > 0.05 else []
    return status, issues


def calculate_data_freshness(rows: list[tuple[str, float]], spec: cat.Series
                             ) -> tuple[int | None, bool, list[str]]:
    """Age of the newest observation, and whether that makes the series stale."""
    if not rows:
        return None, True, ["no observations"]
    age = (date.today() - date.fromisoformat(rows[-1][0])).days
    limit = expected_gap_days(spec)
    if age > limit * settings.CRITICAL_STALE_MULTIPLIER:
        return age, True, [f"CRITICALLY stale: {age}d old, expected within {limit}d"]
    if age > limit * settings.STALE_MULTIPLIER:
        return age, True, [f"stale: {age}d old, expected within {limit}d"]
    return age, False, []


def has_sufficient_history(rows: list[tuple[str, float]], spec: cat.Series) -> bool:
    """Whether the series has enough observations for percentile/z statistics.

    modules/transforms.py consults this before producing a normalised signal;
    EXHOSLUSM495S (12 months of history on FRED) is the reason it exists.
    """
    return len(rows) >= MIN_HISTORY.get(spec.freq, 60)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def calculate_data_quality_score(missing_pct: float, is_stale: bool, age: int | None,
                                 spec: cat.Series, n_issues: int, enough_history: bool) -> float:
    """Combine the checks into 0..1.

    Weighting reflects what actually damages a forecast: staleness and missing
    data are heavily penalised because they mean the model is reasoning about a
    world it cannot see, while a flagged outlier costs little because the value
    is probably real.
    """
    score = 1.0
    score -= min(0.40, missing_pct * 1.5)
    if is_stale and age is not None:
        limit = expected_gap_days(spec)
        overdue = (age - limit) / max(limit, 1)
        score -= min(0.45, 0.15 + overdue * 0.10)
    if not enough_history:
        score -= 0.20
    score -= min(0.15, 0.05 * n_issues)
    return round(max(0.0, min(1.0, score)), 4)


def grade_for(score: float) -> str:
    for threshold, label in GRADE_THRESHOLDS:
        if score >= threshold:
            return label
    return "LOW"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def load_rows(conn: sqlite3.Connection, sid: str) -> list[tuple[str, float]]:
    return [(r[0], r[1]) for r in conn.execute(
        "SELECT observation_date, value FROM observations "
        "WHERE series_id=? AND value IS NOT NULL ORDER BY observation_date", (sid,))]


def assess_series(conn: sqlite3.Connection, spec: cat.Series) -> SeriesQuality:
    rows = load_rows(conn, spec.sid)
    issues: list[str] = []
    issues += validate_schema(conn, spec)
    issues += detect_timestamp_errors(conn, spec)
    issues += detect_duplicates(conn, spec)
    issues += detect_frequency_mismatch(rows, spec)
    missing_pct, miss_issues = detect_missing_values(rows, spec)
    issues += miss_issues
    issues += detect_stale_values(rows, spec)
    issues += detect_outliers(rows, spec)
    revision_status, rev_issues = detect_revision(conn, spec)
    issues += rev_issues
    age, is_stale, fresh_issues = calculate_data_freshness(rows, spec)
    issues += fresh_issues

    enough = has_sufficient_history(rows, spec)
    if not enough:
        issues.append(f"insufficient history for statistics ({len(rows)} obs)")

    score = calculate_data_quality_score(missing_pct, is_stale, age, spec, len(issues), enough)
    last_release = conn.execute(
        "SELECT MAX(release_date) FROM observations WHERE series_id=?", (spec.sid,)).fetchone()[0]

    return SeriesQuality(
        sid=spec.sid, grade=grade_for(score), score=score,
        last_observation=rows[-1][0] if rows else None, last_release=last_release,
        observation_age=age, missing_pct=round(missing_pct, 4), is_stale=is_stale,
        revision_status=revision_status, issues=issues)


def assess_all(conn: sqlite3.Connection, persist: bool = True) -> dict[str, SeriesQuality]:
    out = {s.sid: assess_series(conn, s) for s in cat.ALL_SERIES}
    if persist:
        conn.executemany(
            "INSERT OR REPLACE INTO data_quality (series_id, assessed_at, grade, score, "
            "last_observation, last_release, observation_age, missing_pct, is_stale, "
            "revision_status, issues) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [q.to_row() for q in out.values()])
        conn.commit()
    return out


def health_summary(quality: dict[str, SeriesQuality]) -> dict:
    """The single number the dashboard's 'Data Health: 96%' badge shows, plus
    the counts behind it (spec 29)."""
    if not quality:
        return {"health": 0.0, "n": 0, "stale": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    scores = [q.score for q in quality.values()]
    stale = sum(1 for q in quality.values() if q.is_stale)
    critical = sum(1 for q in quality.values()
                   if any("CRITICALLY" in i or "no observations" in i for i in q.issues))
    return {
        "health": round(sum(scores) / len(scores), 4),
        "n": len(quality),
        "stale": stale,
        "critical": critical,
        "high": sum(1 for q in quality.values() if q.grade == "HIGH"),
        "medium": sum(1 for q in quality.values() if q.grade == "MEDIUM"),
        "low": sum(1 for q in quality.values() if q.grade == "LOW"),
        "stale_series": sorted(q.sid for q in quality.values() if q.is_stale),
    }
