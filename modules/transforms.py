"""
Transformation engine (spec 9).

Turns raw series into comparable, direction-adjusted signals in the range
[-1, +1] where +1 always means "economically positive" (see the sign convention
in config/series.py).

THE LOOK-AHEAD TRAP IN NORMALISATION
------------------------------------
Spec 9 asks for z-scores and historical percentiles, and there is a subtle way
to get them wrong that no release-date filter can catch. Computing a z-score
over the FULL sample uses the mean and standard deviation of the entire history
-- including the future relative to any historical point. A 2007 observation
scored against 1970-2026 statistics has been told about 2008 and 2020. The
backtest would look impressive and the number would be meaningless.

So every normalisation here is EXPANDING: the statistics at observation t use
only observations up to and including t. This costs some stability early in a
series (which is what MIN_PERIODS guards) and is the only version that is
honest. `zscore_expanding` and `percentile_expanding` are the functions the
factor engine uses; `zscore` and `percentile` (full-sample) exist only for
descriptive display and are named so the difference is visible at the call site.

WHY WINSORISED tanh RATHER THAN CLIPPING
----------------------------------------
Turning a z-score into a [-1, +1] signal by clipping throws away the difference
between a 2-sigma and a 6-sigma reading, which is exactly the difference between
"weak" and "2008". tanh(z / SIGNAL_SCALE) is smooth, saturates gracefully, and
keeps the ordering intact at the tails.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from config import series as cat

# z / SIGNAL_SCALE before tanh. 2.0 puts a 2-sigma reading at |0.76| and a
# 4-sigma one at |0.96|: strong signals are distinguishable without a 1-sigma
# wobble looking like a crisis.
SIGNAL_SCALE = 2.0

# Minimum observations before an expanding statistic is trusted.
MIN_PERIODS = {"daily": 250, "weekly": 52, "monthly": 36, "quarterly": 12}

# Periods per year, used to size YoY lags at each native frequency.
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}


# ---------------------------------------------------------------------------
# series <-> frame
# ---------------------------------------------------------------------------
def to_series(rows: list[tuple[str, float]]) -> pd.Series:
    """[(date, value)] -> a date-indexed, sorted, deduplicated pandas Series."""
    if not rows:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="date"))
    idx = pd.to_datetime([d for d, _ in rows])
    s = pd.Series([v for _, v in rows], index=idx, dtype="float64", name="value")
    s.index.name = "date"
    return s[~s.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# 9. rate-of-change family
# ---------------------------------------------------------------------------
def calculate_yoy(s: pd.Series, freq: str) -> pd.Series:
    """Year-over-year percent change at the series' native frequency.

    Uses a positional lag rather than a calendar offset because macro series
    have irregular gaps (holidays, missing weeks) that would make a calendar
    shift silently return NaN for the very observations we care about most.
    """
    lag = PERIODS_PER_YEAR.get(freq, 12)
    if len(s) <= lag:
        return pd.Series(dtype="float64", index=s.index)
    return s.pct_change(lag, fill_method=None) * 100.0


def calculate_mom(s: pd.Series) -> pd.Series:
    return s.pct_change(1, fill_method=None) * 100.0


def calculate_qoq(s: pd.Series, freq: str) -> pd.Series:
    lag = max(1, PERIODS_PER_YEAR.get(freq, 12) // 4)
    return s.pct_change(lag, fill_method=None) * 100.0


def calculate_log_yoy(s: pd.Series, freq: str) -> pd.Series:
    """YoY in log space -- the right transform for indices that span orders of
    magnitude (Nasdaq went from 100 to 20,000), where a percent change at the
    start and the end of the sample are not comparable quantities."""
    lag = PERIODS_PER_YEAR.get(freq, 12)
    if len(s) <= lag:
        return pd.Series(dtype="float64", index=s.index)
    ls = np.log(s.where(s > 0))
    return (ls - ls.shift(lag)) * 100.0


def calculate_diff(s: pd.Series, freq: str, years: float = 1.0) -> pd.Series:
    """Change in the LEVEL over `years`. The correct transform for things
    already expressed as rates: the 10Y yield going 2% -> 4% is a doubling in
    percent terms but the economically meaningful fact is '+200bp'."""
    lag = max(1, int(PERIODS_PER_YEAR.get(freq, 12) * years))
    return s.diff(lag)


def calculate_moving_average(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 2)).mean()


def calculate_momentum(s: pd.Series, freq: str, months: int = 3) -> pd.Series:
    """Change over the last `months` -- is the series improving or deteriorating
    right now, independent of its level."""
    lag = max(1, round(PERIODS_PER_YEAR.get(freq, 12) * months / 12))
    return s.diff(lag)


def calculate_acceleration(s: pd.Series, freq: str, months: int = 3) -> pd.Series:
    """Second derivative: is the momentum itself building or fading. This is
    what turns 'weak' into 'weak but stabilising' at cycle turning points."""
    return calculate_momentum(calculate_momentum(s, freq, months), freq, months)


# ---------------------------------------------------------------------------
# 9. normalisation
# ---------------------------------------------------------------------------
def zscore_expanding(s: pd.Series, min_periods: int) -> pd.Series:
    """Point-in-time z-score: statistics at t use only data up to t.

    ddof=0 matches the population standard deviation used by the descriptive
    zscore() below, so the two differ only in their window, not their estimator.
    """
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def percentile_expanding(s: pd.Series, min_periods: int) -> pd.Series:
    """Point-in-time historical percentile in [0, 1].

    Implemented as a rank over the expanding window. O(n log n) via argsort
    rather than the O(n^2) of a naive expanding().apply, which matters because
    daily series here run to ~20,000 points.
    """
    vals = s.to_numpy(dtype="float64")
    n = len(vals)
    out = np.full(n, np.nan)
    if n == 0:
        return pd.Series(out, index=s.index)
    # Insertion position of each value within the sorted prefix before it.
    order: list[float] = []
    import bisect
    for i, v in enumerate(vals):
        if not math.isnan(v):
            if i + 1 >= min_periods and order:
                out[i] = bisect.bisect_left(order, v) / len(order)
            bisect.insort(order, v)
    return pd.Series(out, index=s.index)


def zscore(s: pd.Series) -> pd.Series:
    """Full-sample z-score. DESCRIPTIVE ONLY -- never feed this to a model or a
    backtest; it knows the whole history including the future. Kept for the
    dashboard's 'where does today sit in the record' displays."""
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd else np.nan)


def percentile(s: pd.Series) -> pd.Series:
    """Full-sample percentile. Same warning as zscore()."""
    return s.rank(pct=True)


def normalize_signal(z: pd.Series, direction: int) -> pd.Series:
    """z-score -> direction-adjusted signal in [-1, +1].

    After this, +1 means 'strongly positive for the economy' for every series in
    the engine, regardless of whether the underlying number goes up or down in
    good times. direction=0 series return NaN: they have no good/bad meaning and
    are handled by the regime classifiers instead.
    """
    if direction == 0:
        return pd.Series(np.nan, index=z.index)
    return np.tanh(z / SIGNAL_SCALE) * direction


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------
def apply_transform(s: pd.Series, spec: cat.Series) -> pd.Series:
    """Apply the catalogue's declared transform for this series."""
    kind = spec.transform
    if kind == "yoy":
        return calculate_yoy(s, spec.freq)
    if kind == "yoy3":
        # 3-period average first: housing starts and durable orders are noisy
        # enough that a single month's YoY is mostly sampling error.
        return calculate_yoy(calculate_moving_average(s, 3), spec.freq)
    if kind == "logyoy":
        return calculate_log_yoy(s, spec.freq)
    if kind == "diff12":
        return calculate_diff(s, spec.freq, 1.0)
    if kind == "level":
        return s
    raise ValueError(f"unknown transform {kind!r} for {spec.sid}")


class SignalFrame:
    """One series, fully transformed. What the factor engine consumes."""

    __slots__ = ("sid", "spec", "raw", "transformed", "z", "pct", "signal", "usable")

    def __init__(self, sid, spec, raw, transformed, z, pct, signal, usable):
        self.sid, self.spec, self.raw = sid, spec, raw
        self.transformed, self.z, self.pct, self.signal = transformed, z, pct, signal
        self.usable = usable

    def at(self, as_of: str | date | None = None) -> dict:
        """The latest values on or before `as_of` (or the very latest)."""
        def _last(series: pd.Series):
            if series is None or series.empty:
                return None
            sub = series if as_of is None else series[series.index <= pd.Timestamp(as_of)]
            sub = sub.dropna()
            if sub.empty:
                return None
            v = float(sub.iloc[-1])
            return None if math.isnan(v) else v

        idx = self.raw if as_of is None else self.raw[self.raw.index <= pd.Timestamp(as_of)]
        idx = idx.dropna()
        return {
            "sid": self.sid,
            "label": self.spec.label,
            "category": self.spec.category,
            "role": self.spec.role,
            "date": idx.index[-1].date().isoformat() if len(idx) else None,
            "raw": _last(self.raw),
            "transformed": _last(self.transformed),
            "zscore": _last(self.z),
            "percentile": _last(self.pct),
            "signal": _last(self.signal),
            "usable": self.usable,
        }

    def momentum(self, as_of: str | date | None = None, months: int = 3) -> float | None:
        """Change in the SIGNAL over `months` -- the arrow the dashboard draws."""
        sig = self.signal.dropna()
        if as_of is not None:
            sig = sig[sig.index <= pd.Timestamp(as_of)]
        if len(sig) < 2:
            return None
        lag = max(1, round(PERIODS_PER_YEAR.get(self.spec.freq, 12) * months / 12))
        if len(sig) <= lag:
            return None
        return float(sig.iloc[-1] - sig.iloc[-1 - lag])


def build_signal(rows: list[tuple[str, float]], spec: cat.Series) -> SignalFrame:
    """Raw observations -> transformed -> expanding z / percentile -> signal."""
    raw = to_series(rows)
    min_p = MIN_PERIODS.get(spec.freq, 36)

    if raw.empty:
        empty = pd.Series(dtype="float64")
        return SignalFrame(spec.sid, spec, raw, empty, empty, empty, empty, usable=False)

    transformed = apply_transform(raw, spec).replace([np.inf, -np.inf], np.nan)

    # Not enough history to describe a distribution -- return the transform but
    # no z/percentile/signal, so a 13-observation series contributes to breadth
    # counts and never to a percentile it cannot support (see EXHOSLUSM495S).
    valid = int(transformed.notna().sum())
    usable = valid >= min_p
    if not usable:
        empty = pd.Series(np.nan, index=raw.index)
        return SignalFrame(spec.sid, spec, raw, transformed, empty, empty, empty, usable=False)

    z = zscore_expanding(transformed, min_p)
    pct = percentile_expanding(transformed, min_p)
    signal = normalize_signal(z, spec.direction)
    return SignalFrame(spec.sid, spec, raw, transformed, z, pct, signal, usable=True)


def build_all(data: dict[str, list[tuple[str, float]]]) -> dict[str, SignalFrame]:
    """Transform every series we have rows for."""
    out: dict[str, SignalFrame] = {}
    for sid, rows in data.items():
        spec = cat.BY_ID.get(sid)
        if spec is None:
            continue
        out[sid] = build_signal(rows, spec)
    return out
