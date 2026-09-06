"""
Leading indicator engine (spec 11).

Produces the breadth panel -- "18 indicators: 3 improving, 4 neutral, 11
weakening" -- plus the two things breadth alone cannot tell you:

  PERSISTENCE  One weak month is noise. The same weakness for three consecutive
               readings is a trend. Without this the panel would flip on
               rounding and every reading would look like a turning point.

  DIVERGENCE   The classic turning-point tell is leading indicators rolling over
               while coincident data still looks fine -- by the time the
               coincident series confirm, the turn is months old. A single
               composite average hides exactly this, because the healthy
               coincident readings cancel the deteriorating leading ones.

Composite indices here are equal-weighted within each role class. Not because
equal weighting is optimal, but because it is honest: with ~8 recessions there is
no defensible way to fit indicator-level weights without overfitting, and an
equal-weighted diffusion index is what the Conference Board's LEI does too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import series as cat
from config import strategy as st
from modules.transforms import PERIODS_PER_YEAR, SignalFrame


@dataclass
class IndicatorState:
    sid: str
    label: str
    signal: float
    momentum: float | None
    state: str            # IMPROVING | NEUTRAL | WEAKENING
    date: str | None
    percentile: float | None


@dataclass
class Breadth:
    total: int
    improving: int
    neutral: int
    weakening: int
    indicators: list[IndicatorState] = field(default_factory=list)

    @property
    def weakening_pct(self) -> float:
        return self.weakening / self.total if self.total else 0.0

    @property
    def improving_pct(self) -> float:
        return self.improving / self.total if self.total else 0.0

    @property
    def net(self) -> float:
        """Diffusion in [-1, +1]: (improving - weakening) / total."""
        return (self.improving - self.weakening) / self.total if self.total else 0.0


def _classify(momentum: float | None) -> str:
    if momentum is None:
        return "NEUTRAL"
    if momentum > st.BREADTH_EPS:
        return "IMPROVING"
    if momentum < -st.BREADTH_EPS:
        return "WEAKENING"
    return "NEUTRAL"


def calculate_signal_breadth(frames: dict[str, SignalFrame], as_of: str | None = None
                             ) -> Breadth:
    """Count leading indicators improving / neutral / weakening.

    Direction is taken from 3-month MOMENTUM of the signal, not from the signal
    level. A series can be at a weak level and improving -- that is what a
    turning point looks like, and a level-based count would call it weak right
    up to the recovery.
    """
    states: list[IndicatorState] = []
    for spec in cat.leading_series():
        frame = frames.get(spec.sid)
        if frame is None or not frame.usable:
            continue
        snap = frame.at(as_of)
        if snap["signal"] is None or math.isnan(snap["signal"]):
            continue
        mom = frame.momentum(as_of, st.MOMENTUM_MONTHS)
        states.append(IndicatorState(
            sid=spec.sid, label=spec.label, signal=float(snap["signal"]),
            momentum=mom, state=_classify(mom), date=snap["date"],
            percentile=snap["percentile"]))

    return Breadth(
        total=len(states),
        improving=sum(1 for s in states if s.state == "IMPROVING"),
        neutral=sum(1 for s in states if s.state == "NEUTRAL"),
        weakening=sum(1 for s in states if s.state == "WEAKENING"),
        indicators=sorted(states, key=lambda s: (s.momentum if s.momentum is not None else 0.0)))


# ---------------------------------------------------------------------------
# composite indices
# ---------------------------------------------------------------------------
def _role_index(frames: dict[str, SignalFrame], role: str) -> pd.Series:
    """Equal-weighted index of every usable series with the given role.

    Series are aligned on a monthly grid before averaging. This is the ONE place
    the engine resamples, and it resamples DOWN (daily -> monthly, taking the
    last observation in each month) rather than up. Downsampling discards
    information; upsampling would invent it, which is what spec 4 forbids.
    """
    cols = []
    for spec in cat.factor_series():
        if spec.role != role or spec.direction == 0:
            continue
        frame = frames.get(spec.sid)
        if frame is None or not frame.usable or frame.signal.empty:
            continue
        s = frame.signal.dropna()
        if s.empty:
            continue
        cols.append(s.resample("ME").last().rename(spec.sid))
    if not cols:
        return pd.Series(dtype="float64")
    df = pd.concat(cols, axis=1)
    # min 3 members so a month where one series happens to exist does not become
    # the "index" for that month.
    return df.mean(axis=1, skipna=True).where(df.notna().sum(axis=1) >= 3)


def calculate_leading_indicator_index(frames: dict[str, SignalFrame]) -> pd.Series:
    return _role_index(frames, "LEADING")


def calculate_coincident_index(frames: dict[str, SignalFrame]) -> pd.Series:
    return _role_index(frames, "COINCIDENT")


def calculate_lagging_index(frames: dict[str, SignalFrame]) -> pd.Series:
    return _role_index(frames, "LAGGING")


def _at(series: pd.Series, as_of: str | None) -> float | None:
    s = series.dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    return None if s.empty else float(s.iloc[-1])


def calculate_signal_acceleration(index: pd.Series, as_of: str | None = None,
                                  months: int = 3) -> float | None:
    """Is the index's rate of change itself building or fading (spec 11)."""
    s = index.dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    if len(s) < months * 2 + 1:
        return None
    recent = float(s.iloc[-1] - s.iloc[-1 - months])
    prior = float(s.iloc[-1 - months] - s.iloc[-1 - months * 2])
    return round(recent - prior, 4)


def calculate_signal_persistence(frames: dict[str, SignalFrame], as_of: str | None = None,
                                 periods: int = st.PERSISTENCE_PERIODS) -> dict:
    """How many consecutive monthly readings the leading index has moved one way.

    Answers "is this a trend or a print?" -- the question a single breadth
    snapshot cannot.
    """
    index = calculate_leading_indicator_index(frames).dropna()
    if as_of is not None:
        index = index[index.index <= pd.Timestamp(as_of)]
    if len(index) < periods + 1:
        return {"direction": "UNKNOWN", "months": 0, "persistent": False}

    diffs = index.diff().dropna()
    if diffs.empty:
        return {"direction": "UNKNOWN", "months": 0, "persistent": False}
    last_sign = np.sign(diffs.iloc[-1])
    run = 0
    for d in reversed(diffs.to_numpy()):
        if np.sign(d) != last_sign or d == 0:
            break
        run += 1
    direction = "IMPROVING" if last_sign > 0 else "WEAKENING" if last_sign < 0 else "FLAT"
    return {"direction": direction, "months": run, "persistent": run >= periods,
            "interpretation": (
                f"Leading index has been {direction.lower()} for {run} consecutive months"
                + (" -- a persistent trend, not a single print."
                   if run >= periods else " -- not yet persistent."))}


def detect_macro_divergence(frames: dict[str, SignalFrame], as_of: str | None = None) -> dict:
    """Leading versus coincident: the turning-point tell (spec 11)."""
    lead = calculate_leading_indicator_index(frames)
    coin = calculate_coincident_index(frames)
    lag = calculate_lagging_index(frames)

    l, c, g = _at(lead, as_of), _at(coin, as_of), _at(lag, as_of)
    if l is None or c is None:
        return {"diverging": False, "reason": "insufficient index coverage"}

    gap = l - c
    diverging = abs(gap) > st.DIVERGENCE_EPS
    if diverging and gap < 0:
        kind, meaning = "NEGATIVE", (
            "Leading indicators have rolled over while coincident data still looks "
            "healthy. This is the classic pre-turn configuration -- coincident "
            "series confirm a downturn only after it has begun.")
    elif diverging:
        kind, meaning = "POSITIVE", (
            "Leading indicators are improving ahead of still-weak coincident data. "
            "The classic pre-recovery configuration.")
    else:
        kind, meaning = "NONE", "Leading and coincident indicators broadly agree."

    return {"leading": round(l, 4), "coincident": round(c, 4),
            "lagging": None if g is None else round(g, 4),
            "gap": round(gap, 4), "diverging": diverging, "kind": kind,
            "interpretation": meaning}


def detect_turning_point(frames: dict[str, SignalFrame], as_of: str | None = None) -> dict:
    """Is the cycle turning right now?

    Requires three things to agree, because any one of them alone fires
    constantly: the leading index at an extreme, its direction reversing, and
    that reversal persisting. Demanding all three is what separates a turning
    point from a wobble.
    """
    index = calculate_leading_indicator_index(frames).dropna()
    if as_of is not None:
        index = index[index.index <= pd.Timestamp(as_of)]
    if len(index) < 15:
        return {"turning": False, "reason": "insufficient history"}

    level = float(index.iloc[-1])
    change_3m = float(index.iloc[-1] - index.iloc[-4])
    change_prior_3m = float(index.iloc[-4] - index.iloc[-7])
    persistence = calculate_signal_persistence(frames, as_of)
    accel = calculate_signal_acceleration(index, as_of)

    reversal = np.sign(change_3m) != np.sign(change_prior_3m) and abs(change_3m) > 0.05
    at_extreme = abs(level) > 0.25
    turning = bool(reversal and at_extreme and persistence["persistent"])

    if turning:
        direction = "UPTURN" if change_3m > 0 else "DOWNTURN"
        msg = (f"Turning point detected: {direction.lower()} from a "
               f"{'weak' if level < 0 else 'strong'} leading index "
               f"({level:+.2f}), reversing after {persistence['months']} months.")
    else:
        direction = "NONE"
        msg = "No turning point: the reversal, extreme-level and persistence conditions do not all hold."

    return {"turning": turning, "direction": direction, "level": round(level, 4),
            "change_3m": round(change_3m, 4), "acceleration": accel,
            "persistence_months": persistence["months"], "interpretation": msg}


def summarise(frames: dict[str, SignalFrame], as_of: str | None = None) -> dict:
    """Everything the dashboard's Leading Indicators page needs, in one call."""
    breadth = calculate_signal_breadth(frames, as_of)
    lead = calculate_leading_indicator_index(frames)
    coin = calculate_coincident_index(frames)
    lag = calculate_lagging_index(frames)
    return {
        "breadth": {
            "total": breadth.total, "improving": breadth.improving,
            "neutral": breadth.neutral, "weakening": breadth.weakening,
            "weakening_pct": round(breadth.weakening_pct, 4),
            "net_diffusion": round(breadth.net, 4),
        },
        "indices": {
            "leading": _at(lead, as_of),
            "coincident": _at(coin, as_of),
            "lagging": _at(lag, as_of),
        },
        "acceleration": calculate_signal_acceleration(lead, as_of),
        "persistence": calculate_signal_persistence(frames, as_of),
        "divergence": detect_macro_divergence(frames, as_of),
        "turning_point": detect_turning_point(frames, as_of),
        "indicators": [
            {"sid": i.sid, "label": i.label, "signal": round(i.signal, 4),
             "momentum": None if i.momentum is None else round(i.momentum, 4),
             "state": i.state, "date": i.date,
             "percentile": None if i.percentile is None else round(i.percentile, 4)}
            for i in breadth.indicators],
    }
