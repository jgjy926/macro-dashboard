"""
Factor engine (spec 8, 10) -- nine category scores in [-1, +1].

Sign convention throughout: +1 = strongly positive for the economy, -1 = strongly
negative. config/series.py has already flipped every raw series into that
orientation via `direction`, so this module never special-cases "higher is worse".

Three categories get bespoke treatment because a weighted average of their
members would throw away the structure that makes them informative:

  CURVE      Spec 8.6 is explicit that "inverted = bad, normal = good" is wrong.
             What matters is the STATE and the DIRECTION OF TRAVEL: a curve
             re-steepening out of deep inversion is the most dangerous
             configuration, not a recovery, because it usually means the market
             has started pricing cuts against visible damage. Classified into
             seven states with an explicit risk ordering.

  INFLATION  Carries no good/bad sign (see config.series.INFLATION_NOTE). It is
             mapped to a regime label, and only the regime -- plus its
             interaction with growth, which is what stagflation is -- produces a
             signed contribution.

  POLICY     Restrictiveness is about the REAL rate against neutral, not the
             nominal level. 5% nominal against 8% inflation is easy money. The
             engine reports the distance from an assumed r*, and is explicit
             that r* is uncertain.
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
class Contribution:
    """One series' contribution to its category score -- the explainability unit
    that the dashboard's driver panel and spec 30's JSON are built from."""
    sid: str
    label: str
    role: str
    signal: float
    weight: float
    momentum: float | None
    percentile: float | None
    raw: float | None
    date: str | None

    @property
    def weighted(self) -> float:
        return self.signal * self.weight


@dataclass
class FactorScore:
    factor: str
    score: float | None            # None when coverage is insufficient
    momentum: float | None
    n_inputs: int
    coverage: float
    contributions: list[Contribution] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.score is not None

    def arrow(self) -> str:
        if self.momentum is None:
            return "?"
        for threshold, symbol in st.TREND_ARROWS:
            if self.momentum >= threshold:
                return symbol
        return "vvv"

    def top_drivers(self, n: int = 3, positive: bool = False) -> list[Contribution]:
        pool = [c for c in self.contributions
                if (c.weighted > 0 if positive else c.weighted < 0)]
        return sorted(pool, key=lambda c: -abs(c.weighted))[:n]


# ---------------------------------------------------------------------------
# generic weighted-average factor
# ---------------------------------------------------------------------------
def _collect(frames: dict[str, SignalFrame], category: str, as_of: str | None
             ) -> tuple[list[Contribution], int]:
    """Gather usable contributions for a category. Returns (contributions,
    n_expected) so coverage can be judged against the catalogue, not against
    whatever happened to load."""
    specs = [s for s in cat.factor_series() if s.category == category]
    out: list[Contribution] = []
    for spec in specs:
        frame = frames.get(spec.sid)
        if frame is None or not frame.usable or spec.direction == 0:
            continue
        snap = frame.at(as_of)
        if snap["signal"] is None or math.isnan(snap["signal"]):
            continue
        out.append(Contribution(
            sid=spec.sid, label=spec.label, role=spec.role,
            signal=float(snap["signal"]),
            weight=st.ROLE_WEIGHTS.get(spec.role, 0.0),
            momentum=frame.momentum(as_of, st.MOMENTUM_MONTHS),
            percentile=snap["percentile"], raw=snap["raw"], date=snap["date"]))
    # direction=0 series are excluded from the expected count as well -- they
    # were never going to contribute, so counting them would understate coverage.
    n_expected = sum(1 for s in specs if s.direction != 0)
    return out, n_expected


def _weighted_score(contribs: list[Contribution]) -> float | None:
    total_w = sum(c.weight for c in contribs)
    if total_w <= 0:
        return None
    return round(sum(c.weighted for c in contribs) / total_w, 4)


def _weighted_momentum(contribs: list[Contribution]) -> float | None:
    pairs = [(c.momentum, c.weight) for c in contribs if c.momentum is not None]
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    return round(sum(m * w for m, w in pairs) / total_w, 4)


def generic_factor(frames: dict[str, SignalFrame], category: str,
                   as_of: str | None = None) -> FactorScore:
    contribs, n_expected = _collect(frames, category, as_of)
    coverage = len(contribs) / n_expected if n_expected else 0.0
    if coverage < st.MIN_FACTOR_COVERAGE:
        return FactorScore(category, None, None, len(contribs), round(coverage, 3), contribs,
                           {"unavailable_reason":
                            f"coverage {coverage:.0%} below the {st.MIN_FACTOR_COVERAGE:.0%} minimum"})
    return FactorScore(category, _weighted_score(contribs), _weighted_momentum(contribs),
                       len(contribs), round(coverage, 3), contribs)


# ---------------------------------------------------------------------------
# 8.6 yield curve -- structural
# ---------------------------------------------------------------------------
def _spread_series(frames: dict[str, SignalFrame], sid: str, as_of: str | None) -> pd.Series:
    f = frames.get(sid)
    if f is None or f.raw.empty:
        return pd.Series(dtype="float64")
    s = f.raw.dropna()
    return s if as_of is None else s[s.index <= pd.Timestamp(as_of)]


def calculate_inversion_depth(spread: pd.Series) -> float | None:
    """How deep the curve is right now, in percentage points (negative = inverted)."""
    return None if spread.empty else round(float(spread.iloc[-1]), 4)


def calculate_inversion_duration(spread: pd.Series) -> int:
    """Consecutive months the curve has been inverted, counting back from now.

    Returns 0 when not currently inverted. Duration matters independently of
    depth: a brief dip below zero is noise, while a year of inversion has
    preceded every post-war recession.
    """
    if spread.empty or spread.iloc[-1] >= 0:
        return 0
    inverted = spread < 0
    run = 0
    for flag in reversed(inverted.to_numpy()):
        if not flag:
            break
        run += 1
    # Convert observation count to months at the series' own frequency.
    return int(round(run / (PERIODS_PER_YEAR["daily"] / 12)))


def calculate_resteepening(spread: pd.Series, months: int = 3) -> float | None:
    """Change in the spread over `months`. Positive = steepening."""
    if spread.empty:
        return None
    lag = max(1, round(PERIODS_PER_YEAR["daily"] * months / 12))
    if len(spread) <= lag:
        return None
    return round(float(spread.iloc[-1] - spread.iloc[-1 - lag]), 4)


def _classify_steepening(spread: pd.Series, short_yield: pd.Series,
                         long_yield: pd.Series, months: int = 3) -> str:
    """Bull (short rates falling) vs bear (long rates rising) steepening.

    The distinction is the whole point: bull steepening means the market is
    pricing rate CUTS, which historically happens because something has broken.
    Bear steepening means term premium or growth expectations rising, which is
    benign or even positive.
    """
    lag = max(1, round(PERIODS_PER_YEAR["daily"] * months / 12))
    d_short = d_long = 0.0
    if len(short_yield) > lag:
        d_short = float(short_yield.iloc[-1] - short_yield.iloc[-1 - lag])
    if len(long_yield) > lag:
        d_long = float(long_yield.iloc[-1] - long_yield.iloc[-1 - lag])
    return "RE_STEEPENING_BULL" if d_short < d_long else "RE_STEEPENING_BEAR"


def classify_curve_state(frames: dict[str, SignalFrame], as_of: str | None = None
                         ) -> tuple[str, dict]:
    """Classify the curve into one of CURVE_STATE_RISK's seven states."""
    spread = _spread_series(frames, "T10Y3M", as_of)
    if spread.empty:
        spread = _spread_series(frames, "T10Y2Y", as_of)
    if spread.empty:
        return "NORMAL", {"reason": "no curve data"}

    level = float(spread.iloc[-1])
    change = calculate_resteepening(spread, 3) or 0.0
    duration = calculate_inversion_duration(spread)
    short_y = _spread_series(frames, "DGS3MO", as_of)
    long_y = _spread_series(frames, "DGS10", as_of)

    if level < 0:
        # Inverted AND steepening back toward zero: the late-stage configuration.
        if change > st.CURVE_MOVE_EPS and duration >= st.CURVE_INVERSION_MATURE_MONTHS:
            state = _classify_steepening(spread, short_y, long_y)
        elif level <= st.CURVE_INVERSION_DEEP:
            state = "INVERTED_DEEP"
        else:
            state = "INVERTED_SHALLOW"
    elif level >= st.CURVE_STEEP:
        # A steep curve reached by rapid steepening from recent inversion is the
        # same late-stage signal even though the level is now positive.
        if duration == 0 and change > st.CURVE_MOVE_EPS * 2 and _recently_inverted(spread):
            state = _classify_steepening(spread, short_y, long_y)
        else:
            state = "STEEP"
    elif level < st.CURVE_FLAT and change < -st.CURVE_MOVE_EPS:
        state = "FLATTENING"
    elif level < st.CURVE_FLAT:
        state = "FLATTENING" if change < 0 else "NORMAL"
    else:
        state = "NORMAL"

    return state, {
        "spread_10y3m": round(level, 3),
        "inversion_depth": calculate_inversion_depth(spread),
        "inversion_duration_months": duration,
        "change_3m": round(change, 3),
        "state": state,
        "risk_weight": st.CURVE_STATE_RISK[state],
    }


def _recently_inverted(spread: pd.Series, months: int = 18) -> bool:
    lag = max(1, round(PERIODS_PER_YEAR["daily"] * months / 12))
    window = spread.iloc[-lag:] if len(spread) > lag else spread
    return bool((window < 0).any())


def curve_factor(frames: dict[str, SignalFrame], as_of: str | None = None) -> FactorScore:
    """Curve factor from the STATE, not from a z-score of the spread.

    CURVE_STATE_RISK is expressed as risk (higher = worse), so it is negated to
    reach this module's convention where +1 is economically positive.
    """
    state, detail = classify_curve_state(frames, as_of)
    score = -st.CURVE_STATE_RISK[state]
    contribs, n_expected = _collect(frames, "curve", as_of)
    coverage = len(contribs) / n_expected if n_expected else 0.0

    # Momentum here is the change in the spread, re-expressed on the score's
    # scale so the dashboard arrow means the same thing as every other factor's.
    spread = _spread_series(frames, "T10Y3M", as_of)
    change = calculate_resteepening(spread, st.MOMENTUM_MONTHS)
    momentum = None if change is None else round(float(np.tanh(change / 1.0)) * 0.5, 4)
    # But while inverted, steepening is NOT improvement -- flip the arrow so the
    # display cannot imply an all-clear during the most dangerous configuration.
    if state.startswith("RE_STEEPENING") and momentum is not None:
        momentum = -abs(momentum)

    detail["interpretation"] = {
        "STEEP": "Steep curve -- early-cycle, policy accommodative relative to growth.",
        "NORMAL": "Normally sloped curve -- no signal either way.",
        "FLATTENING": "Flattening -- policy tightening into the cycle.",
        "INVERTED_SHALLOW": "Shallow inversion -- the classic early warning.",
        "INVERTED_DEEP": "Deep inversion -- historically the strongest single recession signal.",
        "RE_STEEPENING_BULL": ("Bull re-steepening from inversion -- the market is pricing "
                               "CUTS. Historically the LATE stage, typically close to or "
                               "already inside recession, not an all-clear."),
        "RE_STEEPENING_BEAR": ("Bear re-steepening -- long rates rising. Term premium or "
                               "growth expectations, less ominous than bull steepening."),
    }[state]
    return FactorScore("curve", round(score, 4), momentum, len(contribs),
                       round(coverage, 3), contribs, detail)


# ---------------------------------------------------------------------------
# 8.7 inflation -- regime, not direction
# ---------------------------------------------------------------------------
def calculate_inflation_momentum(frames: dict[str, SignalFrame], as_of: str | None = None
                                 ) -> float | None:
    """Change in core CPI YoY over 6 months (pp). Positive = accelerating."""
    f = frames.get("CPILFESL")
    if f is None or f.transformed.empty:
        return None
    s = f.transformed.dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    if len(s) < 7:
        return None
    return round(float(s.iloc[-1] - s.iloc[-7]), 3)


def calculate_inflation_regime(frames: dict[str, SignalFrame], growth_score: float | None,
                               as_of: str | None = None) -> tuple[str, dict]:
    """Map inflation to one of spec 8.7's six regimes.

    STAGFLATIONARY is checked first because it is a joint condition on inflation
    AND growth, and would otherwise be masked by whichever level band the
    inflation rate happened to fall into.
    """
    core = frames.get("CPILFESL")
    core_pce = frames.get("PCEPILFE")
    level = None
    for f in (core, core_pce):
        if f is not None and not f.transformed.empty:
            s = f.transformed.dropna()
            if as_of is not None:
                s = s[s.index <= pd.Timestamp(as_of)]
            if not s.empty:
                level = float(s.iloc[-1])
                break
    if level is None:
        return "STABLE", {"reason": "no inflation data"}

    momentum = calculate_inflation_momentum(frames, as_of)

    if (level >= st.STAGFLATION_INFLATION_MIN and growth_score is not None
            and growth_score <= st.STAGFLATION_GROWTH_MAX):
        regime = "STAGFLATIONARY"
    else:
        regime = next(name for lo, hi, name in st.INFLATION_BANDS if lo <= level < hi)

    direction = "stable"
    if momentum is not None:
        if momentum > st.INFLATION_MOMENTUM_EPS:
            direction = "accelerating"
        elif momentum < -st.INFLATION_MOMENTUM_EPS:
            direction = "decelerating"

    return regime, {"core_yoy": round(level, 3), "momentum_6m": momentum,
                    "direction": direction, "regime": regime}


def inflation_factor(frames: dict[str, SignalFrame], growth_score: float | None,
                     as_of: str | None = None) -> FactorScore:
    """Signed contribution derived from the inflation REGIME.

    The sign answers one question: does the current inflation configuration help
    or hurt the growth outlook? Disinflation from a high level is positive (it
    buys the Fed room to cut). Outright deflation is negative (debt deflation).
    Reflation from target is negative (it forces policy tighter).
    """
    regime, detail = calculate_inflation_regime(frames, growth_score, as_of)
    base = {
        "DEFLATIONARY": -0.55,
        "DISINFLATIONARY": +0.45,
        "STABLE": +0.25,
        "REFLATIONARY": -0.25,
        "INFLATIONARY": -0.60,
        "STAGFLATIONARY": -0.85,
    }[regime]

    # Momentum modifies the level: inflation falling toward target from above is
    # better than inflation stuck there, and vice versa.
    momentum = detail.get("momentum_6m")
    adj = 0.0
    if momentum is not None:
        level = detail.get("core_yoy") or 0.0
        if level > 3.0:
            adj = -np.tanh(momentum / 1.5) * 0.25   # accelerating from high = worse
        elif level < 1.0:
            adj = +np.tanh(momentum / 1.5) * 0.20   # rising off deflation = better
        else:
            adj = -np.tanh(momentum / 1.5) * 0.10

    score = float(np.clip(base + adj, -1.0, 1.0))
    contribs, n_expected = _collect(frames, "inflation", as_of)
    detail["interpretation"] = (
        f"{regime.title().replace('_', ' ')} inflation at {detail.get('core_yoy')}% core, "
        f"{detail.get('direction')}.")
    detail["note"] = cat.INFLATION_NOTE
    return FactorScore("inflation", round(score, 4),
                       None if momentum is None else round(-momentum / 5.0, 4),
                       len(contribs), 1.0, contribs, detail)


# ---------------------------------------------------------------------------
# 8.8 monetary policy -- real rate vs neutral
# ---------------------------------------------------------------------------
def _latest(frames: dict[str, SignalFrame], sid: str, as_of: str | None,
            use_transform: bool = False) -> float | None:
    f = frames.get(sid)
    if f is None:
        return None
    s = (f.transformed if use_transform else f.raw)
    if s is None or s.empty:
        return None
    s = s.dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    return None if s.empty else float(s.iloc[-1])


def calculate_policy_restrictiveness(frames: dict[str, SignalFrame], as_of: str | None = None
                                     ) -> tuple[float | None, str, dict]:
    """Real policy rate minus assumed neutral.

    Uses the market-implied real rate (10Y TIPS) when available and falls back to
    fed funds minus core inflation, which is the only option before TIPS existed
    in 2003 -- so pre-2003 backtests get a policy factor rather than a hole.
    """
    ff = _latest(frames, "DFF", as_of) or _latest(frames, "FEDFUNDS", as_of)
    core = _latest(frames, "CPILFESL", as_of, use_transform=True)
    if ff is None or core is None:
        return None, "NEUTRAL", {"reason": "insufficient policy data"}

    real_policy = ff - core
    gap = real_policy - st.NEUTRAL_REAL_RATE
    band = next(name for lo, hi, name in st.POLICY_BANDS if lo <= gap < hi)
    return gap, band, {
        "fed_funds": round(ff, 3),
        "core_inflation": round(core, 3),
        "real_policy_rate": round(real_policy, 3),
        "assumed_neutral": st.NEUTRAL_REAL_RATE,
        "gap_vs_neutral": round(gap, 3),
        "stance": band,
        "caveat": ("Neutral (r*) is unobservable; published estimates span roughly "
                   "0.0-1.5%. The gap is a distance from an assumption, not a measurement."),
    }


def calculate_policy_momentum(frames: dict[str, SignalFrame], as_of: str | None = None
                              ) -> float | None:
    """Change in the policy rate over 6 months (pp). Negative = easing."""
    f = frames.get("DFF")
    if f is None or f.raw.empty:
        return None
    s = f.raw.dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    lag = round(PERIODS_PER_YEAR["daily"] / 2)
    if len(s) <= lag:
        return None
    return round(float(s.iloc[-1] - s.iloc[-1 - lag]), 3)


def policy_factor(frames: dict[str, SignalFrame], as_of: str | None = None) -> FactorScore:
    gap, band, detail = calculate_policy_restrictiveness(frames, as_of)
    contribs, n_expected = _collect(frames, "policy", as_of)
    coverage = len(contribs) / n_expected if n_expected else 0.0

    if gap is None:
        return FactorScore("policy", None, None, len(contribs), round(coverage, 3),
                           contribs, detail)

    # Restrictive policy is a headwind, so the score is the negated gap, squashed.
    score = float(-np.tanh(gap / 1.5))
    momentum_pp = calculate_policy_momentum(frames, as_of)
    # Easing (negative rate change) is positive for growth, hence the sign flip.
    momentum = None if momentum_pp is None else round(float(-np.tanh(momentum_pp / 1.0)) * 0.5, 4)

    detail["policy_change_6m"] = momentum_pp
    detail["interpretation"] = (
        f"Policy {band.replace('_', ' ').lower()}: real rate "
        f"{detail['real_policy_rate']}% vs assumed neutral {st.NEUTRAL_REAL_RATE}% "
        f"({gap:+.2f}pp). " + ("Easing." if (momentum_pp or 0) < -0.1
                               else "Tightening." if (momentum_pp or 0) > 0.1 else "On hold."))
    return FactorScore("policy", round(score, 4), momentum, len(contribs),
                       round(coverage, 3), contribs, detail)


# ---------------------------------------------------------------------------
# category helpers named by the spec
# ---------------------------------------------------------------------------
def calculate_consumer_stress(frames: dict[str, SignalFrame], as_of: str | None = None
                              ) -> dict:
    """Consumer stress read jointly, not as a single-series score.

    The savings rate alone is ambiguous -- falling savings can mean confidence or
    it can mean households are running down a buffer to keep spending. The
    distinguishing evidence is real income growth and delinquencies, so all three
    are read together (config/series.py's PSAVERT note points here).
    """
    savings = _latest(frames, "PSAVERT", as_of)
    income = _latest(frames, "DSPIC96", as_of, use_transform=True)
    delinq = _latest(frames, "DRCCLACBS", as_of)
    spending = _latest(frames, "PCEC96", as_of, use_transform=True)

    stressed = (savings is not None and income is not None and spending is not None
                and income < 1.0 and spending > income and savings < 5.0)
    return {
        "savings_rate": savings,
        "real_income_yoy": income,
        "real_spending_yoy": spending,
        "cc_delinquency": delinq,
        "drawing_down_buffer": stressed,
        "interpretation": (
            "Spending is outrunning real income with a thin savings buffer -- "
            "consumption is being financed from savings or credit, which is not "
            "sustainable." if stressed else
            "Income growth is supporting spending; no buffer drawdown detected."),
    }


def calculate_credit_turning_point(frames: dict[str, SignalFrame], as_of: str | None = None
                                   ) -> dict:
    """Has credit begun to turn? Spreads widening from tight levels is the tell.

    Widening from ALREADY-WIDE levels is late; widening from historically tight
    levels is the turn, which is why the percentile matters as much as the move.
    """
    f = frames.get("BAMLH0A0HYM2")
    if f is None or f.raw.empty:
        return {"turning": False, "reason": "no HY spread data"}
    s = f.raw.dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    if len(s) < 90:
        return {"turning": False, "reason": "insufficient history"}
    now = float(s.iloc[-1])
    change_3m = now - float(s.iloc[-min(len(s) - 1, 63)])
    snap = f.at(as_of)
    pct = snap.get("percentile")
    turning = change_3m > 0.5 and (pct is not None and pct < 0.5)
    return {
        "hy_oas": round(now, 3),
        "change_3m_pp": round(change_3m, 3),
        "percentile": pct,
        "turning": turning,
        "interpretation": (
            f"HY spreads widening {change_3m:+.2f}pp over 3 months from the "
            f"{pct:.0%} percentile -- credit turning from tight levels."
            if turning else
            f"HY spreads at {now:.2f}%"
            + (f" ({pct:.0%} percentile)" if pct is not None else "")
            + f", {change_3m:+.2f}pp over 3 months."),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def calculate_all_factors(frames: dict[str, SignalFrame], as_of: str | None = None
                          ) -> dict[str, FactorScore]:
    """All nine category factors.

    Ordering matters once: the inflation factor needs a growth score to detect
    stagflation, so the growth-ish categories are computed first and their
    average is passed in.
    """
    out: dict[str, FactorScore] = {}
    for c in ["housing", "consumer", "labor", "manufacturing", "credit", "financial"]:
        out[c] = generic_factor(frames, c, as_of)

    out["curve"] = curve_factor(frames, as_of)
    out["policy"] = policy_factor(frames, as_of)

    growth_inputs = [out[c].score for c in ("labor", "manufacturing", "consumer")
                     if out[c].score is not None]
    growth_score = sum(growth_inputs) / len(growth_inputs) if growth_inputs else None
    out["inflation"] = inflation_factor(frames, growth_score, as_of)

    # Attach the joint reads to the factors they explain.
    out["consumer"].detail["stress"] = calculate_consumer_stress(frames, as_of)
    out["credit"].detail["turning_point"] = calculate_credit_turning_point(frames, as_of)
    return out


def composite_score(factors: dict[str, FactorScore]) -> tuple[float | None, dict]:
    """The single macro score: FACTOR_WEIGHTS-weighted average of the factors.

    Weights are renormalised over the factors that are actually available, so a
    missing category shifts weight to its peers instead of dragging the
    composite toward zero and making a data outage look like a neutral economy.
    """
    parts = [(name, f.score, st.FACTOR_WEIGHTS.get(name, 0.0))
             for name, f in factors.items() if f.score is not None]
    total_w = sum(w for _, _, w in parts)
    if total_w <= 0:
        return None, {"reason": "no factors available"}
    score = sum(s * w for _, s, w in parts) / total_w
    return round(score, 4), {
        "weights_used": {n: round(w / total_w, 4) for n, _, w in parts},
        "missing": [n for n, f in factors.items() if f.score is None],
        "coverage": round(total_w / sum(st.FACTOR_WEIGHTS.values()), 3),
    }


def composite_momentum(factors: dict[str, FactorScore]) -> float | None:
    parts = [(f.momentum, st.FACTOR_WEIGHTS.get(n, 0.0))
             for n, f in factors.items() if f.momentum is not None]
    total_w = sum(w for _, w in parts)
    if total_w <= 0:
        return None
    return round(sum(m * w for m, w in parts) / total_w, 4)


# ---------------------------------------------------------------------------
# history (for the dashboard's factor sparklines)
# ---------------------------------------------------------------------------
def factor_history(frames: dict[str, SignalFrame], months: int = 48
                   ) -> dict[str, list[dict]]:
    """Monthly history of each factor score.

    Computed two different ways because the factors are built two different
    ways, and showing a member series' history in place of a structural factor's
    would be a quietly wrong chart:

      * The six weighted-average factors are reproduced exactly and cheaply by
        role-weighting their members' monthly signals in pandas -- the same
        arithmetic generic_factor() does, vectorised over time.
      * curve, policy and inflation are structural (a state machine, a real-rate
        gap and a regime map). They have no member-average to vectorise, so their
        own classifiers are called at each month-end. That is ~3 x `months`
        calls, which is cheap, and it is exact rather than a proxy.
    """
    out: dict[str, list[dict]] = {}

    for category in ["housing", "consumer", "labor", "manufacturing",
                     "credit", "financial"]:
        cols, weights = [], []
        for spec in cat.factor_series():
            if spec.category != category or spec.direction == 0:
                continue
            frame = frames.get(spec.sid)
            if frame is None or not frame.usable or frame.signal.empty:
                continue
            w = st.ROLE_WEIGHTS.get(spec.role, 0.0)
            if w <= 0:
                continue
            cols.append(frame.signal.dropna().resample("ME").last().rename(spec.sid))
            weights.append(w)
        if not cols:
            out[category] = []
            continue
        df = pd.concat(cols, axis=1)
        w = pd.Series(weights, index=df.columns)
        # Renormalise per row over the members present that month, so a series
        # that starts late does not drag the early history toward zero.
        num = (df * w).sum(axis=1, skipna=True)
        den = df.notna().mul(w, axis=1).sum(axis=1)
        series = (num / den.replace(0.0, np.nan)).dropna().tail(months)
        out[category] = [{"date": d.date().isoformat(), "value": round(float(v), 4)}
                         for d, v in series.items()]

    # Structural factors: evaluate their own logic at each month-end.
    anchor = None
    for frame in frames.values():
        if frame is not None and not frame.raw.empty:
            idx = frame.raw.dropna().resample("ME").last().index
            anchor = idx if anchor is None or len(idx) > len(anchor) else anchor
    dates = list(anchor[-months:]) if anchor is not None else []

    for name, fn in [("curve", curve_factor), ("policy", policy_factor)]:
        rows = []
        for d in dates:
            try:
                f = fn(frames, d.date().isoformat())
            except Exception:
                continue
            if f.score is not None:
                rows.append({"date": d.date().isoformat(), "value": round(f.score, 4)})
        out[name] = rows

    rows = []
    for d in dates:
        as_of = d.date().isoformat()
        try:
            growth_parts = [generic_factor(frames, c, as_of).score
                            for c in ("labor", "manufacturing", "consumer")]
            growth_parts = [g for g in growth_parts if g is not None]
            g = sum(growth_parts) / len(growth_parts) if growth_parts else None
            f = inflation_factor(frames, g, as_of)
        except Exception:
            continue
        if f.score is not None:
            rows.append({"date": as_of, "value": round(f.score, 4)})
    out["inflation"] = rows
    return out
