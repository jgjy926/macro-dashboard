"""
Regime engine (spec 12).

Classifies the economy into one of seven cycle regimes, plus six independent
dimension regimes (growth, inflation, labour, credit, liquidity, policy).

Spec 12's requirement -- "the final regime should not depend on one indicator" --
is enforced structurally, not by convention: classify() takes only the COMPOSITE
score, its momentum and the LEADING INDEX. Each of those is already an aggregate
of many series, so there is no code path by which a single indicator can move the
headline regime on its own.

The three coordinates are chosen because level and momentum together are what
distinguish otherwise-identical readings. A composite of -0.3 falling is
PRE_RECESSION; the same -0.3 rising is EARLY_RECOVERY. A level-only classifier
cannot tell those apart, and they are opposite investment conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import strategy as st
from modules import leading
from modules.factors import FactorScore
from modules.transforms import SignalFrame


@dataclass
class RegimeState:
    regime: str
    strength: float                 # 0..1 confidence in this specific label
    trend: str                      # IMPROVING | STABLE | DETERIORATING
    growth_regime: str
    inflation_regime: str
    labor_regime: str
    credit_regime: str
    liquidity_regime: str
    policy_regime: str
    detail: dict = field(default_factory=dict)

    @property
    def position(self) -> int:
        """Index into REGIME_ORDER, for the dashboard's cycle timeline."""
        return st.REGIME_ORDER.index(self.regime) if self.regime in st.REGIME_ORDER else -1


# ---------------------------------------------------------------------------
# dimension regimes
# ---------------------------------------------------------------------------
def _band(score: float | None, labels: tuple[str, str, str, str, str],
          cuts=(-0.45, -0.15, 0.15, 0.45)) -> str:
    """Map a [-1, +1] score onto five ordered labels."""
    if score is None:
        return "UNKNOWN"
    if score < cuts[0]:
        return labels[0]
    if score < cuts[1]:
        return labels[1]
    if score < cuts[2]:
        return labels[2]
    if score < cuts[3]:
        return labels[3]
    return labels[4]


def classify_growth_regime(factors: dict[str, FactorScore]) -> str:
    """Growth is the average of the three categories that measure activity --
    labour, manufacturing and consumer -- not of everything."""
    parts = [factors[c].score for c in ("labor", "manufacturing", "consumer")
             if c in factors and factors[c].score is not None]
    score = sum(parts) / len(parts) if parts else None
    return _band(score, ("CONTRACTING", "WEAK", "SLUGGISH", "SOLID", "STRONG"))


def classify_labor_regime(factors: dict[str, FactorScore]) -> str:
    return _band(factors.get("labor", FactorScore("labor", None, None, 0, 0)).score,
                 ("DETERIORATING", "SOFTENING", "COOLING", "SOLID", "TIGHT"))


def classify_credit_regime(factors: dict[str, FactorScore]) -> str:
    return _band(factors.get("credit", FactorScore("credit", None, None, 0, 0)).score,
                 ("STRESSED", "TIGHTENING", "NEUTRAL", "ACCOMMODATIVE", "LOOSE"))


def classify_liquidity_regime(factors: dict[str, FactorScore]) -> str:
    """Liquidity blends financial conditions with the policy stance -- the
    availability of money, as distinct from its price."""
    parts = [factors[c].score for c in ("financial", "policy")
             if c in factors and factors[c].score is not None]
    score = sum(parts) / len(parts) if parts else None
    return _band(score, ("SEVERELY_TIGHT", "TIGHT", "NEUTRAL", "AMPLE", "ABUNDANT"))


def classify_policy_regime(factors: dict[str, FactorScore]) -> str:
    detail = factors.get("policy", FactorScore("policy", None, None, 0, 0)).detail
    return detail.get("stance", "NEUTRAL")


def classify_inflation_regime(factors: dict[str, FactorScore]) -> str:
    detail = factors.get("inflation", FactorScore("inflation", None, None, 0, 0)).detail
    return detail.get("regime", "STABLE")


# ---------------------------------------------------------------------------
# cycle regime
# ---------------------------------------------------------------------------
def _matches(rule, level: float, momentum: float) -> bool:
    _, lo_l, hi_l, lo_m, hi_m = rule
    return lo_l <= level < hi_l and lo_m <= momentum < hi_m


def classify_cycle(level: float | None, momentum: float | None,
                   leading_index: float | None) -> tuple[str, float, dict]:
    """Pick a cycle regime from (composite level, momentum, leading index).

    REGIME_RULES overlap deliberately -- the cycle has no crisp boundaries -- so
    when several rules match, the leading index breaks the tie toward the
    forward-looking answer. That is the right bias for a forecasting engine:
    when the level says LATE_CYCLE and the leaders say SLOWDOWN, the leaders are
    the ones with predictive content.
    """
    if level is None:
        return st.REGIME_FALLBACK, 0.0, {"reason": "no composite score"}
    mom = momentum if momentum is not None else 0.0

    matches = [r[0] for r in st.REGIME_RULES if _matches(r, level, mom)]
    if not matches:
        # Outside every rule box: fall back to the nearest rule by centre distance
        # rather than to a hardcoded default, so an extreme reading lands
        # somewhere meaningful instead of on REGIME_FALLBACK.
        def centre_distance(rule):
            _, lo_l, hi_l, lo_m, hi_m = rule
            cl = (max(lo_l, -1.0) + min(hi_l, 1.0)) / 2
            cm = (max(lo_m, -1.0) + min(hi_m, 1.0)) / 2
            return abs(level - cl) + abs(mom - cm)
        best = min(st.REGIME_RULES, key=centre_distance)
        return best[0], 0.35, {"reason": "no exact rule match; nearest rule used",
                               "level": level, "momentum": mom}

    if len(matches) == 1:
        return matches[0], 0.85, {"level": level, "momentum": mom, "candidates": matches}

    # Tie-break toward whichever candidate sits further along the cycle in the
    # direction the leading index points.
    li = leading_index if leading_index is not None else level
    ordered = sorted(matches, key=lambda m: st.REGIME_ORDER.index(m)
                     if m in st.REGIME_ORDER else 99)
    chosen = ordered[-1] if li < level else ordered[0]
    strength = round(max(0.4, 1.0 - 0.18 * (len(matches) - 1)), 3)
    return chosen, strength, {"level": level, "momentum": mom, "candidates": matches,
                              "leading_index": li,
                              "tie_break": "leading index" if len(matches) > 1 else None}


def classify_trend(momentum: float | None) -> str:
    if momentum is None:
        return "UNKNOWN"
    if momentum > st.TREND_EPS:
        return "IMPROVING"
    if momentum < -st.TREND_EPS:
        return "DETERIORATING"
    return "STABLE"


def classify(factors: dict[str, FactorScore], composite: float | None,
             momentum: float | None, frames: dict[str, SignalFrame],
             as_of: str | None = None) -> RegimeState:
    lead_index = leading.calculate_leading_indicator_index(frames)
    li = lead_index.dropna()
    if as_of is not None:
        li = li[li.index <= pd.Timestamp(as_of)]
    li_val = float(li.iloc[-1]) if not li.empty else None

    regime, strength, detail = classify_cycle(composite, momentum, li_val)
    breadth = leading.calculate_signal_breadth(frames, as_of)
    divergence = leading.detect_macro_divergence(frames, as_of)

    # A negative divergence -- leaders rolling over ahead of coincident data --
    # is evidence the current label understates deterioration, so it reduces the
    # strength we claim for it rather than silently changing the label.
    if divergence.get("kind") == "NEGATIVE" and regime in ("EXPANSION", "LATE_CYCLE"):
        strength = round(strength * 0.75, 3)
        detail["strength_note"] = (
            "Confidence in this label reduced: leading indicators are diverging "
            "negatively from coincident data, which historically precedes a "
            "downgrade of the cycle regime.")

    detail.update({
        "breadth_weakening_pct": round(breadth.weakening_pct, 4),
        "breadth": {"improving": breadth.improving, "neutral": breadth.neutral,
                    "weakening": breadth.weakening, "total": breadth.total},
        "divergence": divergence,
        "leading_index": li_val,
    })

    return RegimeState(
        regime=regime, strength=strength, trend=classify_trend(momentum),
        growth_regime=classify_growth_regime(factors),
        inflation_regime=classify_inflation_regime(factors),
        labor_regime=classify_labor_regime(factors),
        credit_regime=classify_credit_regime(factors),
        liquidity_regime=classify_liquidity_regime(factors),
        policy_regime=classify_policy_regime(factors),
        detail=detail)


def describe(state: RegimeState) -> str:
    """One human sentence for the dashboard banner."""
    pretty = state.regime.replace("_", " ").title()
    return (f"{pretty}: growth {state.growth_regime.lower()}, "
            f"labour {state.labor_regime.lower()}, credit {state.credit_regime.lower()}, "
            f"inflation {state.inflation_regime.lower()}, policy "
            f"{state.policy_regime.replace('_', ' ').lower()}. "
            f"Trend {state.trend.lower()}.")


def history(frames: dict[str, SignalFrame], factors_fn, months: int = 60) -> list[dict]:
    """Regime label at each of the last `months` month-ends.

    Used for the dashboard's regime timeline. Note this is a REVISED-DATA
    history: it shows how today's engine reads the past, which is the right
    thing for a timeline. It is NOT what the engine would have said at the time
    -- that question is answered by modules/backtest.py using real vintages, and
    the two must never be conflated.
    """
    lead = leading.calculate_leading_indicator_index(frames).dropna()
    if lead.empty:
        return []
    dates = lead.index[-months:]
    out = []
    for d in dates:
        as_of = d.date().isoformat()
        f = factors_fn(frames, as_of)
        from modules.factors import composite_momentum, composite_score
        score, _ = composite_score(f)
        mom = composite_momentum(f)
        li = lead[lead.index <= d]
        regime, strength, _ = classify_cycle(
            score, mom, float(li.iloc[-1]) if not li.empty else None)
        out.append({"date": as_of, "regime": regime, "composite": score,
                    "momentum": mom, "strength": strength})
    return out
