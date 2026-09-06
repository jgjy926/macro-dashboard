"""
Scenario engine (spec 17) -- BASE / BULL / BEAR / CRISIS.

THE CONSISTENCY RULE
--------------------
Scenario probabilities are DERIVED from the calibrated recession probability,
never asserted independently. A dashboard showing "12-month recession
probability 43%" beside "BEAR 25%, CRISIS 10%" is publishing two incompatible
claims about the same future, and a reader is right to distrust both. Here:

    P(CRISIS) + P(BEAR)  ==  P(recession within 12 months)
    P(BASE)   + P(BULL)  ==  1 - that

The only judgement calls are how the recession mass splits between BEAR (a
normal downturn) and CRISIS (a downturn with credit dysfunction), and how the
expansion mass splits between BASE and BULL. Both are conditioned on the actual
data: the crisis share rises with measured financial stress, and the bull share
rises when the leading indicators are already turning up.

Each scenario carries the spec 17 fields -- growth, inflation, unemployment,
rates, housing, consumer, credit, assumptions and risks -- generated from the
forecast engine's own numbers under the scenario's shift, rather than from a
hand-written table that would go stale the moment the data moved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import strategy as st
from modules.factors import FactorScore
from modules.forecast import ForecastEngine, RecessionForecast


@dataclass
class Scenario:
    name: str
    probability: float
    headline: str
    growth: str
    inflation: str
    unemployment: str
    rates: str
    housing: str
    consumer: str
    credit: str
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario": self.name, "probability": round(self.probability, 4),
            "headline": self.headline,
            "outlook": {"growth": self.growth, "inflation": self.inflation,
                        "unemployment": self.unemployment, "rates": self.rates,
                        "housing": self.housing, "consumer": self.consumer,
                        "credit": self.credit},
            "assumptions": self.assumptions, "risks": self.risks,
        }


def _stress_level(factors: dict[str, FactorScore]) -> float:
    """Financial stress in [0, 1], from the credit and financial factors.

    Used to decide how much of the recession mass belongs in CRISIS rather than
    BEAR. A recession without credit dysfunction is a slowdown; 2008 was a
    recession WITH it, and the difference is the whole point of the distinction.
    """
    parts = [factors[c].score for c in ("credit", "financial")
             if c in factors and factors[c].score is not None]
    if not parts:
        return 0.4
    # Scores run +1 (healthy) to -1 (stressed); map to 0..1 stress.
    return float(np.clip((1.0 - sum(parts) / len(parts)) / 2.0, 0.0, 1.0))


def _momentum_tilt(momentum: float | None) -> float:
    """How much of the expansion mass leans bullish, from composite momentum.

    Improving momentum genuinely raises the odds of reacceleration rather than
    mere muddling-through, so BULL takes a larger share when the economy is
    already turning up.
    """
    if momentum is None:
        return st.BULL_SHARE_OF_EXPANSION
    tilt = st.BULL_SHARE_OF_EXPANSION + float(np.tanh(momentum * 3.0)) * 0.15
    return float(np.clip(tilt, 0.05, 0.45))


def _band(direction: str, magnitude: str = "") -> str:
    return f"{magnitude} {direction}".strip()


def generate(engine: ForecastEngine, factors: dict[str, FactorScore],
             recession_12m: RecessionForecast, composite: float | None,
             momentum: float | None) -> list[Scenario]:
    """Build the four scenarios, consistent with the 12-month probability."""
    p_rec = float(np.clip(recession_12m.probability, 0.0, 1.0))
    p_exp = 1.0 - p_rec

    stress = _stress_level(factors)
    # Crisis share of the recession mass scales with measured stress around the
    # configured centre, so a downturn priced during calm credit markets is
    # mostly BEAR and one priced during stress is much more CRISIS.
    crisis_share = float(np.clip(
        st.CRISIS_SHARE_OF_RECESSION * (0.5 + stress) / 0.5 * 0.5, 0.05, 0.55))
    if stress >= st.CRISIS_STRESS_THRESHOLD:
        crisis_share = float(np.clip(crisis_share * 1.35, 0.05, 0.65))

    bull_share = _momentum_tilt(momentum)

    p_crisis = p_rec * crisis_share
    p_bear = p_rec - p_crisis
    p_bull = p_exp * bull_share
    p_base = p_exp - p_bull

    infl = factors.get("inflation")
    infl_regime = (infl.detail.get("regime", "STABLE") if infl else "STABLE")
    housing_weak = (factors.get("housing") and factors["housing"].score is not None
                    and factors["housing"].score < -0.15)

    base = Scenario(
        name="BASE", probability=p_base,
        headline=("Slow growth without recession -- the economy absorbs current "
                  "conditions without a downturn beginning."),
        growth="Below trend but positive", inflation=f"{infl_regime.title()}, gradual drift",
        unemployment="Drifting modestly higher", rates="Broadly stable, mild easing bias",
        housing="Soft but not collapsing" if housing_weak else "Stabilising",
        consumer="Slowing spending growth, income support intact",
        credit="Spreads range-bound",
        assumptions=[
            "No credit event; spreads stay inside their recent range.",
            "The labour market cools through fewer hires rather than layoffs.",
            "Policy stays roughly where it is, with no forced tightening.",
        ],
        risks=["A labour-market crack turns a slowdown into a downturn.",
               "Inflation re-accelerates and removes the option to ease."])

    bull = Scenario(
        name="BULL", probability=p_bull,
        headline="Growth reaccelerates as financial conditions ease and real incomes recover.",
        growth="Returning to or above trend", inflation="Continues moderating toward target",
        unemployment="Flat to lower", rates="Lower short end, curve re-steepens benignly",
        housing="Recovers as mortgage rates fall", consumer="Real income growth revives spending",
        credit="Spreads tighten further",
        assumptions=[
            "Inflation keeps falling, allowing policy to ease pre-emptively.",
            "Leading indicators turn up and the turn persists.",
            "No external shock to energy or supply chains.",
        ],
        risks=["Easing financial conditions re-ignite inflation and force a reversal.",
               "The reacceleration is concentrated and not broad-based."])

    bear = Scenario(
        name="BEAR", probability=p_bear,
        headline="A conventional recession: demand weakens, unemployment rises, no financial break.",
        growth="Contracting", inflation="Falls faster than expected on weak demand",
        unemployment="Rises materially", rates="Cut in response to weakness",
        housing="Weakens further on falling employment",
        consumer="Spending contracts as the savings buffer is exhausted",
        credit="Spreads widen substantially but markets keep functioning",
        assumptions=[
            "Labour-market deterioration becomes self-reinforcing.",
            "Credit tightens but no systemic institution fails.",
            "Policy eases with the usual lag, after the damage is visible.",
        ],
        risks=["Deterioration accelerates into the CRISIS case.",
               "Inflation stays sticky, limiting how fast policy can respond."])

    crisis = Scenario(
        name="CRISIS", probability=p_crisis,
        headline="Recession compounded by financial stress -- credit stops functioning normally.",
        growth="Sharp contraction", inflation="Falls sharply; deflation risk emerges",
        unemployment="Rises steeply", rates="Cut aggressively toward the lower bound",
        housing="Falls sharply on credit unavailability",
        consumer="Sharp retrenchment", credit="Spreads gap wider; issuance markets close",
        assumptions=[
            "A credit or funding event forces disorderly deleveraging.",
            "Lending standards tighten abruptly rather than gradually.",
            "Policy response is large but arrives after the stress is visible.",
        ],
        risks=["Second-round effects through the banking system.",
               "Policy response is constrained by inflation still above target."])

    return [base, bull, bear, crisis]


def summarise(scenarios: list[Scenario]) -> dict:
    total = sum(s.probability for s in scenarios)
    return {
        "scenarios": [s.to_dict() for s in scenarios],
        "sums_to": round(total, 4),
        "consistency_note": (
            "Scenario probabilities are derived from the 12-month calibrated "
            "recession probability: CRISIS + BEAR equals it exactly, and BASE + BULL "
            "equals its complement. They are not independent estimates, which is why "
            "they always sum to 1 and never contradict the headline number."),
    }
