"""
Recommendation and explainability engine (spec 21, 30).

Two outputs from one analysis:

  A MACHINE-READABLE EXPLANATION (spec 30) -- the JSON blob with regime,
  probability, confidence, drivers and offsetting factors, which the dashboard
  and any downstream system consume.

  INVALIDATION CONDITIONS (spec 21) -- what would change the view. This is the
  part most forecast products omit, and it is the part that makes a forecast
  falsifiable rather than merely confident. The conditions here are DERIVED from
  the current drivers, not written from a template: if housing is the biggest
  negative contributor, "housing permits stabilise" appears in the improve list;
  if housing is already fine, it does not, and something that actually matters
  takes its place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import series as cat
from config import strategy as st
from modules.confidence import ConfidenceReport
from modules.factors import Contribution, FactorScore
from modules.forecast import RecessionForecast
from modules.regime import RegimeState

# What "this improves" and "this worsens" look like per series, in the reader's
# language rather than the model's. Only series that can plausibly appear as a
# top driver need an entry; anything missing falls back to a generic phrasing
# built from the label, so the catalogue can grow without touching this table.
INVALIDATION_PHRASES: dict[str, tuple[str, str]] = {
    "PERMIT": ("building permits stabilise or turn up", "building permits fall further"),
    "HOUST": ("housing starts stabilise", "housing starts keep falling"),
    "HSN1F": ("new home sales recover", "new home sales weaken further"),
    "MSACSR": ("new-home inventory clears", "months-supply of homes rises further"),
    "MORTGAGE30US": ("mortgage rates fall", "mortgage rates rise further"),
    "ICSA": ("initial jobless claims decline", "initial claims rise materially"),
    "CCSA": ("continuing claims roll over", "continuing claims keep climbing"),
    "TEMPHELPS": ("temporary employment stabilises", "temp employment keeps shrinking"),
    "AWHMAN": ("factory hours recover", "factory hours are cut further"),
    "UNRATE": ("unemployment stops rising", "unemployment rises materially"),
    "PAYEMS": ("payroll growth re-accelerates", "payroll growth stalls"),
    "JTSJOL": ("job openings stabilise", "job openings fall further"),
    "BAMLH0A0HYM2": ("high-yield spreads tighten", "high-yield spreads widen"),
    "BAMLC0A0CM": ("investment-grade spreads tighten", "investment-grade spreads widen"),
    "BAMLH0A3HYC": ("CCC spreads tighten", "the low-quality credit tail gaps wider"),
    "DRTSCILM": ("banks stop tightening lending standards", "banks tighten lending standards further"),
    "NFCI": ("financial conditions ease", "financial conditions tighten further"),
    "ANFCI": ("financial conditions ease relative to the cycle", "financial conditions tighten further"),
    "STLFSI4": ("financial stress recedes", "financial stress rises"),
    "T10Y3M": ("the 10Y-3M curve normalises without a bull steepening",
               "the curve inverts further or bull-steepens sharply"),
    "T10Y2Y": ("the 10Y-2Y curve normalises", "the 10Y-2Y curve inverts further"),
    "NEWORDER": ("capital goods orders recover", "capital goods orders keep contracting"),
    "ISRATIO": ("the inventory/sales ratio falls back", "inventories keep building against sales"),
    "UMCSENT": ("consumer sentiment recovers", "consumer sentiment deteriorates further"),
    "ALTSALES": ("vehicle sales stabilise", "vehicle sales keep falling"),
    "RRSFS": ("real retail sales re-accelerate", "real retail sales contract"),
    "DSPIC96": ("real income growth holds up", "real income growth stalls"),
    "PCEC96": ("consumption growth holds up", "consumer spending contracts"),
    "VIXCLS": ("volatility subsides", "volatility spikes"),
    "NASDAQCOM": ("equities recover", "equities fall materially"),
    "INDPRO": ("industrial production stabilises", "industrial production contracts"),
    "TCU": ("capacity utilisation stabilises", "capacity utilisation falls further"),
}


@dataclass
class Recommendation:
    regime: str
    trend: str
    recession_probability_12m: float
    confidence: float
    confidence_band: str
    drivers: list[dict] = field(default_factory=list)
    offsetting: list[dict] = field(default_factory=list)
    improves_if: list[str] = field(default_factory=list)
    worsens_if: list[str] = field(default_factory=list)
    narrative: str = ""
    caveats: list[str] = field(default_factory=list)


def _phrase(c: Contribution, worse: bool) -> str:
    pair = INVALIDATION_PHRASES.get(c.sid)
    if pair:
        return pair[1] if worse else pair[0]
    label = c.label.lower()
    return f"{label} deteriorates further" if worse else f"{label} stabilises"


def _rank_contributions(factors: dict[str, FactorScore], negative: bool
                        ) -> list[Contribution]:
    """Rank every series across every factor by its weighted contribution to the
    composite -- not within its own factor.

    The factor weight has to be included: a strongly negative reading inside a
    3%-weighted category is a smaller driver of the headline than a mildly
    negative one inside an 18%-weighted category, and ranking within factors
    would hide that.
    """
    scored: list[tuple[float, Contribution]] = []
    for name, f in factors.items():
        fw = st.FACTOR_WEIGHTS.get(name, 0.0)
        for c in f.contributions:
            impact = c.weighted * fw
            if (impact < 0) if negative else (impact > 0):
                scored.append((abs(impact), c))
    scored.sort(key=lambda kv: -kv[0])
    # One entry per series -- the same reading should not occupy two slots.
    seen, out = set(), []
    for _, c in scored:
        if c.sid in seen:
            continue
        seen.add(c.sid)
        out.append(c)
    return out


def _driver_dict(c: Contribution, factors: dict[str, FactorScore]) -> dict:
    spec = cat.BY_ID.get(c.sid)
    category = spec.category if spec else ""
    return {
        "sid": c.sid, "label": c.label, "category": category, "role": c.role,
        "signal": round(c.signal, 4),
        "momentum": None if c.momentum is None else round(c.momentum, 4),
        "percentile": None if c.percentile is None else round(c.percentile, 4),
        "raw": c.raw, "date": c.date,
        "impact": round(c.weighted * st.FACTOR_WEIGHTS.get(category, 0.0), 5),
    }


def build(regime: RegimeState, factors: dict[str, FactorScore],
          recession: RecessionForecast, confidence: ConfidenceReport,
          composite: float | None, health: dict) -> Recommendation:
    negatives = _rank_contributions(factors, negative=True)
    positives = _rank_contributions(factors, negative=False)

    drivers = [_driver_dict(c, factors) for c in negatives[:st.TOP_DRIVERS]]
    offsets = [_driver_dict(c, factors) for c in positives[:st.TOP_OFFSETS]]

    # Improvement conditions come from what is currently WRONG; deterioration
    # conditions from what is currently holding up. That asymmetry is the point:
    # the view changes when the binding constraints change.
    improves = [_phrase(c, worse=False) for c in negatives[:st.TOP_DRIVERS]]
    worsens = [_phrase(c, worse=True) for c in positives[:st.TOP_OFFSETS]]
    # Add the biggest negatives' further deterioration too -- a bad thing getting
    # worse is as much a trigger as a good thing turning.
    for c in negatives[:2]:
        p = _phrase(c, worse=True)
        if p not in worsens:
            worsens.append(p)

    rec = Recommendation(
        regime=regime.regime, trend=regime.trend,
        recession_probability_12m=recession.probability,
        confidence=confidence.score, confidence_band=confidence.band,
        drivers=drivers, offsetting=offsets,
        improves_if=improves[:st.TOP_DRIVERS], worsens_if=worsens[:st.TOP_DRIVERS])

    rec.narrative = narrative(regime, recession, confidence, composite, drivers, offsets)
    rec.caveats = caveats(recession, confidence, health)
    return rec


def narrative(regime: RegimeState, recession: RecessionForecast,
              confidence: ConfidenceReport, composite: float | None,
              drivers: list[dict], offsets: list[dict]) -> str:
    """Spec 30: 'The dashboard should convert this into human-readable language.'"""
    pretty = regime.regime.replace("_", " ").lower()
    trend_word = {"IMPROVING": "improving", "DETERIORATING": "deteriorating",
                  "STABLE": "broadly stable", "UNKNOWN": "unclear"}[regime.trend]
    parts = [
        f"The economy reads as {pretty} with a {trend_word} trend "
        f"(composite {composite:+.2f})." if composite is not None else
        f"The economy reads as {pretty} with a {trend_word} trend.",
        f"The {recession.horizon_m}-month recession probability is "
        f"{recession.probability:.0%}, against an unconditional historical base rate of "
        f"{recession.base_rate:.0%}.",
    ]
    if drivers:
        names = ", ".join(d["label"].lower() for d in drivers[:3])
        parts.append(f"The main drags are {names}.")
    if offsets:
        names = ", ".join(d["label"].lower() for d in offsets[:2])
        parts.append(f"Offsetting this, {names} remain supportive.")
    parts.append(confidence.summary())
    return " ".join(parts)


def caveats(recession: RecessionForecast, confidence: ConfidenceReport,
            health: dict) -> list[str]:
    """Honest limitations, surfaced rather than buried."""
    out = []
    if not recession.calibrated:
        out.append("The probability is UNCALIBRATED -- treat it as a ranking, not a frequency.")
    if confidence.band == "LOW":
        weak = confidence.weakest(1)[0]
        out.append(f"Confidence is LOW, most limited by {weak[0].replace('_', ' ')}.")
    if health.get("stale", 0):
        out.append(f"{health['stale']} series are stale: "
                   f"{', '.join(health.get('stale_series', [])[:5])}.")
    if health.get("critical", 0):
        out.append(f"{health['critical']} series have critical data issues.")
    if recession.signal_weight < 0.7:
        out.append(
            f"At {recession.horizon_m} months the forecast is only "
            f"{recession.signal_weight:.0%} signal and "
            f"{1 - recession.signal_weight:.0%} historical base rate -- no indicator has "
            f"demonstrated reliable precision at this horizon.")
    out.append("NBER recession dates are announced with a lag of up to a year, so the "
               "most recent months of the historical record are provisional.")
    return out


def to_explanation(rec: Recommendation, regime: RegimeState,
                   horizons: list[RecessionForecast]) -> dict:
    """The spec 30 machine-readable explanation, extended to all horizons."""
    return {
        "regime": rec.regime,
        "trend": rec.trend,
        "recession_probability_12m": rec.recession_probability_12m,
        "recession_probability": {f"{f.horizon_m}m": f.probability for f in horizons},
        "raw_macro_risk_score": {f"{f.horizon_m}m": f.raw_score for f in horizons},
        "confidence": rec.confidence,
        "confidence_band": rec.confidence_band,
        "dimensions": {
            "growth": regime.growth_regime, "inflation": regime.inflation_regime,
            "labor": regime.labor_regime, "credit": regime.credit_regime,
            "liquidity": regime.liquidity_regime, "policy": regime.policy_regime,
        },
        "drivers": [d["label"] + f" ({d['signal']:+.2f})" for d in rec.drivers],
        "offsetting_factors": [d["label"] + f" ({d['signal']:+.2f})" for d in rec.offsetting],
        "improves_if": rec.improves_if,
        "worsens_if": rec.worsens_if,
        "narrative": rec.narrative,
        "caveats": rec.caveats,
    }
