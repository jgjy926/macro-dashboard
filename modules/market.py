"""
Market regime + asset-class implications (spec 19, 20).

THE SEPARATION SPEC 19 DEMANDS
------------------------------
Spec 19 is unusually blunt about this:

    Do NOT:  Recession Probability = Macro Risk + CAPE

Valuation must never enter the recession probability. Expensive equities do not
make a recession more likely; they make the CONSEQUENCES of one worse. Those are
different claims and mixing them corrupts both.

The dependency here is strictly one-way, and enforced by the module structure:
this module IMPORTS from the macro engine and nothing in the macro engine
imports from here. There is no code path by which CAPE, equity prices or risk
appetite can reach a factor score or a recession probability.

    MACRO ENGINE  -->  MARKET REGIME ENGINE  -->  ASSET IMPLICATIONS
    (growth, inflation, labour, credit, recession)
                       (+ valuation, earnings, liquidity, risk appetite)

WHY THERE IS NO CAPE HERE
-------------------------
Shiller CAPE is not available from any keyless source the engine already uses,
and scraping a third-party page for it would put a fragile dependency on the
critical path for a number that moves slowly. The valuation component is instead
built from the equity market's own long-run z-score and the equity risk premium
implied against real yields -- both computed from series already in the
catalogue. `valuation_note` says exactly this on the dashboard rather than
letting a reader assume CAPE is in there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import strategy as st
from modules.factors import FactorScore
from modules.transforms import SignalFrame


@dataclass
class MarketRegime:
    score: float                 # -1 (hostile) .. +1 (supportive)
    label: str
    components: dict[str, float] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)
    valuation_note: str = ""


@dataclass
class AssetImplication:
    asset: str
    score: float
    stance: str
    rationale: str


def _latest(frames: dict[str, SignalFrame], sid: str, as_of: str | None = None,
            transformed: bool = False) -> float | None:
    f = frames.get(sid)
    if f is None:
        return None
    s = (f.transformed if transformed else f.raw)
    if s is None or s.empty:
        return None
    s = s.dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    return None if s.empty else float(s.iloc[-1])


def _change(frames: dict[str, SignalFrame], sid: str, as_of: str | None = None,
            months: int = 6) -> float | None:
    """Change in a series' raw level over `months`, at its own frequency."""
    from modules.transforms import PERIODS_PER_YEAR
    from config import series as cat
    f = frames.get(sid)
    if f is None or f.raw.empty:
        return None
    s = f.raw.dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    spec = cat.BY_ID.get(sid)
    lag = max(1, round(PERIODS_PER_YEAR.get(spec.freq if spec else "daily", 252)
                       * months / 12))
    if len(s) <= lag:
        return None
    return float(s.iloc[-1] - s.iloc[-1 - lag])


def _percentile(frames: dict[str, SignalFrame], sid: str, as_of: str | None = None
                ) -> float | None:
    f = frames.get(sid)
    if f is None or not f.usable:
        return None
    return f.at(as_of).get("percentile")


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------
def valuation_component(frames: dict[str, SignalFrame], as_of: str | None = None
                        ) -> tuple[float, str]:
    """Equity valuation proxy. Negative score = expensive.

    Built from the equity market's 12-month log change percentile (a momentum-
    stretch proxy) and the real 10-year yield, because a given equity level is
    more demanding when the risk-free real rate is high. Not CAPE -- see the
    module docstring.
    """
    eq_pct = _percentile(frames, "NASDAQCOM", as_of) or _percentile(frames, "SP500", as_of)
    real_yield = _latest(frames, "DFII10", as_of)

    if eq_pct is None:
        return 0.0, "no equity series available for a valuation proxy"

    # High percentile of 12m gains = stretched = negative for forward returns.
    stretch = -(eq_pct - 0.5) * 2.0
    rate_drag = 0.0
    if real_yield is not None:
        # A real yield above ~2% raises the bar equities must clear.
        rate_drag = -float(np.clip((real_yield - 1.0) / 2.0, -0.5, 0.6))
    score = float(np.clip(0.6 * stretch + 0.4 * rate_drag, -1.0, 1.0))
    return round(score, 4), (
        f"equity 12m change at the {eq_pct:.0%} percentile"
        + (f", real 10Y yield {real_yield:.2f}%" if real_yield is not None else ""))


def earnings_component(factors: dict[str, FactorScore]) -> tuple[float, str]:
    """Earnings proxy from the real economy.

    No free, keyless S&P earnings series exists, so nominal activity -- the
    manufacturing and consumer factors -- stands in for the revenue line that
    drives earnings. Explicitly a proxy.
    """
    parts = [factors[c].score for c in ("manufacturing", "consumer")
             if c in factors and factors[c].score is not None]
    if not parts:
        return 0.0, "no activity factors available"
    score = sum(parts) / len(parts)
    return round(float(score), 4), (
        f"proxied by manufacturing and consumer activity ({score:+.2f}); no free "
        f"keyless S&P earnings series exists")


def liquidity_component(factors: dict[str, FactorScore],
                        frames: dict[str, SignalFrame], as_of: str | None = None
                        ) -> tuple[float, str]:
    nfci = _latest(frames, "NFCI", as_of)
    parts = [factors[c].score for c in ("financial", "policy")
             if c in factors and factors[c].score is not None]
    score = sum(parts) / len(parts) if parts else 0.0
    return round(float(score), 4), (
        f"financial conditions and policy stance"
        + (f"; NFCI {nfci:+.2f} ({'tight' if nfci > 0 else 'loose'})" if nfci is not None else ""))


def credit_component(factors: dict[str, FactorScore]) -> tuple[float, str]:
    f = factors.get("credit")
    if f is None or f.score is None:
        return 0.0, "credit factor unavailable"
    tp = f.detail.get("turning_point", {})
    return round(f.score, 4), tp.get("interpretation", "credit factor")


def rates_component(frames: dict[str, SignalFrame], as_of: str | None = None
                    ) -> tuple[float, str]:
    """Direction of travel in long rates. Falling rates support most assets."""
    f = frames.get("DGS10")
    if f is None or f.raw.empty:
        return 0.0, "no 10Y yield data"
    s = f.raw.dropna()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    if len(s) < 130:
        return 0.0, "insufficient yield history"
    change_6m = float(s.iloc[-1] - s.iloc[-126])
    score = float(np.clip(-change_6m / 1.0, -1.0, 1.0))
    return round(score, 4), (
        f"10Y yield {'down' if change_6m < 0 else 'up'} {abs(change_6m):.2f}pp over 6 months")


def risk_appetite_component(frames: dict[str, SignalFrame], as_of: str | None = None
                            ) -> tuple[float, str]:
    vix = _latest(frames, "VIXCLS", as_of)
    hy_pct = _percentile(frames, "BAMLH0A0HYM2", as_of)
    parts, notes = [], []
    if vix is not None:
        parts.append(float(np.clip((22.0 - vix) / 12.0, -1.0, 1.0)))
        notes.append(f"VIX {vix:.1f}")
    if hy_pct is not None:
        # Low HY-spread percentile = tight spreads = strong risk appetite.
        parts.append(float(np.clip(1.0 - hy_pct * 2.0, -1.0, 1.0)))
        notes.append(f"HY spread at the {hy_pct:.0%} percentile")
    if not parts:
        return 0.0, "no risk-appetite series available"
    return round(sum(parts) / len(parts), 4), ", ".join(notes)


# ---------------------------------------------------------------------------
# market regime
# ---------------------------------------------------------------------------
def calculate_market_regime(factors: dict[str, FactorScore],
                            frames: dict[str, SignalFrame],
                            as_of: str | None = None) -> MarketRegime:
    comps, notes = {}, {}
    for key, (value, why) in {
        "valuation": valuation_component(frames, as_of),
        "earnings": earnings_component(factors),
        "liquidity": liquidity_component(factors, frames, as_of),
        "credit": credit_component(factors),
        "rates": rates_component(frames, as_of),
        "risk_appetite": risk_appetite_component(frames, as_of),
    }.items():
        comps[key] = value
        notes[key] = why

    total_w = sum(st.MARKET_WEIGHTS.values())
    score = sum(comps[k] * w for k, w in st.MARKET_WEIGHTS.items()) / total_w
    label = next((lab for thr, lab in st.ASSET_BANDS if score >= thr), "NEGATIVE")
    return MarketRegime(
        score=round(float(score), 4), label=label, components=comps, explanations=notes,
        valuation_note=(
            "Valuation is a proxy built from the equity market's own 12-month change "
            "percentile and the real 10-year yield. Shiller CAPE is NOT included: no "
            "keyless free source carries it, and the engine will not put a scraped "
            "third-party page on the critical path. CAPE would belong here, in the "
            "market layer -- never in the recession probability (spec 19)."))


# ---------------------------------------------------------------------------
# asset implications
# ---------------------------------------------------------------------------
def calculate_asset_implications(factors: dict[str, FactorScore],
                                 market: MarketRegime,
                                 frames: dict[str, SignalFrame],
                                 recession_prob: float,
                                 as_of: str | None = None) -> list[AssetImplication]:
    """Directional stance per asset class from four macro drivers.

    An implication layer, not a return forecast (spec 20). The sensitivities in
    strategy.ASSET_SENSITIVITY encode textbook relationships with relative
    magnitudes; they are not estimated betas and the output is a stance, never
    a target.
    """
    growth_parts = [factors[c].score for c in ("labor", "manufacturing", "consumer")
                    if c in factors and factors[c].score is not None]
    growth = sum(growth_parts) / len(growth_parts) if growth_parts else 0.0

    infl = factors.get("inflation")
    infl_regime = infl.detail.get("regime", "STABLE") if infl else "STABLE"
    # Inflation PRESSURE (+1 = rising prices), distinct from the inflation
    # factor's growth-friendliness score, because assets respond to the
    # pressure, not to whether it is convenient for growth.
    inflation = {"DEFLATIONARY": -0.8, "DISINFLATIONARY": -0.4, "STABLE": 0.0,
                 "REFLATIONARY": 0.5, "INFLATIONARY": 0.9,
                 "STAGFLATIONARY": 0.8}.get(infl_regime, 0.0)

    credit_score = factors["credit"].score if factors.get("credit") and \
        factors["credit"].score is not None else 0.0
    # Stress runs 0 (calm) .. 1 (severe), lifted by the recession probability so
    # a high probability tightens every stance even before spreads move.
    stress = float(np.clip((1.0 - credit_score) / 2.0 * 0.7 + recession_prob * 0.6, 0.0, 1.0))

    # ASSET_SENSITIVITY's real_rate coefficients describe how an asset responds
    # to real rates MOVING, not to their level -- "bonds fall when real yields
    # rise". Feeding the level here would score a high-but-stable real rate as a
    # persistent headwind, which inverts the conclusion: high real yields make
    # bonds cheap, it is RISING real yields that hurt them. So the driver is the
    # 6-month change, scaled so a 1pp move is a full-strength signal.
    real_rate_change = _change(frames, "DFII10", as_of, months=6)
    if real_rate_change is None:
        real_rate_change = _change(frames, "DGS10", as_of, months=6)
    real_rate_drv = 0.0 if real_rate_change is None else         float(np.clip(real_rate_change / 1.0, -1.5, 1.5))

    drivers = {"growth": growth, "inflation": inflation,
               "stress": stress * 2 - 1, "real_rate": real_rate_drv}

    out: list[AssetImplication] = []
    for asset in st.ASSET_CLASSES:
        sens = st.ASSET_SENSITIVITY[asset]
        score = sum(sens[k] * drivers[k] for k in sens) / sum(abs(v) for v in sens.values())
        # Valuation is an equity-specific drag, applied only where it belongs.
        if asset in ("US Equities", "REITs"):
            score = score * 0.8 + market.components.get("valuation", 0.0) * 0.2
        score = float(np.clip(score, -1.0, 1.0))
        stance = next((lab for thr, lab in st.ASSET_BANDS if score >= thr), "NEGATIVE")

        top = sorted(sens.items(), key=lambda kv: -abs(kv[1] * drivers[kv[0]]))[:2]
        names = {"real_rate": "real-rate change (6m)", "stress": "credit stress",
                 "growth": "growth", "inflation": "inflation pressure"}
        rationale = "; ".join(
            f"{names.get(k, k)} {drivers[k]:+.2f} (sensitivity {v:+.1f})" for k, v in top)
        out.append(AssetImplication(asset, round(score, 4), stance, rationale))
    return out


def summarise(factors: dict[str, FactorScore], frames: dict[str, SignalFrame],
              recession_prob: float, as_of: str | None = None) -> dict:
    regime = calculate_market_regime(factors, frames, as_of)
    assets = calculate_asset_implications(factors, regime, frames, recession_prob, as_of)
    return {
        "regime": {
            "score": regime.score, "label": regime.label,
            "components": regime.components, "explanations": regime.explanations,
            "valuation_note": regime.valuation_note,
        },
        "assets": [{"asset": a.asset, "score": a.score, "stance": a.stance,
                    "rationale": a.rationale} for a in assets],
        "separation_note": (
            "This layer CONSUMES the macro engine's output and never feeds back into "
            "it. Valuation, earnings and risk appetite affect the asset stances only; "
            "they cannot move the recession probability (spec 19)."),
        "disclaimer": ("Directional implications, not investment advice and not a return "
                       "forecast. Relationships are regime-dependent and can invert."),
    }
