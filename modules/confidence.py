"""
Confidence engine (spec 18).

Probability and confidence are different quantities, and conflating them is the
most common failure of forecasting dashboards. Spec 18 makes the point with two
examples that must both be expressible:

    Recession probability: 61%   Confidence: LOW
    Recession probability: 61%   Confidence: HIGH

The first says "the indicators point here but they disagree, the data is stale,
and the model has never seen a configuration like this". The second says "every
factor agrees, every model agrees, the data is clean, and this looks like
episodes the model has scored well on". Same number, opposite actionability.

Confidence here is a weighted average of six components (strategy.CONFIDENCE_WEIGHTS),
each in [0, 1], each reported individually so the dashboard can say WHY confidence
is low rather than only that it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import strategy as st
from modules.factors import FactorScore
from modules.forecast import RecessionForecast
from modules.transforms import SignalFrame


@dataclass
class ConfidenceReport:
    score: float
    band: str
    components: dict[str, float] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)

    def weakest(self, n: int = 2) -> list[tuple[str, float]]:
        return sorted(self.components.items(), key=lambda kv: kv[1])[:n]

    def summary(self) -> str:
        weak = self.weakest(2)
        return (f"Confidence {self.score:.0%} ({self.band}). "
                f"Most limited by {weak[0][0].replace('_', ' ')} "
                f"({weak[0][1]:.0%})" + (f" and {weak[1][0].replace('_', ' ')} "
                                         f"({weak[1][1]:.0%})" if len(weak) > 1 else "") + ".")


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------
def indicator_agreement(factors: dict[str, FactorScore]) -> tuple[float, str]:
    """Do the factors point the same way?

    Measured as 1 minus the normalised dispersion of the factor scores. Eight
    factors all at -0.5 is a coherent picture; four at -0.9 and four at +0.4
    averages to the same composite while meaning something completely different,
    and only the dispersion distinguishes them.
    """
    scores = [f.score for f in factors.values() if f.score is not None]
    if len(scores) < 3:
        return 0.3, "too few factors available to judge agreement"
    sd = float(np.std(scores))
    # Max plausible dispersion across [-1, +1] scores is ~1.0; scale to [0, 1].
    agreement = float(np.clip(1.0 - sd / 0.7, 0.0, 1.0))
    same_sign = sum(1 for s in scores if s < 0) / len(scores)
    consensus = max(same_sign, 1 - same_sign)
    combined = 0.6 * agreement + 0.4 * consensus
    return round(combined, 4), (
        f"{consensus:.0%} of factors share a sign, dispersion {sd:.2f} "
        f"across {len(scores)} factors")


def model_agreement(wf_results: dict, horizon_m: int) -> tuple[float, str]:
    """Do the models agree with each other on today's reading?

    Uses the spread of their walk-forward AUCs as a proxy for structural
    agreement: when one model discriminates well and another is near chance,
    they are describing different relationships and the published number is
    sensitive to which one was selected.
    """
    aucs = [r.auc for r in wf_results.values() if getattr(r, "auc", None) is not None]
    if len(aucs) < 2:
        return 0.4, "fewer than two models produced comparable scores"
    spread = max(aucs) - min(aucs)
    agreement = float(np.clip(1.0 - spread / 0.30, 0.0, 1.0))
    return round(agreement, 4), (
        f"walk-forward AUC ranges {min(aucs):.2f}-{max(aucs):.2f} across "
        f"{len(aucs)} models")


def historical_accuracy(forecast: RecessionForecast) -> tuple[float, str]:
    """The selected model's own out-of-sample discrimination at this horizon.

    Rescaled so AUC 0.5 (chance) maps to 0 and 1.0 maps to 1 -- an AUC of 0.5 is
    worth no confidence at all, but on a raw scale it would still contribute 0.5.
    """
    if forecast.auc is None:
        return 0.35, "no walk-forward AUC available for the selected model"
    score = float(np.clip((forecast.auc - 0.5) * 2, 0.0, 1.0))
    return round(score, 4), (
        f"{forecast.model} walk-forward AUC {forecast.auc:.3f} at {forecast.horizon_m}m")


def data_quality_component(health: dict) -> tuple[float, str]:
    score = float(health.get("health", 0.0))
    return round(score, 4), (
        f"{health.get('high', 0)} HIGH / {health.get('medium', 0)} MEDIUM / "
        f"{health.get('low', 0)} LOW quality series, {health.get('stale', 0)} stale")


def data_freshness_component(health: dict, quality: dict) -> tuple[float, str]:
    """How current the inputs are, weighted toward the model's own features.

    A stale peripheral series barely matters; a stale feature the model
    depends on matters a great deal, so MODEL_FEATURES are counted twice.
    """
    if not quality:
        return 0.0, "no quality assessments"
    total = weighted = 0.0
    for sid, q in quality.items():
        w = 2.0 if sid in st.MODEL_FEATURES else 1.0
        total += w
        weighted += w * (0.0 if q.is_stale else 1.0)
    score = weighted / total if total else 0.0
    stale_features = [s for s in st.MODEL_FEATURES
                      if s in quality and quality[s].is_stale]
    return round(score, 4), (
        f"{health.get('stale', 0)} of {health.get('n', 0)} series stale"
        + (f"; model features stale: {', '.join(stale_features)}" if stale_features
           else "; no model features stale"))


def regime_similarity(frames: dict[str, SignalFrame], X: pd.DataFrame) -> tuple[float, str]:
    """Is today like anything the model was trained on?

    Nearest-neighbour distance from today's feature vector to its historical
    counterparts. If today is unprecedented, the model is extrapolating, and
    extrapolation from 8 recessions deserves low confidence regardless of how
    clean the data is. This is the component that would have flagged early 2020.
    """
    if X.empty or len(X) < 60:
        return 0.3, "insufficient history to judge similarity"
    arr = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy(dtype="float64")
    today, history = arr[-1], arr[:-1]
    dists = np.sqrt(np.sum((history - today) ** 2, axis=1))
    nearest = float(np.percentile(dists, 1))     # 1st percentile ~ the closest analogues
    typical = float(np.median(dists))
    if typical == 0:
        return 0.5, "degenerate feature distances"
    ratio = nearest / typical
    score = float(np.clip(1.0 - ratio, 0.0, 1.0))
    return round(score, 4), (
        f"closest historical analogue is at {nearest:.2f} versus a typical "
        f"{typical:.2f} distance -- "
        + ("today closely resembles past episodes" if ratio < 0.4
           else "today is only loosely similar to anything in the record" if ratio < 0.75
           else "today has no close historical analogue; the model is extrapolating"))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def calculate_confidence(factors: dict[str, FactorScore], forecast: RecessionForecast,
                         wf_results: dict, health: dict, quality: dict,
                         frames: dict[str, SignalFrame], X: pd.DataFrame
                         ) -> ConfidenceReport:
    comps: dict[str, float] = {}
    notes: dict[str, str] = {}

    for key, (value, why) in {
        "indicator_agreement": indicator_agreement(factors),
        "model_agreement": model_agreement(wf_results, forecast.horizon_m),
        "historical_accuracy": historical_accuracy(forecast),
        "data_quality": data_quality_component(health),
        "data_freshness": data_freshness_component(health, quality),
        "regime_similarity": regime_similarity(frames, X),
    }.items():
        comps[key] = value
        notes[key] = why

    total_w = sum(st.CONFIDENCE_WEIGHTS.values())
    score = sum(comps[k] * w for k, w in st.CONFIDENCE_WEIGHTS.items()) / total_w
    band = next((label for threshold, label in st.CONFIDENCE_BANDS if score >= threshold), "LOW")
    return ConfidenceReport(round(float(score), 4), band, comps, notes)
