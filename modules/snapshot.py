"""
Snapshot builder -- the one place a full engine run is assembled.

Everything downstream (the text report, the dashboard JSON feed, the alert
evaluation, the database write) consumes the SAME snapshot dict produced here.
That is deliberate: if the report computed its own numbers it would eventually
disagree with the dashboard, and there would be no way to tell which was right.
One computation, many renderings.

The snapshot is plain JSON-serialisable data -- no pandas objects, no numpy
scalars -- so it can be written to disk, hashed for reproducibility, and read by
a static frontend with no Python on the other end.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from config import series as cat
from config import settings
from config import strategy as st
from modules import (alerts, backtest as backtest_engine, confidence,
                     factors as factor_engine, forecast, ingestion, leading, market,
                     quality, recommend, regime as regime_engine, runlog, scenarios,
                     transforms, vintage)
from modules.log import log_info

FACTOR_LABELS = {
    "housing": "Housing", "consumer": "Consumer", "labor": "Labor",
    "manufacturing": "Manufacturing", "credit": "Credit", "curve": "Yield Curve",
    "inflation": "Inflation", "policy": "Policy", "financial": "Financial",
}


def _clean(obj):
    """Make a structure JSON-safe: numpy scalars -> python, NaN/Inf -> None.

    NaN is the important case. json.dumps emits a bare `NaN` token, which is not
    valid JSON and which every browser's JSON.parse rejects -- so a single NaN
    anywhere would break the entire dashboard feed rather than blanking one cell.
    """
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    return obj


def _series_history(frame: transforms.SignalFrame, months: int = 48) -> list[dict]:
    """Monthly history of one series, for the dashboard's sparklines."""
    if frame is None or frame.raw.empty:
        return []
    raw = frame.raw.dropna().resample("ME").last().tail(months)
    sig = (frame.signal.dropna().resample("ME").last().reindex(raw.index)
           if not frame.signal.empty else pd.Series(index=raw.index, dtype="float64"))
    out = []
    for d, v in raw.items():
        s = sig.get(d)
        out.append({"date": d.date().isoformat(), "value": _clean(v),
                    "signal": _clean(s) if s is not None and not pd.isna(s) else None})
    return out


def build(conn, *, retrain: bool = False, as_of: str | None = None,
          include_history: bool = True, run_type: str = "daily"
          ) -> tuple[dict, dict]:
    """Run the full engine.

    Returns (snapshot, artifacts). The snapshot is plain JSON-safe data for
    rendering; artifacts holds the live Python objects (frames, factor scores,
    regime state, forecasts) that persist() needs to write to the database.
    Returning both keeps the expensive computation to exactly one pass -- the
    alternative, recomputing inside persist(), would double the run time and
    open the door to the two disagreeing.
    """
    t_start = datetime.now(timezone.utc)
    as_of = as_of or date.today().isoformat()

    # -- data ---------------------------------------------------------------
    rows = {s.sid: quality.load_rows(conn, s.sid) for s in cat.ALL_SERIES}
    frames = transforms.build_all(rows)
    q = quality.assess_all(conn)
    health = quality.health_summary(q)

    # -- factors, breadth, regime -------------------------------------------
    F = factor_engine.calculate_all_factors(frames)
    composite, comp_meta = factor_engine.composite_score(F)
    momentum = factor_engine.composite_momentum(F)
    lead = leading.summarise(frames)
    R = regime_engine.classify(F, composite, momentum, frames)

    # -- forecasts -----------------------------------------------------------
    engine = forecast.ForecastEngine(frames, F, rows.get("USREC", []), retrain=retrain)
    horizons = engine.all_horizons()
    engine.save_cache()
    f12 = next((h for h in horizons if h.horizon_m == 12), horizons[-1])

    conf = confidence.calculate_confidence(
        F, f12, engine._wf.get(12, {}), health, q, frames, engine.X)

    scen = scenarios.generate(engine, F, f12, composite, momentum)
    mkt = market.summarise(F, frames, f12.probability)
    rec = recommend.build(R, F, f12, conf, composite, health)

    variables = engine.all_variables() if run_type != "quick" else {}
    _bt = backtest_engine.latest_backtest_summary(conn)

    # -- assemble -------------------------------------------------------------
    snap = {
        "meta": {
            "as_of": as_of,
            "as_of_display": date.fromisoformat(as_of).strftime("%d %b %Y").upper(),
            "generated_at": t_start.isoformat(timespec="seconds"),
            "model_version": settings.MODEL_VERSION,
            "feature_version": settings.FEATURE_VERSION,
            "run_type": run_type,
            "retrained": engine.retrained,
            "series_count": len(cat.ALL_SERIES),
            "data_note": ("All data from free, keyless public sources (FRED, ALFRED, BLS). "
                          "Freshness reflects the DATA date, not the fetch date."),
            "disclaimer": ("Decision-support for monitoring purposes. Not investment "
                           "advice and not a guarantee. Macro relationships are "
                           "regime-dependent and can and do invert."),
        },
        "headline": {
            "regime": R.regime,
            "regime_display": R.regime.replace("_", " ").title(),
            "regime_position": R.position,
            "regime_order": st.REGIME_ORDER,
            "regime_strength": R.strength,
            "trend": R.trend,
            "composite": composite,
            "composite_momentum": momentum,
            "recession_12m": f12.probability,
            "recession_12m_pct": f"{f12.probability:.1%}",
            "recession_band": f12.band(),
            "confidence": conf.score,
            "confidence_pct": f"{conf.score:.0%}",
            "confidence_band": conf.band,
            "summary": regime_engine.describe(R),
        },
        "recession": {
            "horizons": [{
                "horizon_m": h.horizon_m,
                "probability": h.probability,
                "raw_score": h.raw_score,
                "model": h.model,
                "calibrated": h.calibrated,
                "base_rate": h.base_rate,
                "signal_weight": h.signal_weight,
                "auc": h.auc, "brier": h.brier,
                "band": h.band(), "note": h.note,
            } for h in horizons],
            "calibrated_all": all(h.calibrated for h in horizons),
            "score_vs_probability_note": (
                "RAW SCORE is the model's uncalibrated output; CALIBRATED is that value "
                "blended toward the horizon's historical base rate and mapped through a "
                "calibrator fitted on out-of-sample predictions. They are different "
                "objects and only the calibrated one should be read as a frequency."),
            "drivers": f12.drivers,
        },
        "factors": [{
            "factor": name,
            "label": FACTOR_LABELS.get(name, name.title()),
            "score": f.score,
            "momentum": f.momentum,
            "arrow": f.arrow(),
            "coverage": f.coverage,
            "n_inputs": f.n_inputs,
            "available": f.available,
            "interpretation": f.detail.get("interpretation", ""),
            "detail": _clean(f.detail),
            "top_negative": [{"sid": c.sid, "label": c.label, "signal": c.signal}
                             for c in f.top_drivers(3)],
            "top_positive": [{"sid": c.sid, "label": c.label, "signal": c.signal}
                             for c in f.top_drivers(3, positive=True)],
            "contributions": [{
                "sid": c.sid, "label": c.label, "role": c.role, "signal": c.signal,
                "momentum": c.momentum, "percentile": c.percentile,
                "raw": c.raw, "date": c.date,
            } for c in f.contributions],
        } for name, f in F.items()],
        "composite_meta": comp_meta,
        "leading": lead,
        "regime": {
            "regime": R.regime, "strength": R.strength, "trend": R.trend,
            "dimensions": {
                "growth": R.growth_regime, "inflation": R.inflation_regime,
                "labor": R.labor_regime, "credit": R.credit_regime,
                "liquidity": R.liquidity_regime, "policy": R.policy_regime,
            },
            "detail": _clean(R.detail),
        },
        "confidence": {
            "score": conf.score, "band": conf.band,
            "components": conf.components, "explanations": conf.explanations,
            "summary": conf.summary(),
            "note": ("Confidence is NOT probability. It answers how much to trust the "
                     "probability, from indicator and model agreement, historical "
                     "accuracy, data quality and freshness, and how similar today is to "
                     "anything in the training record."),
        },
        "scenarios": scenarios.summarise(scen),
        "market": mkt,
        "recommendation": {
            "regime": rec.regime, "trend": rec.trend,
            "recession_probability_12m": rec.recession_probability_12m,
            "confidence": rec.confidence, "confidence_band": rec.confidence_band,
            "drivers": rec.drivers, "offsetting": rec.offsetting,
            "improves_if": rec.improves_if, "worsens_if": rec.worsens_if,
            "narrative": rec.narrative, "caveats": rec.caveats,
        },
        "explanation": recommend.to_explanation(rec, R, horizons),
        "variables": {
            name: [{
                "horizon_m": v.horizon_m, "direction": v.direction, "point": v.point,
                "lo": v.lo, "hi": v.hi, "unit": v.unit, "confidence": v.confidence,
                "rmse": v.rmse, "label": v.label, "drivers": v.drivers, "note": v.note,
            } for v in vs] for name, vs in variables.items()
        },
        "health": health,
        "quality": [{
            "sid": s.sid, "label": cat.BY_ID[s.sid].label,
            "category": cat.BY_ID[s.sid].category,
            "frequency": cat.BY_ID[s.sid].freq,
            "grade": s.grade, "score": s.score,
            "last_observation": s.last_observation, "last_release": s.last_release,
            "observation_age": s.observation_age, "missing_pct": s.missing_pct,
            "is_stale": s.is_stale, "revision_status": s.revision_status,
            "issues": s.issues, "vintage_policy": cat.BY_ID[s.sid].vintage,
            "note": cat.BY_ID[s.sid].note,
        } for s in q.values()],
        "model_performance": engine.model_comparison(12),
        "alerts": alerts.recent(conn, days=60),
        # Capped: the full 45-day list ran to ~90 entries and 18% of the feed,
        # and the page shows the 20 largest anyway. Sorted by magnitude upstream,
        # so the cap keeps the ones that matter.
        "revisions": ingestion.recent_revisions(conn, days=45)[:25],
        "vintage_coverage": vintage.coverage_report(conn),
        # The last stored vintage-true backtest. Read, never recomputed: a daily
        # run must not spend twenty minutes re-deriving it, and must not show an
        # empty Model Performance panel either.
        "backtest": _bt,
        # Spec 26's "historical probability line": the vintage-true backtest
        # path, NOT today's model replayed over revised history.
        #
        # Falls back to the summary's own copy for the same reason the summary
        # itself does -- a CI runner holds no vintage store, so the database
        # query comes back empty there and only the committed JSON has the path.
        "recession_history": (backtest_engine.probability_history(conn, 12)
                              or (_bt or {}).get("probability_history") or []),
        "runs": runlog.latest_runs(conn, 10),
    }

    artifacts = {
        "frames": frames, "factors": F, "regime": R, "horizons": horizons,
        "scenarios": scen, "variables": variables, "engine": engine,
        "quality": q, "confidence": conf,
    }

    if include_history:
        snap["history"] = {
            sid: _series_history(frames[sid])
            for sid in st.MODEL_FEATURES + ["UNRATE", "CPIAUCSL", "PAYEMS", "INDPRO",
                                            "HOUST", "DGS10", "SP500", "GOLD"]
            if sid in frames
        }
        snap["indices_history"] = _index_history(frames)
        # Each factor card's sparkline shows THAT FACTOR's own history. Showing
        # one member series instead would be a different quantity wearing the
        # factor's label.
        snap["factor_history"] = factor_engine.factor_history(frames)

    return _clean(snap), artifacts


def _index_history(frames: dict, months: int = 120) -> dict:
    """Leading / coincident / lagging index history, for the breadth chart."""
    out = {}
    for name, fn in [("leading", leading.calculate_leading_indicator_index),
                     ("coincident", leading.calculate_coincident_index),
                     ("lagging", leading.calculate_lagging_index)]:
        s = fn(frames).dropna().tail(months)
        out[name] = [{"date": d.date().isoformat(), "value": _clean(v)}
                     for d, v in s.items()]
    return out


def persist(conn, snap: dict, run: runlog.RunContext, artifacts: dict) -> None:
    """Write the snapshot's results and stamp the run for reproducibility."""
    as_of = snap["meta"]["as_of"]
    runlog.persist_snapshot(conn, run.run_id, as_of, artifacts["factors"],
                            artifacts["regime"], artifacts["horizons"],
                            artifacts["scenarios"])
    if artifacts.get("variables"):
        runlog.persist_variable_forecasts(conn, run.run_id, as_of, artifacts["variables"])

    input_hash = runlog.hash_inputs(artifacts["frames"])
    # The output hash covers the results, deliberately excluding meta (which
    # carries a wall-clock timestamp that would change on every run and make the
    # hash useless for detecting real change).
    output_hash = runlog.hash_payload(
        {k: snap[k] for k in ("headline", "recession", "factors", "regime",
                              "scenarios", "confidence") if k in snap})
    api_errors = conn.execute(
        "SELECT COUNT(*) FROM api_errors WHERE DATE(logged_at) = DATE('now')").fetchone()[0]

    run.record(
        data_timestamp=max((s["last_observation"] or "") for s in snap["quality"]) or as_of,
        data_vintage="live-latest",
        input_hash=input_hash, output_hash=output_hash,
        regime=snap["headline"]["regime"],
        prob_12m=snap["headline"]["recession_12m"],
        confidence=snap["headline"]["confidence"],
        health=snap["health"], api_errors=api_errors)

    snap["meta"]["run_id"] = run.run_id
    snap["meta"]["input_hash"] = input_hash
    snap["meta"]["output_hash"] = output_hash
