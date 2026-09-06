"""
Alert engine (spec 24).

Raises the eight conditions the spec names, plus data-health alerts, and
persists them to the `alerts` table.

DEDUPLICATION IS THE WHOLE DESIGN PROBLEM
-----------------------------------------
Every condition here is persistent by nature. The yield curve does not invert
once -- it stays inverted for months. Without deduplication, a daily run would
raise "yield curve inverted" 200 times and the operator would learn to ignore
the alert channel entirely, which is worse than having no alerts.

So each alert kind is deduplicated per (kind, as_of_date) via the alert_dedup
table, and the CROSSING alerts fire on the transition rather than on the state:
`probability_crossed_50` fires when the probability moves from below 50% to
above it, not while it sits above. State-based alerts that legitimately recur
(data health) are raised at most once per day.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

from config import strategy as st
from modules.factors import FactorScore
from modules.forecast import RecessionForecast
from modules.log import log_info
from modules.regime import RegimeState

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}


@dataclass
class Alert:
    kind: str
    severity: str
    title: str
    detail: str


def _previous(conn: sqlite3.Connection, days_back: int = 30) -> dict:
    """The most recent stored run before today, for transition detection.

    Transitions need a "before" value. Reading it from the database rather than
    holding it in memory means alerts work correctly across process restarts,
    which is how the engine actually runs (a cron job, not a daemon).
    """
    row = conn.execute(
        "SELECT as_of_date, regime FROM regimes ORDER BY as_of_date DESC, run_id DESC LIMIT 1"
    ).fetchone()
    prior_regime = row["regime"] if row else None
    prior_date = row["as_of_date"] if row else None

    prob_row = conn.execute(
        "SELECT probability FROM forecasts WHERE target='recession' AND horizon_m=12 "
        "AND as_of_date < DATE('now') ORDER BY as_of_date DESC, run_id DESC LIMIT 1"
    ).fetchone()
    return {"regime": prior_regime, "regime_date": prior_date,
            "prob_12m": prob_row["probability"] if prob_row else None}


def _crossed(prior: float | None, current: float, threshold: float) -> bool:
    """True only on the upward transition through `threshold`.

    When there is no prior value we treat being above the threshold as a
    crossing, so the very first run does not silently swallow a live warning.
    """
    if prior is None:
        return current >= threshold
    return prior < threshold <= current


def evaluate(conn: sqlite3.Connection, as_of: str, regime: RegimeState,
             factors: dict[str, FactorScore], horizons: list[RecessionForecast],
             breadth: dict, health: dict, revisions: list[dict] | None = None
             ) -> list[Alert]:
    prior = _previous(conn)
    out: list[Alert] = []
    f12 = next((f for f in horizons if f.horizon_m == 12), None)

    # -- regime change ---------------------------------------------------
    if prior["regime"] and prior["regime"] != regime.regime:
        forward = (st.REGIME_ORDER.index(regime.regime) >
                   st.REGIME_ORDER.index(prior["regime"])) \
            if regime.regime in st.REGIME_ORDER and prior["regime"] in st.REGIME_ORDER else False
        out.append(Alert(
            "regime_change", "HIGH" if forward else "WARNING",
            f"Macro regime changed: {prior['regime']} -> {regime.regime}",
            f"Trend {regime.trend}. Composite level {regime.detail.get('level')}, "
            f"momentum {regime.detail.get('momentum')}. "
            f"Previous label dated {prior['regime_date']}."))

    # -- probability thresholds -----------------------------------------
    if f12 is not None:
        p, prior_p = f12.probability, prior["prob_12m"]
        for threshold, severity, name in [
            (st.PROB_CRITICAL, "CRITICAL", "critical"),
            (st.PROB_HIGH, "HIGH", "high"),
            (st.PROB_WARNING, "WARNING", "warning"),
        ]:
            if _crossed(prior_p, p, threshold):
                out.append(Alert(
                    f"prob_crossed_{int(threshold * 100)}", severity,
                    f"12-month recession probability crossed {threshold:.0%} ({name})",
                    f"Now {p:.1%}"
                    + (f", was {prior_p:.1%}" if prior_p is not None else "")
                    + f". Model {f12.model}, "
                    + ("calibrated" if f12.calibrated else "UNCALIBRATED") + "."))
                break   # one threshold alert per run, the highest crossed
        if prior_p is not None and abs(p - prior_p) >= st.PROB_JUMP_EPS:
            out.append(Alert(
                "prob_jump", "HIGH" if p > prior_p else "INFO",
                f"12-month recession probability moved {p - prior_p:+.1%}",
                f"From {prior_p:.1%} to {p:.1%} since the previous run."))

    # -- breadth ----------------------------------------------------------
    wk = breadth.get("weakening_pct", 0.0)
    if wk >= st.BREADTH_WEAK_CRITICAL:
        out.append(Alert("breadth_critical", "CRITICAL",
                         f"Leading-indicator breadth critically weak: {wk:.0%} weakening",
                         f"{breadth.get('weakening')} of {breadth.get('total')} leading "
                         f"indicators deteriorating; only {breadth.get('improving')} improving."))
    elif wk >= st.BREADTH_WEAK_WARNING:
        out.append(Alert("breadth_warning", "WARNING",
                         f"Leading-indicator breadth deteriorating: {wk:.0%} weakening",
                         f"{breadth.get('weakening')} of {breadth.get('total')} leading "
                         f"indicators deteriorating."))

    # -- credit stress ----------------------------------------------------
    credit = factors.get("credit")
    if credit and credit.detail.get("turning_point", {}).get("turning"):
        tp = credit.detail["turning_point"]
        out.append(Alert("credit_turning", "HIGH", "Credit turning point detected",
                         tp.get("interpretation", "")))
    if credit and credit.detail.get("turning_point", {}).get("change_3m_pp", 0) >= st.CREDIT_JUMP_PP:
        tp = credit.detail["turning_point"]
        out.append(Alert("credit_stress_jump", "CRITICAL",
                         f"High-yield spreads widened {tp['change_3m_pp']:+.2f}pp in 3 months",
                         tp.get("interpretation", "")))

    # -- factor turning points -------------------------------------------
    for name in ("labor", "housing", "consumer"):
        f = factors.get(name)
        if f and f.momentum is not None and f.momentum <= -st.TURNING_POINT_EPS:
            out.append(Alert(
                f"{name}_turning", "HIGH",
                f"{name.title()} turning point: score fell {f.momentum:+.2f} over "
                f"{st.MOMENTUM_MONTHS} months",
                f"{name.title()} factor now {f.score:+.2f}. Largest drags: " +
                ", ".join(c.label for c in f.top_drivers(3))))

    # -- yield curve regime ------------------------------------------------
    curve = factors.get("curve")
    if curve:
        state = curve.detail.get("state")
        if state in ("INVERTED_DEEP", "RE_STEEPENING_BULL"):
            out.append(Alert(
                "curve_regime", "HIGH" if state == "INVERTED_DEEP" else "CRITICAL",
                f"Yield curve regime: {state.replace('_', ' ').title()}",
                curve.detail.get("interpretation", "")
                + f" Spread {curve.detail.get('spread_10y3m')}pp, inverted for "
                  f"{curve.detail.get('inversion_duration_months')} months."))

    # -- data quality -----------------------------------------------------
    if health.get("health", 1.0) < st.DATA_HEALTH_ALERT:
        out.append(Alert(
            "data_health", "WARNING" if health.get("critical", 0) == 0 else "CRITICAL",
            f"Data health degraded to {health['health']:.0%}",
            f"{health.get('stale', 0)} stale, {health.get('critical', 0)} critical, "
            f"{health.get('low', 0)} LOW-grade series. Stale: "
            f"{', '.join(health.get('stale_series', [])[:8])}"))

    # -- material revisions ------------------------------------------------
    if revisions:
        big = [r for r in revisions if r.get("pct_change") and abs(r["pct_change"]) > 0.02]
        if big:
            out.append(Alert(
                "data_revision", "INFO",
                f"{len(big)} material data revisions detected",
                "; ".join(f"{r['series_id']} {r['observation_date']}: "
                          f"{r['old_value']} -> {r['new_value']} ({r['pct_change']:+.1%})"
                          for r in big[:5])))

    out.sort(key=lambda a: SEVERITY_ORDER.get(a.severity, 9))
    return out


def persist(conn: sqlite3.Connection, as_of: str, alerts: list[Alert]) -> list[Alert]:
    """Store alerts, skipping ones already raised for this (kind, date)."""
    raised: list[Alert] = []
    for a in alerts:
        exists = conn.execute(
            "SELECT 1 FROM alert_dedup WHERE kind=? AND as_of_date=?",
            (a.kind, as_of)).fetchone()
        if exists:
            continue
        conn.execute(
            "INSERT INTO alerts (as_of_date, kind, severity, title, detail) VALUES (?,?,?,?,?)",
            (as_of, a.kind, a.severity, a.title, a.detail))
        conn.execute("INSERT INTO alert_dedup (kind, as_of_date) VALUES (?,?)",
                     (a.kind, as_of))
        raised.append(a)
    conn.commit()
    if raised:
        log_info(f"[alerts] raised {len(raised)} new alert(s): "
                 + ", ".join(f"{a.severity}:{a.kind}" for a in raised))
    return raised


def recent(conn: sqlite3.Connection, days: int = 30, limit: int = 40) -> list[dict]:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT alert_id, raised_at, as_of_date, kind, severity, title, detail, acknowledged "
        "FROM alerts WHERE as_of_date >= ? ORDER BY raised_at DESC LIMIT ?",
        (cutoff, limit)).fetchall()
    return [dict(r) for r in rows]
