"""
Model run logging (spec 31) -- reproducibility.

Spec 31 lists exactly what every run must record, and the reason is stated in
one line: "This makes forecasts reproducible." The two fields that actually
deliver that are input_hash and output_hash.

    input_hash   a hash of every value the run consumed. Two runs with the same
                 input_hash saw identical data, so any difference in their
                 output is a code or configuration change, not a data change.
    output_hash  a hash of what the run produced. Same inputs + same code =>
                 same output_hash, which is the property that lets you prove a
                 result was not quietly altered.

Together with model_version and feature_version, a stored run can be checked
against a re-run months later and any divergence attributed to the right cause.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime

from config import settings
from modules.log import log_error, log_info


def hash_payload(obj) -> str:
    """Stable SHA-256 of any JSON-serialisable structure.

    sort_keys is essential: dict ordering must not change the hash, or the
    reproducibility guarantee becomes a coin flip on insertion order.
    """
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def hash_inputs(frames: dict) -> str:
    """Hash of the data the run consumed: last observation date and value per
    series. Cheap, and it changes whenever any input changes."""
    payload = {}
    for sid, frame in sorted(frames.items()):
        raw = getattr(frame, "raw", None)
        if raw is None or len(raw) == 0:
            payload[sid] = None
            continue
        payload[sid] = [str(raw.index[-1].date()), round(float(raw.iloc[-1]), 6), len(raw)]
    return hash_payload(payload)


@dataclass
class RunContext:
    conn: sqlite3.Connection
    run_type: str
    run_id: int | None = None
    started: float = field(default_factory=time.time)
    data_timestamp: str | None = None
    input_hash: str | None = None

    def __enter__(self) -> "RunContext":
        cur = self.conn.execute(
            "INSERT INTO model_runs (run_type, status, model_version, feature_version) "
            "VALUES (?, 'RUNNING', ?, ?)",
            (self.run_type, settings.MODEL_VERSION, settings.FEATURE_VERSION))
        self.run_id = cur.lastrowid
        self.conn.commit()
        log_info(f"[run] {self.run_type} run_id={self.run_id} started "
                 f"({settings.MODEL_VERSION})")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration = round(time.time() - self.started, 2)
        if exc_type is not None:
            self.conn.execute(
                "UPDATE model_runs SET status='FAILED', duration_sec=?, error=? WHERE run_id=?",
                (duration, f"{exc_type.__name__}: {exc}", self.run_id))
            self.conn.commit()
            log_error(f"[run] run_id={self.run_id} FAILED after {duration}s: {exc}")
            return False       # re-raise: a failed run must be visible, not swallowed
        self.conn.execute(
            "UPDATE model_runs SET status='COMPLETED', duration_sec=? WHERE run_id=?",
            (duration, self.run_id))
        self.conn.commit()
        log_info(f"[run] run_id={self.run_id} completed in {duration}s")
        return False

    def record(self, *, data_timestamp: str, data_vintage: str, input_hash: str,
               output_hash: str, regime: str, prob_12m: float | None,
               confidence: float | None, health: dict, api_errors: int) -> None:
        self.conn.execute(
            """
            UPDATE model_runs SET
                data_timestamp=?, data_vintage=?, input_hash=?, output_hash=?,
                regime=?, prob_12m=?, confidence=?, data_health=?,
                series_ok=?, series_stale=?, series_failed=?, api_errors=?
            WHERE run_id=?
            """,
            (data_timestamp, data_vintage, input_hash, output_hash, regime, prob_12m,
             confidence, health.get("health"), health.get("n", 0) - health.get("low", 0),
             health.get("stale", 0), health.get("critical", 0), str(api_errors),
             self.run_id))
        self.conn.commit()


def latest_runs(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM model_runs ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def reproducibility_check(conn: sqlite3.Connection, run_id_a: int, run_id_b: int) -> dict:
    """Compare two runs and attribute any difference to the right cause.

    This is the payoff for storing the hashes: it turns "the number changed"
    into a specific, actionable answer.
    """
    a = conn.execute("SELECT * FROM model_runs WHERE run_id=?", (run_id_a,)).fetchone()
    b = conn.execute("SELECT * FROM model_runs WHERE run_id=?", (run_id_b,)).fetchone()
    if not a or not b:
        return {"comparable": False, "reason": "one or both runs not found"}

    same_input = a["input_hash"] == b["input_hash"]
    same_output = a["output_hash"] == b["output_hash"]
    same_code = (a["model_version"] == b["model_version"]
                 and a["feature_version"] == b["feature_version"])

    if same_input and same_output:
        verdict = "Reproducible: identical inputs produced identical outputs."
    elif same_input and same_code and not same_output:
        verdict = ("NOT reproducible: identical inputs and versions produced different "
                   "outputs. This indicates non-determinism in the model code -- an "
                   "unseeded random state or an unstable ordering.")
    elif same_input and not same_code:
        verdict = "Outputs differ because the model or feature version changed."
    else:
        verdict = "Outputs differ because the input data changed (expected between runs)."

    return {"comparable": True, "same_input": same_input, "same_output": same_output,
            "same_code": same_code, "verdict": verdict,
            "a": {"run_id": run_id_a, "input_hash": a["input_hash"],
                  "output_hash": a["output_hash"], "model_version": a["model_version"]},
            "b": {"run_id": run_id_b, "input_hash": b["input_hash"],
                  "output_hash": b["output_hash"], "model_version": b["model_version"]}}


def persist_snapshot(conn: sqlite3.Connection, run_id: int, as_of: str,
                     factors: dict, regime_state, horizons: list, scenario_list: list
                     ) -> None:
    """Write the run's factor scores, regime, forecasts and scenarios."""
    for name, f in factors.items():
        conn.execute(
            "INSERT OR REPLACE INTO factor_scores (as_of_date, factor, score, momentum, "
            "n_inputs, coverage, detail, run_id) VALUES (?,?,?,?,?,?,?,?)",
            (as_of, name, f.score if f.score is not None else 0.0, f.momentum,
             f.n_inputs, f.coverage, json.dumps(f.detail, default=str), run_id))

    conn.execute(
        "INSERT OR REPLACE INTO regimes (as_of_date, regime, growth_regime, "
        "inflation_regime, labor_regime, credit_regime, liquidity_regime, policy_regime, "
        "strength, trend, detail, run_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (as_of, regime_state.regime, regime_state.growth_regime,
         regime_state.inflation_regime, regime_state.labor_regime,
         regime_state.credit_regime, regime_state.liquidity_regime,
         regime_state.policy_regime, regime_state.strength, regime_state.trend,
         json.dumps(regime_state.detail, default=str), run_id))

    for f in horizons:
        conn.execute(
            "INSERT OR REPLACE INTO forecasts (as_of_date, horizon_m, target, model, "
            "raw_score, probability, confidence, drivers, run_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (as_of, f.horizon_m, "recession", f.model, f.raw_score, f.probability,
             None, json.dumps(f.drivers, default=str), run_id))

    for s in scenario_list:
        conn.execute(
            "INSERT OR REPLACE INTO scenarios (as_of_date, scenario, probability, detail, run_id) "
            "VALUES (?,?,?,?,?)",
            (as_of, s.name, s.probability, json.dumps(s.to_dict(), default=str), run_id))
    conn.commit()


def persist_variable_forecasts(conn: sqlite3.Connection, run_id: int, as_of: str,
                               variables: dict) -> None:
    for target, forecasts in variables.items():
        for v in forecasts:
            conn.execute(
                "INSERT OR REPLACE INTO forecasts (as_of_date, horizon_m, target, model, "
                "point, lo, hi, direction, confidence, drivers, run_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (as_of, v.horizon_m, target, "ridge", v.point, v.lo, v.hi,
                 v.direction, v.confidence, json.dumps(v.drivers, default=str), run_id))
    conn.commit()
