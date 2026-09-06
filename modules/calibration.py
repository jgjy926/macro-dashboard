"""
Probability calibration (spec 16).

The objective, in the spec's words: when the engine says 70%, that should
historically correspond to roughly a 70% occurrence rate. Discrimination and
calibration are different properties -- a model can rank perfectly (AUC 0.9) and
still be systematically overconfident, and a recession model is far more often
used for the number than for the ranking.

Two things are kept rigorously separate here, per spec 13's instruction not to
label a weighted score as a probability:

    RAW MACRO RISK SCORE     the model's uncalibrated output
    CALIBRATED PROBABILITY   that output mapped through a fitted calibrator

Both are stored and both are shown. The calibrator is fitted ONLY on
walk-forward out-of-sample predictions. Fitting it on in-sample fits would
calibrate away the overconfidence that the in-sample fit created, producing a
curve that looks perfect and generalises not at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config import strategy as st
from modules.log import log_info, log_warning


@dataclass
class Calibrator:
    method: str                    # isotonic | sigmoid | identity
    fitted: bool
    n_samples: int
    model: object | None = None
    note: str = ""

    def transform(self, p: np.ndarray | float) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(p, dtype="float64"))
        if not self.fitted or self.model is None:
            return np.clip(arr, 0.0, 1.0)
        if self.method == "isotonic":
            out = self.model.predict(arr)
        else:
            out = self.model.predict_proba(arr.reshape(-1, 1))[:, 1]
        return np.clip(out, 0.0, 1.0)

    def one(self, p: float) -> float:
        return float(self.transform(p)[0])


def calibrate_probability(pred: pd.Series, actual: pd.Series,
                          method: str | None = None) -> Calibrator:
    """Fit a calibrator on out-of-sample predictions.

    The estimator is chosen by sample size, because "calibrate or do not
    calibrate" is a false choice at these sample counts:

      >= CALIBRATION_MIN_SAMPLES        isotonic (non-parametric, needs volume)
      >= CALIBRATION_SMALL_SAMPLE_MIN   sigmoid / Platt (two parameters, stable)
      below that                        identity, and `note` says why

    Isotonic on 40 points is a step function that reproduces the training noise
    exactly, so it is not offered there. Platt scaling is, because two parameters
    cannot overfit 40 points the way a free-form monotone fit can -- and refusing
    to calibrate at all is not the safe option it looks like: the vintage backtest
    has only ~140 points per horizon, and leaving those uncalibrated produced a
    NEGATIVE Brier skill at every horizon while the model discriminated fine.
    """
    mask = pred.notna() & actual.notna()
    p, a = pred[mask].to_numpy(dtype="float64"), actual[mask].to_numpy(dtype="float64")

    # Choose the estimator by sample size rather than refusing outright. Isotonic
    # needs many points; Platt (sigmoid) has two parameters and is the standard
    # small-sample choice. An explicit `method` argument always wins.
    if method is None:
        if len(p) >= st.CALIBRATION_MIN_SAMPLES:
            method = st.CALIBRATION_METHOD
        elif len(p) >= st.CALIBRATION_SMALL_SAMPLE_MIN:
            method = "sigmoid"

    if method is None:
        return Calibrator("identity", False, len(p), None,
                          note=(f"only {len(p)} out-of-sample points, below the "
                                f"{st.CALIBRATION_SMALL_SAMPLE_MIN} minimum for even "
                                f"two-parameter Platt scaling -- probabilities are "
                                f"reported UNCALIBRATED rather than fitted to a sample "
                                f"too small to support it"))
    if len(np.unique(a)) < 2:
        return Calibrator("identity", False, len(p), None,
                          note="no outcome variation in the validation window")

    try:
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, a)
        else:
            model = LogisticRegression(max_iter=1000).fit(p.reshape(-1, 1), a)
    except Exception as e:
        log_warning(f"[calibration] fit failed ({e}); using identity")
        return Calibrator("identity", False, len(p), None, note=f"fit failed: {e}")

    why = (" (Platt scaling rather than isotonic: below the "
           f"{st.CALIBRATION_MIN_SAMPLES}-point isotonic threshold, where a "
           "non-parametric fit would reproduce training noise)"
           if method == "sigmoid" and len(p) < st.CALIBRATION_MIN_SAMPLES else "")
    return Calibrator(method, True, len(p), model,
                      note=f"{method} calibration fitted on {len(p)} out-of-sample points{why}")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def calculate_brier_score(pred: np.ndarray | pd.Series, actual: np.ndarray | pd.Series
                          ) -> float | None:
    p = np.asarray(pred, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    mask = ~(np.isnan(p) | np.isnan(a))
    if mask.sum() == 0:
        return None
    return round(float(np.mean((p[mask] - a[mask]) ** 2)), 5)


def brier_skill_score(pred, actual, base_rate=None) -> float | None:
    """Brier relative to a base-rate ("climatology") reference forecast.

    The raw Brier score of a rare event is misleadingly good -- predicting 0%
    every month scores ~0.09 on a 10% base rate and forecasts nothing. The skill
    score answers the question that matters: is this better than the base rate?
    Positive = yes, 0 = no better, negative = worse.

    `base_rate` may be a scalar OR a per-observation array, and the difference
    matters more than it looks. The conventional choice is the realised frequency
    of the very window being scored -- which is an ORACLE: a forecaster in 1995
    could not know how many recessions 1990-2026 would contain. Scoring a
    real-time engine against it charges the engine for not knowing the future
    base rate, which is not a forecasting error.

    So the backtest passes a point-in-time climatology array here (what a
    base-rate forecaster could actually have quoted at each date) as the primary
    reference, and reports the conventional full-sample version alongside for
    comparability.
    """
    a = np.asarray(actual, dtype="float64")
    mask = ~np.isnan(a)
    if mask.sum() == 0:
        return None
    if base_rate is None:
        ref_vals = np.full(mask.sum(), float(np.mean(a[mask])))
    elif np.isscalar(base_rate):
        ref_vals = np.full(mask.sum(), float(base_rate))
    else:
        ref_vals = np.asarray(base_rate, dtype="float64")[mask]
    bs = calculate_brier_score(pred, actual)
    ref = float(np.mean((ref_vals - a[mask]) ** 2))
    if bs is None or ref == 0:
        return None
    return round(1.0 - bs / ref, 4)


# ---------------------------------------------------------------------------
# climatology and shrinkage
# ---------------------------------------------------------------------------
def horizon_outcome_series(recession_rows: list[tuple[str, float]], horizon_m: int):
    """Monthly (date, outcome) for 'a recession starts within horizon_m months'.

    Precomputed once so the point-in-time climatology below is an O(1) lookup
    rather than an O(n) scan per as-of date -- the naive version took minutes
    across five horizons and 147 dates.
    """
    if not recession_rows:
        return pd.DatetimeIndex([]), np.array([])
    s = pd.Series({pd.Timestamp(d): v for d, v in recession_rows}).sort_index()
    s = s.resample("ME").last().ffill()
    starts = list(s.index[(s == 1) & (s.shift(1) == 0)])
    if not starts:
        return s.index, np.zeros(len(s))
    # searchsorted on the DatetimeIndex, NOT integer comparison. An earlier
    # version compared `.value` (always nanoseconds) against
    # `index.values.astype("int64")`, which under pandas 3.0's datetime64[s]
    # resolution is SECONDS -- so every comparison was off by 10^9 and the base
    # rate came out at 0.9 instead of 0.03. This form has no resolution
    # assumption to get wrong.
    starts_idx = pd.DatetimeIndex(starts)
    out = np.zeros(len(s))
    for i, m in enumerate(s.index):
        end = m + pd.DateOffset(months=horizon_m)
        lo = starts_idx.searchsorted(m, side="right")     # first start strictly after m
        hi = starts_idx.searchsorted(end, side="right")   # first start after m+h
        out[i] = 1.0 if hi > lo else 0.0
    return s.index, out


def empirical_base_rate(recession_rows: list[tuple[str, float]], horizon_m: int,
                        as_of: str | None = None,
                        fallback: float | None = None) -> float:
    """P(a recession starts within horizon_m months), from history up to as_of.

    Point-in-time by construction: only months whose outcome window had already
    CLOSED by as_of are counted, so a date whose answer was not yet known cannot
    contribute. Replaces the hardcoded HORIZON_BASE_RATE, which was both a mild
    look-ahead (a full-sample constant applied at historical dates) and biased
    high for the modern era.
    """
    default = fallback if fallback is not None else st.HORIZON_BASE_RATE.get(horizon_m, 0.15)
    idx, out = horizon_outcome_series(recession_rows, horizon_m)
    if len(idx) == 0:
        return default
    if as_of is not None:
        cutoff = pd.Timestamp(as_of) - pd.DateOffset(months=horizon_m)
        sel = out[idx <= cutoff]
    else:
        sel = out
    if len(sel) < 24:
        return default
    return float(sel.mean())


@dataclass
class ShrinkageCalibrator:
    """A one-parameter calibrator: p = base + lambda * (score - base).

    Why one parameter rather than isotonic or Platt. In the vintage backtest each
    horizon has ~135 scoreable points containing THREE recession episodes, and
    the calibrator is refitted at every as-of date. A two-parameter Platt fit on
    that is unstable, and an isotonic fit is hopeless; refitting either one
    freely across dates was measurably destroying the model's ranking -- pooled
    AUC on the calibrated series fell to 0.44 while the raw score held 0.70.

    This form cannot do that. It is monotone in the score for any lambda >= 0, it
    has a single interpretable parameter ("how much of the model's sharpness does
    the evidence actually support"), and lambda is itself shrunk by the number of
    recession EVENTS seen rather than the row count -- so with no events it is 0,
    the engine quotes the base rate, and skill is 0 rather than negative.
    """
    base: float
    lam: float
    n_points: int
    n_events: int
    note: str = ""

    def one(self, score: float) -> float:
        return float(np.clip(self.base + self.lam * (score - self.base), 0.0, 1.0))

    def transform(self, scores) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(scores, dtype="float64"))
        return np.clip(self.base + self.lam * (arr - self.base), 0.0, 1.0)


def fit_shrinkage(scores, actual, base: float,
                  evidence_prior: int | None = None) -> ShrinkageCalibrator:
    """Least-squares optimal sharpness, shrunk by how much evidence supports it.

    The unconstrained optimum of Brier over p = b + L(s - b) is
    L* = sum((s-b)(y-b)) / sum((s-b)^2) -- a closed form, no iteration. It is
    clipped to [0, 1] (a negative L would mean inverting the model, which on this
    sample size is overwhelmingly more likely to be noise than a real inversion)
    and then multiplied by n_events / (n_events + evidence_prior).
    """
    c = st.SHRINKAGE_EVIDENCE_PRIOR if evidence_prior is None else evidence_prior
    s = np.asarray(scores, dtype="float64")
    y = np.asarray(actual, dtype="float64")
    mask = ~(np.isnan(s) | np.isnan(y))
    s, y = s[mask], y[mask]
    n, n_events = len(s), int(y.sum())

    if n < st.SHRINKAGE_MIN_POINTS or n_events == 0:
        return ShrinkageCalibrator(base, 0.0, n, n_events, note=(
            f"no sharpness applied: {n} points and {n_events} recession events seen "
            f"-- the engine quotes the base rate until the signal has demonstrated "
            f"value out of sample"))
    u = s - base
    denom = float((u * u).sum())
    if denom <= 0:
        return ShrinkageCalibrator(base, 0.0, n, n_events, note="score has no variance")
    raw_lam = float((u * (y - base)).sum() / denom)
    lam = max(0.0, min(1.0, raw_lam)) * n_events / (n_events + c)
    return ShrinkageCalibrator(base, lam, n, n_events, note=(
        f"lambda {lam:.3f} (unshrunk {raw_lam:.3f}) from {n} points and {n_events} "
        f"recession events; base rate {base:.3f}"))


def calculate_calibration_curve(pred, actual, bins: int | None = None) -> list[dict]:
    """Reliability curve: mean predicted vs observed frequency, per bin.

    Uses fixed-width bins rather than quantile bins so the x-axis is the
    probability scale the reader expects; empty bins are reported as empty
    instead of silently dropped, because an empty 70-80% bin is itself
    information (the model never says 70-80%).
    """
    n_bins = bins or st.CALIBRATION_BINS
    p = np.asarray(pred, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    mask = ~(np.isnan(p) | np.isnan(a))
    p, a = p[mask], a[mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        n = int(sel.sum())
        out.append({
            "bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
            "n": n,
            "mean_predicted": round(float(p[sel].mean()), 4) if n else None,
            "observed_frequency": round(float(a[sel].mean()), 4) if n else None,
        })
    return out


def expected_calibration_error(pred, actual, bins: int | None = None) -> float | None:
    """ECE: sample-weighted mean gap between predicted and observed, in [0, 1].

    One number for "how far off are the probabilities". 0.05 means the typical
    forecast is 5 percentage points from the realised frequency.
    """
    curve = calculate_calibration_curve(pred, actual, bins)
    total = sum(b["n"] for b in curve)
    if not total:
        return None
    err = sum(b["n"] * abs(b["mean_predicted"] - b["observed_frequency"])
              for b in curve if b["n"] and b["mean_predicted"] is not None)
    return round(err / total, 4)


@dataclass
class CalibrationReport:
    horizon_m: int
    model: str
    calibrator: Calibrator
    brier_raw: float | None = None
    brier_calibrated: float | None = None
    skill_raw: float | None = None
    skill_calibrated: float | None = None
    ece_raw: float | None = None
    ece_calibrated: float | None = None
    curve: list[dict] = field(default_factory=list)

    def improved(self) -> bool:
        if self.brier_raw is None or self.brier_calibrated is None:
            return False
        return self.brier_calibrated <= self.brier_raw

    def to_dict(self) -> dict:
        return {
            "horizon_m": self.horizon_m, "model": self.model,
            "method": self.calibrator.method, "fitted": self.calibrator.fitted,
            "n_samples": self.calibrator.n_samples, "note": self.calibrator.note,
            "brier_raw": self.brier_raw, "brier_calibrated": self.brier_calibrated,
            "brier_skill_raw": self.skill_raw,
            "brier_skill_calibrated": self.skill_calibrated,
            "ece_raw": self.ece_raw, "ece_calibrated": self.ece_calibrated,
            "improved": self.improved(), "curve": self.curve,
        }


def build_report(pred: pd.Series, actual: pd.Series, horizon_m: int, model: str
                 ) -> CalibrationReport:
    """Fit a calibrator and measure whether it actually helped.

    The engine does not assume calibration improves things -- it checks. If the
    fitted calibrator has a worse Brier score than the raw output, `improved()`
    is False and modules/forecast.py keeps the raw probability, because a
    calibrator that hurts is just an extra layer of overfitting.
    """
    cal = calibrate_probability(pred, actual)
    calibrated = pd.Series(cal.transform(pred.to_numpy()), index=pred.index)
    rep = CalibrationReport(horizon_m=horizon_m, model=model, calibrator=cal)
    rep.brier_raw = calculate_brier_score(pred, actual)
    rep.brier_calibrated = calculate_brier_score(calibrated, actual)
    rep.skill_raw = brier_skill_score(pred, actual)
    rep.skill_calibrated = brier_skill_score(calibrated, actual)
    rep.ece_raw = expected_calibration_error(pred, actual)
    rep.ece_calibrated = expected_calibration_error(calibrated, actual)
    rep.curve = calculate_calibration_curve(calibrated, actual)
    log_info(f"[calibration] h={horizon_m}m {model}: Brier {rep.brier_raw} -> "
             f"{rep.brier_calibrated}, ECE {rep.ece_raw} -> {rep.ece_calibrated} "
             f"({cal.method}, n={cal.n_samples})")
    return rep
