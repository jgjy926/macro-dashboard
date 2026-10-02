"""
Forecast engine (spec 13, 14).

Produces two very different kinds of output, and keeps them apart:

  RECESSION (spec 13)  A probability at 3/6/9/12/18 months. Stored as BOTH a raw
                       macro risk score and a calibrated probability, because
                       spec 13 forbids labelling a weighted score as a
                       probability. The dashboard shows both.

  MACRO VARIABLES (14) Growth, GDP, inflation, unemployment, policy rate, long
                       rates, housing, consumer, credit. Each returns a
                       direction, a point estimate with an interval, a
                       confidence, and its drivers.

HOW THE NUMERIC FORECASTS ARE MADE, AND WHAT THEY ARE NOT
---------------------------------------------------------
Each numeric forecast is a ridge regression of the target's future YoY change on
today's nine factor scores, fitted on history and validated walk-forward. The
prediction interval is the walk-forward residual spread, NOT a textbook
regression interval -- an in-sample interval on macro data is roughly half the
width it should be, and quoting one would be the single most misleading number
on the dashboard.

These are not structural macroeconomic models. They are conditional-mean
estimates from a handful of factor scores, and their intervals are wide because
that is the truth about 12-month macro forecasting. Every result carries the
walk-forward RMSE that produced its interval, so the reader can see the model's
actual historical error rather than an assertion of precision.
"""
from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import series as cat
from config import settings
from config import strategy as st
from modules import calibration, models
from modules.factors import FactorScore
from modules.log import log_info, log_warning
from modules.transforms import SignalFrame

# A full walk-forward pass over five models and five horizons takes minutes.
# Spec 25 asks for a DAILY snapshot but only MONTHLY calibration and QUARTERLY
# retraining, so the daily run reuses a cached fit and only the scheduled
# retraining cadence pays that cost. The cache key is the feature matrix's shape
# and content hash, so any change to the data or the feature set invalidates it
# automatically rather than relying on anyone remembering to clear it.
CACHE_PATH = settings.CACHE_DIR / "model_cache.pkl"

# Clamp raw model output before blending. A probit or GBM can saturate to
# exactly 0.0 or 1.0 by driving its linear predictor past the point where the
# link function underflows -- which is a numerical artefact, not a claim of
# certainty, and must never be published as one.
PROB_FLOOR, PROB_CEIL = 0.002, 0.998


# ---------------------------------------------------------------------------
# recession
# ---------------------------------------------------------------------------
@dataclass
class RecessionForecast:
    horizon_m: int
    raw_score: float               # the model's uncalibrated output
    probability: float             # calibrated (or raw, if calibration was declined)
    model: str
    calibrated: bool
    base_rate: float
    signal_weight: float
    auc: float | None = None
    brier: float | None = None
    drivers: list[dict] = field(default_factory=list)
    note: str = ""

    def band(self) -> str:
        if self.probability >= st.PROB_CRITICAL:
            return "CRITICAL"
        if self.probability >= st.PROB_HIGH:
            return "HIGH"
        if self.probability >= st.PROB_WARNING:
            return "ELEVATED"
        return "NORMAL"


@dataclass
class VariableForecast:
    target: str
    label: str
    horizon_m: int
    direction: str                 # UP | DOWN | FLAT
    point: float | None
    lo: float | None
    hi: float | None
    unit: str
    confidence: float
    rmse: float | None = None
    n_train: int = 0
    drivers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    note: str = ""


class ForecastEngine:
    """Holds the fitted state for one run, so the five horizons share one
    feature matrix and one walk-forward pass instead of recomputing per call."""

    def __init__(self, frames: dict[str, SignalFrame], factors: dict[str, FactorScore],
                 recession_rows: list[tuple[str, float]], *, retrain: bool = False,
                 use_cache: bool = True):
        self.frames = frames
        self.factors = factors
        self.recession = recession_rows
        self.X = models.build_feature_matrix(frames)
        self._wf: dict[int, dict[str, models.WalkForwardResult]] = {}
        self._selected: dict[int, tuple[str, str]] = {}
        self._calibrators: dict[int, calibration.CalibrationReport] = {}
        self._fitted: dict[int, models.FittedModel] = {}
        self.use_cache = use_cache
        self.retrained = False
        self.stale_model_month: str | None = None   # set when last month's fit is reused
        self._cache_key = self._compute_cache_key()
        self._static_key = self._compute_cache_key(with_month=False)
        if use_cache and not retrain:
            self._load_cache()

    # -- fit cache --------------------------------------------------------
    def _compute_cache_key(self, with_month: bool = True) -> str:
        """Identify a fit by what it was TRAINED on, not by every input byte.

        The first version hashed the whole feature matrix, which meant any change
        to any observation invalidated the cache -- and macro data changes every
        single day. So the "daily reuses a cached fit" design never actually
        engaged in production: every daily run silently paid a full five-model,
        five-horizon walk-forward (~6 minutes). It only looked fast locally
        because the data had not moved between two runs on the same afternoon.

        Refitting daily is also not what spec 25 asks for -- it puts retraining
        on the WEEKLY and MONTHLY cadence, and daily is meant to be a cheap
        snapshot. Standard practice agrees: you do not refit a recession model
        because one weekly claims print landed.

        So the key covers the things that genuinely change what a fit means --
        the feature set, the model/feature versions, and the training month --
        and deliberately ignores the newest observations, which are applied to
        the cached model at prediction time. `retrain=True` (weekly, monthly,
        quarterly) always forces a fresh fit regardless.
        """
        if self.X.empty:
            return "empty"
        h = hashlib.sha256()
        h.update(",".join(map(str, self.X.columns)).encode())
        h.update(f"{settings.MODEL_VERSION}|{settings.FEATURE_VERSION}".encode())
        # Training month, not the newest observation date: within a calendar
        # month the fitted model is unchanged and only the inputs it scores move.
        # `with_month=False` is the same fit identity minus the month -- what
        # decides whether last month's fit may stand in (ALLOW_STALE_MODEL).
        if with_month:
            h.update(self.training_month.encode())
        return h.hexdigest()[:16]

    @property
    def training_month(self) -> str:
        return "" if self.X.empty else str(self.X.index[-1])[:7]

    def _load_cache(self) -> None:
        try:
            if not CACHE_PATH.exists():
                return
            with open(CACHE_PATH, "rb") as f:
                blob = pickle.load(f)
            if blob.get("key") != self._cache_key:
                # Same features and versions, only an older month: under
                # ALLOW_STALE_MODEL score with it now and leave the refit to the
                # separate retrain job. Old caches carry no static_key and so
                # always refit -- the safe default.
                if (settings.ALLOW_STALE_MODEL and blob.get("static_key")
                        and blob["static_key"] == self._static_key):
                    self.stale_model_month = blob.get("month") or "unknown"
                    log_warning(f"[forecast] model cache is from {self.stale_model_month}, "
                                f"data is now {self.training_month}: reusing it for this run "
                                f"(MACRO_ALLOW_STALE_MODEL) and flagging a retrain")
                    try:
                        settings.RETRAIN_MARKER.parent.mkdir(parents=True, exist_ok=True)
                        settings.RETRAIN_MARKER.write_text(self.training_month, encoding="utf-8")
                    except OSError as e:
                        log_warning(f"[forecast] could not write retrain marker: {e}")
                else:
                    log_info("[forecast] model cache stale (new training month, or the "
                             "feature set / model version changed) -- will retrain")
                    return
            self._wf = blob["wf"]
            self._selected = blob["selected"]
            self._calibrators = blob["calibrators"]
            self._fitted = blob["fitted"]
            log_info(f"[forecast] reusing cached fits for horizons {sorted(self._fitted)}")
        except Exception as e:
            log_warning(f"[forecast] could not read model cache ({e}); retraining")
            self._wf, self._selected, self._calibrators, self._fitted = {}, {}, {}, {}

    def save_cache(self) -> None:
        # A reused stale fit must never be re-saved under THIS month's key: that
        # would launder last month's model into a "fresh" one and the refit
        # would never happen.
        if not self.use_cache or not self._fitted or self.stale_model_month:
            return
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump({"key": self._cache_key, "static_key": self._static_key,
                             "month": self.training_month, "wf": self._wf,
                             "selected": self._selected, "calibrators": self._calibrators,
                             "fitted": self._fitted}, f)
            settings.RETRAIN_MARKER.unlink(missing_ok=True)
        except Exception as e:
            log_warning(f"[forecast] could not write model cache: {e}")

    # -- model plumbing ---------------------------------------------------
    def _prepare(self, horizon_m: int):
        """Walk-forward, select, calibrate and finally fit -- once per horizon."""
        if horizon_m in self._wf:
            return
        self.retrained = True
        y = models.build_target(self.recession, horizon_m, self.X.index)
        Xa, ya = models.align(self.X, y)
        if Xa.empty:
            self._wf[horizon_m] = {}
            self._selected[horizon_m] = ("rule", "no usable training data")
            return

        results = models.compare_models(Xa, ya, horizon_m)
        results["ensemble"] = models.ensemble_walk_forward(results, horizon_m)
        self._wf[horizon_m] = results
        name, reason = models.select_model(results)
        self._selected[horizon_m] = (name, reason)

        chosen = results.get(name)
        if chosen and not chosen.predictions.empty:
            self._calibrators[horizon_m] = calibration.build_report(
                chosen.predictions, chosen.actuals, horizon_m, name)

        # The ensemble has no single fitted object to predict with; fall back to
        # its best member for the live prediction and say so in the note.
        fit_name = name if name != "ensemble" else max(
            (m for m in st.ENSEMBLE_MEMBERS if results.get(m) and results[m].auc is not None),
            key=lambda m: results[m].auc, default="rule")
        fitted = models.fit_final(Xa, ya, fit_name, horizon_m)
        if fitted:
            wf = results.get(fit_name)
            fitted.auc = wf.auc if wf else None
            fitted.brier = wf.brier if wf else None
            self._fitted[horizon_m] = fitted

    def _blend_with_base_rate(self, model_p: float, horizon_m: int) -> tuple[float, float]:
        """Shrink the model probability toward the unconditional base rate.

        The weight falls with the horizon (strategy.SIGNAL_WEIGHT): at 3 months
        the indicators genuinely see the near future; at 18 months no indicator
        has demonstrated that reach, and a model claiming 80% at 18 months is
        claiming precision the historical record does not support. This makes the
        long-horizon numbers less dramatic and more defensible.
        """
        w = st.SIGNAL_WEIGHT.get(horizon_m, 0.6)
        base = st.HORIZON_BASE_RATE.get(horizon_m, 0.15)
        return w * model_p + (1 - w) * base, w

    def forecast_recession_probability(self, horizon_m: int) -> RecessionForecast:
        self._prepare(horizon_m)
        name, reason = self._selected.get(horizon_m, ("rule", ""))
        fitted = self._fitted.get(horizon_m)
        base = st.HORIZON_BASE_RATE.get(horizon_m, 0.15)

        if fitted is None or self.X.empty:
            return RecessionForecast(
                horizon_m, base, base, "base_rate", False, base, 0.0,
                note=("No model could be fitted (insufficient data); the unconditional "
                      "historical base rate is reported instead of a forecast."))

        latest = self.X.iloc[[-1]].fillna(self.X.median(numeric_only=True)).fillna(0.0)
        raw = float(np.clip(fitted.predict(latest), PROB_FLOOR, PROB_CEIL))
        blended, w = self._blend_with_base_rate(raw, horizon_m)

        rep = self._calibrators.get(horizon_m)
        if rep is not None and rep.calibrator.fitted and rep.improved():
            prob, calibrated = rep.calibrator.one(blended), True
            note = f"{reason}. {rep.calibrator.note}."
        else:
            prob, calibrated = blended, False
            why = (rep.calibrator.note if rep else "no calibration report")
            if rep is not None and rep.calibrator.fitted and not rep.improved():
                why = (f"{rep.calibrator.method} calibration was fitted but did not improve "
                       f"the Brier score ({rep.brier_raw} -> {rep.brier_calibrated}), so the "
                       f"uncalibrated probability is reported")
            note = f"{reason}. UNCALIBRATED: {why}."

        wf = self._wf.get(horizon_m, {}).get(fitted.name)
        return RecessionForecast(
            horizon_m=horizon_m, raw_score=round(raw, 4), probability=round(float(prob), 4),
            model=fitted.name, calibrated=calibrated, base_rate=base, signal_weight=w,
            auc=wf.auc if wf else None, brier=wf.brier if wf else None,
            drivers=self._recession_drivers(fitted), note=note)

    def _recession_drivers(self, fitted: models.FittedModel) -> list[dict]:
        """Which features are pushing this forecast, and in which direction.

        Importance says how much a feature matters to the model; the feature's
        own current signal says which way it is pushing. Reporting importance
        alone would list the same features every month regardless of the
        economy, which is not an explanation.
        """
        importance = models.feature_importance(fitted)
        out = []
        for sid, imp in list(importance.items())[:8]:
            frame = self.frames.get(sid)
            snap = frame.at() if frame else {}
            sig = snap.get("signal")
            spec = cat.BY_ID.get(sid)
            out.append({
                "sid": sid,
                "label": spec.label if spec else sid,
                "importance": imp,
                "signal": None if sig is None else round(float(sig), 4),
                "pushing": ("toward recession" if (sig is not None and sig < -0.1)
                            else "away from recession" if (sig is not None and sig > 0.1)
                            else "neutral"),
                "percentile": snap.get("percentile"),
            })
        return out

    def all_horizons(self) -> list[RecessionForecast]:
        return [self.forecast_recession_probability(h) for h in st.HORIZONS]

    def model_comparison(self, horizon_m: int = 12) -> dict:
        """The Model Performance page's table (spec 15, 28 page 9)."""
        self._prepare(horizon_m)
        results = self._wf.get(horizon_m, {})
        name, reason = self._selected.get(horizon_m, ("rule", ""))
        rep = self._calibrators.get(horizon_m)
        return {
            "horizon_m": horizon_m,
            "selected": name,
            "selection_reason": reason,
            "models": [
                {"model": n, "auc": r.auc, "brier": r.brier, "folds": r.n_folds,
                 "note": r.note} for n, r in results.items()],
            "calibration": rep.to_dict() if rep else None,
            "features": list(self.X.columns),
            "note": ("Walk-forward, time-ordered, with the last `horizon` months purged "
                     "from every training fold so overlapping outcome windows cannot leak "
                     "the answer backwards across the split."),
        }

    # -- numeric variable forecasts ---------------------------------------
    def _factor_frame(self) -> pd.DataFrame:
        """Monthly history of the nine factor scores -- the regressors.

        Uses the leading/coincident/lagging role indices plus the curve spread
        rather than re-deriving every factor at every historical month, which
        would be an O(months x series) recomputation for a marginal gain in
        fidelity.
        """
        from modules import leading
        cols = {
            "leading": leading.calculate_leading_indicator_index(self.frames),
            "coincident": leading.calculate_coincident_index(self.frames),
            "lagging": leading.calculate_lagging_index(self.frames),
        }
        for sid, name in [("T10Y3M", "curve"), ("BAMLH0A0HYM2", "hy_spread"),
                          ("NFCI", "fin_conditions"), ("DFF", "policy_rate"),
                          ("CPILFESL", "core_cpi")]:
            f = self.frames.get(sid)
            if f is None or f.raw.empty:
                continue
            base = f.transformed if sid == "CPILFESL" else f.raw
            cols[name] = base.dropna().resample("ME").last()
        df = pd.concat(cols, axis=1).sort_index()
        return df.ffill(limit=3)

    def _forecast_numeric(self, sid: str, horizon_m: int, *, unit: str,
                          label: str, use_transform: bool = True,
                          invert_direction: bool = False) -> VariableForecast:
        """Ridge regression of the target's value `horizon_m` months ahead on
        today's factor history, with a walk-forward residual interval."""
        frame = self.frames.get(sid)
        if frame is None or frame.raw.empty:
            return VariableForecast(sid, label, horizon_m, "UNKNOWN", None, None, None,
                                    unit, 0.0, note=f"{sid} unavailable")

        target = (frame.transformed if use_transform else frame.raw).dropna()
        target = target.resample("ME").last()
        Xf = self._factor_frame()
        common = Xf.index.intersection(target.index)
        Xf, target = Xf.loc[common], target.loc[common]

        # Predict the value h months AHEAD from today's factors.
        y_fwd = target.shift(-horizon_m)
        data = pd.concat([Xf, y_fwd.rename("_y")], axis=1).dropna()
        if len(data) < 80:
            return VariableForecast(sid, label, horizon_m, "UNKNOWN", None, None, None,
                                    unit, 0.0,
                                    note=f"only {len(data)} aligned observations; too few to fit")

        Xd, yd = data.drop(columns="_y"), data["_y"]

        # Walk-forward residuals -> honest interval width.
        residuals: list[float] = []
        step = 12
        start = max(60, len(Xd) // 3)
        while start < len(Xd):
            stop = min(start + step, len(Xd))
            tr = Xd.index[:start]
            # Purge: rows within horizon_m of the test block share its future.
            tr = tr[tr < Xd.index[start] - pd.DateOffset(months=horizon_m)]
            if len(tr) < 40:
                start = stop
                continue
            try:
                m = Ridge(alpha=1.0).fit(Xd.loc[tr], yd.loc[tr])
                pred = m.predict(Xd.iloc[start:stop])
                residuals.extend((pred - yd.iloc[start:stop].to_numpy()).tolist())
            except Exception:
                pass
            start = stop

        rmse = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None

        try:
            final = Ridge(alpha=1.0).fit(Xd, yd)
            latest = Xf.dropna().iloc[[-1]]
            point = float(final.predict(latest[Xd.columns])[0])
        except Exception as e:
            return VariableForecast(sid, label, horizon_m, "UNKNOWN", None, None, None,
                                    unit, 0.0, note=f"fit failed: {e}")

        current = float(target.iloc[-1])
        delta = point - current
        # A move smaller than a third of the model's own historical error is not
        # a forecast of change, it is noise inside the error bar.
        eps = (rmse or abs(current) * 0.1) / 3.0
        if abs(delta) < eps:
            direction = "FLAT"
        else:
            direction = "UP" if delta > 0 else "DOWN"
        if invert_direction and direction in ("UP", "DOWN"):
            direction = "DOWN" if direction == "UP" else "UP"

        lo = hi = None
        if rmse is not None:
            lo, hi = point - rmse, point + rmse   # ~1 sigma of realised error

        # Confidence falls with horizon and with the model's own error relative
        # to the variability of what it is predicting.
        spread = float(yd.std()) or 1.0
        skill = max(0.0, 1.0 - (rmse or spread) / spread)
        confidence = round(float(np.clip(skill * (1.0 - 0.03 * horizon_m), 0.0, 0.95)), 3)

        return VariableForecast(
            target=sid, label=label, horizon_m=horizon_m, direction=direction,
            point=round(point, 3), lo=None if lo is None else round(lo, 3),
            hi=None if hi is None else round(hi, 3), unit=unit, confidence=confidence,
            rmse=None if rmse is None else round(rmse, 3), n_train=len(Xd),
            drivers=self._numeric_drivers(final, Xd.columns),
            note=("Interval is +/-1 walk-forward RMSE (realised historical error), not an "
                  "in-sample regression interval. Current value "
                  f"{round(current, 3)}{unit}."))

    @staticmethod
    def _numeric_drivers(model, columns) -> list[str]:
        pairs = sorted(zip(columns, model.coef_), key=lambda kv: -abs(kv[1]))[:3]
        return [f"{name} ({coef:+.2f})" for name, coef in pairs]

    # -- the spec 14 surface ---------------------------------------------
    def forecast_growth(self, horizon_m: int = 12) -> VariableForecast:
        return self._forecast_numeric("INDPRO", horizon_m, unit="% YoY",
                                      label="Industrial Production (growth proxy)")

    def forecast_gdp(self, horizon_m: int = 12) -> VariableForecast:
        """GDP is proxied by real personal consumption YoY.

        PCE is ~68% of GDP, is monthly rather than quarterly, and is published
        with a 30-day lag instead of a quarter -- so it carries most of GDP's
        signal at four times the frequency and a fraction of the delay.
        """
        f = self._forecast_numeric("PCEC96", horizon_m, unit="% YoY",
                                   label="Real Consumption (GDP proxy)")
        f.note += (" GDP is proxied by real PCE (~68% of GDP, monthly, 30-day lag) rather "
                   "than quarterly GDP itself.")
        return f

    def forecast_inflation(self, horizon_m: int = 12) -> VariableForecast:
        return self._forecast_numeric("CPIAUCSL", horizon_m, unit="% YoY", label="Headline CPI")

    def forecast_core_inflation(self, horizon_m: int = 12) -> VariableForecast:
        return self._forecast_numeric("CPILFESL", horizon_m, unit="% YoY", label="Core CPI")

    def forecast_unemployment(self, horizon_m: int = 12) -> VariableForecast:
        return self._forecast_numeric("UNRATE", horizon_m, unit="%", label="Unemployment Rate",
                                      use_transform=False)

    def forecast_policy_rate(self, horizon_m: int = 12) -> VariableForecast:
        return self._forecast_numeric("DFF", horizon_m, unit="%", label="Fed Funds Rate",
                                      use_transform=False)

    def forecast_long_term_rates(self, horizon_m: int = 12) -> VariableForecast:
        return self._forecast_numeric("DGS10", horizon_m, unit="%", label="10Y Treasury Yield",
                                      use_transform=False)

    def forecast_housing(self, horizon_m: int = 12) -> VariableForecast:
        return self._forecast_numeric("HOUST", horizon_m, unit="% YoY", label="Housing Starts")

    def forecast_consumer(self, horizon_m: int = 12) -> VariableForecast:
        return self._forecast_numeric("RRSFS", horizon_m, unit="% YoY", label="Real Retail Sales")

    def forecast_credit(self, horizon_m: int = 12) -> VariableForecast:
        return self._forecast_numeric("BAMLH0A0HYM2", horizon_m, unit="%",
                                      label="High Yield Spread", use_transform=False,
                                      invert_direction=False)

    def all_variables(self, horizons=(3, 6, 12)) -> dict[str, list[VariableForecast]]:
        """Every spec 14 forecast at the requested horizons."""
        fns = {
            "growth": self.forecast_growth,
            "gdp": self.forecast_gdp,
            "inflation": self.forecast_inflation,
            "core_inflation": self.forecast_core_inflation,
            "unemployment": self.forecast_unemployment,
            "policy_rate": self.forecast_policy_rate,
            "long_rates": self.forecast_long_term_rates,
            "housing": self.forecast_housing,
            "consumer": self.forecast_consumer,
            "credit": self.forecast_credit,
        }
        out: dict[str, list[VariableForecast]] = {}
        for name, fn in fns.items():
            out[name] = []
            for h in horizons:
                try:
                    out[name].append(fn(h))
                except Exception as e:
                    log_warning(f"[forecast] {name}@{h}m failed: {e}")
        return out
