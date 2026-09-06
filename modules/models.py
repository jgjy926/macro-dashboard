"""
Model engine (spec 15) -- rule, logit, probit, random forest, gradient boosting.

Spec 15 says plainly: "Do not assume ML is superior. Compare models through
walk-forward validation. Only use an ensemble if it improves out-of-sample
performance." This module implements that literally. Every model is scored the
same way, by the same walk-forward procedure, and the selection is made on
out-of-sample AUC -- with the rule-based model as a first-class competitor, not
a baseline to be beaten by assumption.

THE SAMPLE-SIZE PROBLEM, STATED HONESTLY
----------------------------------------
There are about 8 usable recessions since 1970. Monthly observations make the
row count look like ~670, but the effective sample is the number of independent
RECESSION EPISODES, and that is single digits. Three consequences shape
everything here:

  * The feature set is capped at ~14 (config.strategy.MODEL_FEATURES), not 78.
  * Regularisation is strong by default and trees are shallow (depth 3-4).
  * Walk-forward splits are strictly time-ordered with a purge gap. Overlapping
    horizons make adjacent rows nearly identical -- "recession within 12 months"
    on consecutive months shares 11 months of the same future -- so a random
    split would leak the answer across the fold boundary and produce an AUC of
    0.99 that means nothing. See `_purge` below.

Any model reporting near-perfect discrimination on this problem is measuring a
leak, not skill.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import strategy as st
from modules.log import log_info, log_warning
from modules.transforms import SignalFrame


# Minimum months of history for a series to be usable as a model feature.
# 20 years spans at least two recessions, which is the floor for a feature to
# contribute anything the model can generalise from.
MIN_FEATURE_MONTHS = 240


# ---------------------------------------------------------------------------
# feature matrix and target
# ---------------------------------------------------------------------------
def build_feature_matrix(frames: dict[str, SignalFrame],
                         features: list[str] | None = None) -> pd.DataFrame:
    """Monthly feature matrix from the direction-adjusted signals.

    Downsamples each series to month-end (last observation in the month) and
    forward-fills at most 3 months. The forward fill exists because quarterly
    series (SLOOS lending standards) would otherwise blank two months in three;
    it is capped so a discontinued series cannot be carried forward forever and
    quietly become a constant the model leans on.
    """
    features = features or st.MODEL_FEATURES
    cols = []
    for sid in features:
        frame = frames.get(sid)
        if frame is None or not frame.usable or frame.signal.empty:
            log_warning(f"[models] feature {sid} unavailable -- excluded from the matrix")
            continue
        s = frame.signal.dropna().resample("ME").last()
        if len(s) < MIN_FEATURE_MONTHS:
            # A feature far shorter than the training sample is worse than a
            # missing one: align() median-imputes it for every earlier row, so
            # the model trains on a constant and is then handed a live value at
            # prediction time. That is a silent distribution shift, and it is
            # exactly what FRED's licensed-series truncation would cause if this
            # guard were not here (ICE BofA OAS: 3 years; SP500: 10).
            log_warning(f"[models] feature {sid} has only {len(s)} months of history "
                        f"(minimum {MIN_FEATURE_MONTHS}) -- EXCLUDED. A feature this "
                        f"short would be a median-imputed constant across most of the "
                        f"training sample.")
            continue
        cols.append(s.rename(sid))
    if not cols:
        return pd.DataFrame()
    X = pd.concat(cols, axis=1).sort_index()
    return X.ffill(limit=3)


def build_target(recession: list[tuple[str, float]], horizon_m: int,
                 index: pd.DatetimeIndex) -> pd.Series:
    """1 if a recession STARTS within the next `horizon_m` months, else 0.

    Deliberately "starts within", not "is in recession in N months". The
    question the engine is asked is whether a downturn is coming, and a model
    trained on "currently in recession" would learn to recognise a recession
    already underway -- useful for nowcasting, useless for forecasting.

    Months already inside a recession are set to NaN and dropped from training:
    asking "will a recession start in the next 12 months" while one is running
    is not a well-posed question, and including those rows teaches the model
    that recession conditions predict no new recession.
    """
    if not recession:
        return pd.Series(dtype="float64", index=index)
    rec = pd.Series({pd.Timestamp(d): v for d, v in recession}).sort_index()
    rec = rec.resample("ME").last().ffill()

    starts = ((rec == 1) & (rec.shift(1) == 0))
    start_dates = list(rec.index[starts])

    y = pd.Series(0.0, index=index)
    for t in index:
        if t not in rec.index:
            nearest = rec.index[rec.index <= t]
            in_rec = bool(rec.loc[nearest[-1]] == 1) if len(nearest) else False
        else:
            in_rec = bool(rec.loc[t] == 1)
        if in_rec:
            y.loc[t] = np.nan
            continue
        window_end = t + pd.DateOffset(months=horizon_m)
        y.loc[t] = 1.0 if any(t < sd <= window_end for sd in start_dates) else 0.0
    return y


def align(X: pd.DataFrame, y: pd.Series, min_features: float = 0.6
          ) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows the models cannot use: missing target, or too few features.

    Rows are kept when at least `min_features` of the columns are present and
    the remainder are median-imputed, rather than requiring a complete row.
    Requiring completeness would throw away everything before the newest
    feature's start date -- JOLTS begins in 2000, which would erase the 1990
    recession from training entirely.
    """
    if X.empty or y.empty:
        return X, y
    common = X.index.intersection(y.dropna().index)
    X2, y2 = X.loc[common], y.loc[common]
    keep = X2.notna().mean(axis=1) >= min_features
    X2, y2 = X2[keep], y2[keep]
    X2 = X2.fillna(X2.median(numeric_only=True)).fillna(0.0)
    return X2, y2


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class RuleModel:
    """The expert rule-based model (spec 15 #1).

    Not a fallback -- a genuine competitor. It has one free parameter pair
    (intercept, slope) fitted by maximum likelihood on the training fold, so it
    is estimated on exactly the same data as the others but constrained to a
    single, interpretable summary of the evidence: the equal-weighted mean of
    the direction-adjusted signals. With ~8 recessions this parsimony is a real
    advantage, not a handicap, and the walk-forward comparison usually shows it.
    """

    name = "rule"

    def __init__(self):
        self.intercept = st.RULE_LOGIT_INTERCEPT
        self.slope = st.RULE_LOGIT_SLOPE

    @staticmethod
    def _risk(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = X.to_numpy(dtype="float64") if isinstance(X, pd.DataFrame) else np.asarray(X, dtype="float64")
        # Signals are +1 = good; risk is the negation, so + = risk.
        return -np.nanmean(arr, axis=1)

    def fit(self, X, y):
        risk = self._risk(X)
        yv = np.asarray(y, dtype="float64")

        def nll(params):
            a, b = params
            z = np.clip(a + b * risk, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-9
            return -np.sum(yv * np.log(p + eps) + (1 - yv) * np.log(1 - p + eps))

        res = minimize(nll, x0=[st.RULE_LOGIT_INTERCEPT, st.RULE_LOGIT_SLOPE],
                       method="Nelder-Mead")
        if res.success:
            self.intercept, self.slope = float(res.x[0]), float(res.x[1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        z = np.clip(self.intercept + self.slope * self._risk(X), -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])

    def params(self) -> dict:
        return {"intercept": self.intercept, "slope": self.slope}


class ProbitModel:
    """Probit (spec 15 #3), the standard functional form in the academic
    recession-forecasting literature (Estrella-Mishkin).

    scikit-learn has no probit and statsmodels is not a dependency, so this is a
    direct maximum-likelihood fit with a normal-CDF link, L2-penalised to match
    the logit's regularisation so the two are compared on equal footing rather
    than one being handicapped.
    """

    name = "probit"

    def __init__(self, penalty: float = 1.0 / st.LOGIT_C):
        self.penalty = penalty
        self.coef_: np.ndarray | None = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(np.asarray(X, dtype="float64"))
        Xd = np.column_stack([np.ones(len(Xs)), Xs])
        yv = np.asarray(y, dtype="float64")

        def nll(beta):
            eta = np.clip(Xd @ beta, -8, 8)
            p = np.clip(norm.cdf(eta), 1e-9, 1 - 1e-9)
            ll = np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p))
            # Penalise slopes only -- shrinking the intercept would bias the
            # base rate toward 50%, which is badly wrong for a rare event.
            return -ll + self.penalty * np.sum(beta[1:] ** 2)

        res = minimize(nll, x0=np.zeros(Xd.shape[1]), method="L-BFGS-B")
        self.coef_ = res.x
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xs = self.scaler.transform(np.asarray(X, dtype="float64"))
        eta = np.clip(np.column_stack([np.ones(len(Xs)), Xs]) @ self.coef_, -8, 8)
        p = norm.cdf(eta)
        return np.column_stack([1 - p, p])

    def params(self) -> dict:
        return {"coef": [] if self.coef_ is None else [round(float(c), 5) for c in self.coef_]}


def make_model(name: str):
    """Construct a fresh, unfitted model. Seeds are fixed (spec 31)."""
    if name == "rule":
        return RuleModel()
    if name == "probit":
        return ProbitModel()
    if name == "logit":
        return _ScaledLogit()
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=st.FOREST_TREES, max_depth=st.FOREST_MAX_DEPTH,
            min_samples_leaf=10, class_weight="balanced",
            random_state=st.RANDOM_SEED, n_jobs=-1)
    if name == "gbm":
        return GradientBoostingClassifier(
            n_estimators=st.GBM_TREES, learning_rate=st.GBM_LEARNING_RATE,
            max_depth=st.GBM_MAX_DEPTH, subsample=0.8,
            random_state=st.RANDOM_SEED)
    raise ValueError(f"unknown model {name!r}")


class _ScaledLogit:
    """Logistic regression with standardisation folded in, so coefficients are
    comparable across features and the reported drivers mean something."""

    name = "logit"

    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=st.LOGIT_C, max_iter=2000,
                                      class_weight="balanced",
                                      random_state=st.RANDOM_SEED)
        self.features: list[str] = []

    def fit(self, X, y):
        self.features = list(X.columns) if isinstance(X, pd.DataFrame) else []
        self.clf.fit(self.scaler.fit_transform(np.asarray(X, dtype="float64")),
                     np.asarray(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(np.asarray(X, dtype="float64")))

    def params(self) -> dict:
        return {"intercept": float(self.clf.intercept_[0]),
                "coef": dict(zip(self.features,
                                 (round(float(c), 5) for c in self.clf.coef_[0])))}


# ---------------------------------------------------------------------------
# walk-forward validation
# ---------------------------------------------------------------------------
def _purge(train_idx: pd.DatetimeIndex, test_start: pd.Timestamp, horizon_m: int
           ) -> pd.DatetimeIndex:
    """Remove training rows whose outcome window overlaps the test period.

    THE critical detail. The label at month t looks forward `horizon_m` months.
    A row at t = test_start - 3 months, with a 12-month horizon, has an outcome
    determined mostly by months inside the test set. Training on it leaks the
    test answer backwards. Purging the last `horizon_m` months of every training
    fold is what makes the reported AUC an honest out-of-sample number.
    """
    cutoff = test_start - pd.DateOffset(months=horizon_m)
    return train_idx[train_idx < cutoff]


@dataclass
class WalkForwardResult:
    model: str
    horizon_m: int
    predictions: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    actuals: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    n_folds: int = 0
    auc: float | None = None
    brier: float | None = None
    note: str = ""


def walk_forward(X: pd.DataFrame, y: pd.Series, model_name: str, horizon_m: int,
                 min_train_months: int | None = None,
                 step_months: int | None = None) -> WalkForwardResult:
    """Strictly time-ordered walk-forward with purging (spec 22)."""
    min_train = min_train_months or st.WALK_FORWARD_MIN_TRAIN_MONTHS
    step = step_months or st.WALK_FORWARD_STEP_MONTHS
    res = WalkForwardResult(model=model_name, horizon_m=horizon_m)
    if len(X) < min_train + step:
        res.note = f"insufficient data: {len(X)} rows, need {min_train + step}"
        return res

    preds: list[pd.Series] = []
    actuals: list[pd.Series] = []
    start = min_train
    while start < len(X):
        stop = min(start + step, len(X))
        test_idx = X.index[start:stop]
        train_idx = _purge(X.index[:start], test_idx[0], horizon_m)
        if len(train_idx) < 36:
            start = stop
            continue
        ytr = y.loc[train_idx]
        # A fold with one class cannot be fitted; skip rather than crash. This
        # is normal early in the sample, before the first recession.
        if ytr.nunique() < 2:
            start = stop
            continue
        try:
            model = make_model(model_name).fit(X.loc[train_idx], ytr)
            p = model.predict_proba(X.loc[test_idx])[:, 1]
        except Exception as e:
            log_warning(f"[walkforward] {model_name} h={horizon_m} fold failed: {e}")
            start = stop
            continue
        preds.append(pd.Series(p, index=test_idx))
        actuals.append(y.loc[test_idx])
        res.n_folds += 1
        start = stop

    if not preds:
        res.note = "no usable folds"
        return res

    res.predictions = pd.concat(preds)
    res.actuals = pd.concat(actuals)
    mask = res.actuals.notna()
    res.predictions, res.actuals = res.predictions[mask], res.actuals[mask]
    if res.actuals.nunique() > 1:
        res.auc = round(float(roc_auc_score(res.actuals, res.predictions)), 4)
    res.brier = round(float(np.mean((res.predictions - res.actuals) ** 2)), 5)
    return res


def compare_models(X: pd.DataFrame, y: pd.Series, horizon_m: int,
                   models: list[str] | None = None) -> dict[str, WalkForwardResult]:
    return {m: walk_forward(X, y, m, horizon_m) for m in (models or st.MODELS)}


def select_model(results: dict[str, WalkForwardResult]) -> tuple[str, str]:
    """Pick the published model by out-of-sample AUC (spec 15).

    Returns (name, reason). The ensemble is only considered a winner if it beats
    the best single model by ENSEMBLE_MIN_AUC_GAIN -- spec 15's "only use an
    ensemble if it improves out-of-sample performance", made into a rule rather
    than a preference.
    """
    scored = [(n, r.auc) for n, r in results.items() if r.auc is not None]
    if not scored:
        return "rule", "no model produced a scoreable walk-forward result; falling back to the rule model"
    scored.sort(key=lambda kv: -kv[1])
    best_name, best_auc = scored[0]

    if "ensemble" in results and results["ensemble"].auc is not None:
        singles = [(n, a) for n, a in scored if n != "ensemble"]
        if singles:
            top_single, top_auc = singles[0]
            if results["ensemble"].auc >= top_auc + st.ENSEMBLE_MIN_AUC_GAIN:
                return "ensemble", (
                    f"ensemble AUC {results['ensemble'].auc:.3f} beats the best single "
                    f"model ({top_single} {top_auc:.3f}) by more than the "
                    f"{st.ENSEMBLE_MIN_AUC_GAIN:.3f} threshold")
            return top_single, (
                f"{top_single} selected: ensemble AUC {results['ensemble'].auc:.3f} did not "
                f"beat it ({top_auc:.3f}) by the required {st.ENSEMBLE_MIN_AUC_GAIN:.3f}")
    return best_name, f"{best_name} had the best walk-forward AUC ({best_auc:.3f})"


def ensemble_walk_forward(results: dict[str, WalkForwardResult], horizon_m: int
                          ) -> WalkForwardResult:
    """Equal-weighted average of the ENSEMBLE_MEMBERS' out-of-sample predictions.

    Averaged on the walk-forward predictions rather than refitted, so the
    ensemble is scored on exactly the same out-of-sample points as its members
    and the comparison in select_model() is like-for-like.
    """
    members = [results[m] for m in st.ENSEMBLE_MEMBERS
               if m in results and not results[m].predictions.empty]
    out = WalkForwardResult(model="ensemble", horizon_m=horizon_m)
    if len(members) < 2:
        out.note = "fewer than two members produced predictions"
        return out
    df = pd.concat([m.predictions.rename(m.model) for m in members], axis=1).dropna()
    if df.empty:
        out.note = "members share no common prediction dates"
        return out
    out.predictions = df.mean(axis=1)
    out.actuals = members[0].actuals.reindex(out.predictions.index)
    mask = out.actuals.notna()
    out.predictions, out.actuals = out.predictions[mask], out.actuals[mask]
    out.n_folds = min(m.n_folds for m in members)
    if out.actuals.nunique() > 1:
        out.auc = round(float(roc_auc_score(out.actuals, out.predictions)), 4)
    out.brier = round(float(np.mean((out.predictions - out.actuals) ** 2)), 5)
    return out


# ---------------------------------------------------------------------------
# fitting for live prediction
# ---------------------------------------------------------------------------
@dataclass
class FittedModel:
    name: str
    horizon_m: int
    model: object
    features: list[str]
    train_rows: int
    train_end: str
    auc: float | None = None
    brier: float | None = None

    def predict(self, x: pd.DataFrame) -> float:
        return float(self.model.predict_proba(x)[:, 1][-1])

    def params(self) -> dict:
        return self.model.params() if hasattr(self.model, "params") else {
            "type": type(self.model).__name__}


def fit_final(X: pd.DataFrame, y: pd.Series, model_name: str, horizon_m: int
              ) -> FittedModel | None:
    """Fit on all available history, for predicting today.

    The last `horizon_m` months are excluded from training: their labels depend
    on months that have not happened yet, so they are not merely unknown, they
    are unknowable. Including them (with whatever the label defaults to) would
    teach the model that the present looks like whatever that default is.
    """
    if X.empty or y.empty:
        return None
    cutoff = X.index[-1] - pd.DateOffset(months=horizon_m)
    idx = X.index[X.index <= cutoff]
    ytr = y.reindex(idx).dropna()
    idx = ytr.index
    if len(idx) < 60 or ytr.nunique() < 2:
        return None
    try:
        model = make_model(model_name).fit(X.loc[idx], ytr)
    except Exception as e:
        log_warning(f"[models] final fit failed for {model_name} h={horizon_m}: {e}")
        return None
    return FittedModel(name=model_name, horizon_m=horizon_m, model=model,
                       features=list(X.columns), train_rows=len(idx),
                       train_end=idx[-1].date().isoformat())


def persist_artifact(conn, fitted: FittedModel, metrics: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO model_artifacts (model, horizon_m, train_end, features, "
        "params, metrics) VALUES (?,?,?,?,?,?)",
        (fitted.name, fitted.horizon_m, fitted.train_end, json.dumps(fitted.features),
         json.dumps(fitted.params(), default=str), json.dumps(metrics, default=str)))
    conn.commit()


def feature_importance(fitted: FittedModel) -> dict[str, float]:
    """Per-feature importance, on whatever scale the model natively provides.

    Tree importances and standardised logit coefficients are NOT the same
    quantity, so they are normalised to sum to 1 and the caller is expected to
    read them as relative ranking within one model, never across models.
    """
    m = fitted.model
    raw: dict[str, float] = {}
    if hasattr(m, "feature_importances_"):
        raw = dict(zip(fitted.features, (float(v) for v in m.feature_importances_)))
    elif isinstance(m, _ScaledLogit):
        raw = {k: abs(v) for k, v in m.params()["coef"].items()}
    elif isinstance(m, ProbitModel) and m.coef_ is not None:
        raw = dict(zip(fitted.features, (abs(float(c)) for c in m.coef_[1:])))
    elif isinstance(m, RuleModel):
        raw = {f: 1.0 for f in fitted.features}      # equal-weighted by construction
    total = sum(raw.values())
    return {k: round(v / total, 4) for k, v in sorted(raw.items(), key=lambda kv: -kv[1])} \
        if total else {}
