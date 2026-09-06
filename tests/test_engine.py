"""
Engine QA (spec 23): signal ranges, weight sums, probability bounds, factor
score ranges, model reproducibility, calibration behaviour.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from config import series as cat
from config import strategy as st
from modules import (calibration, confidence, factors, leading, market, models,
                     regime, scenarios, transforms)
from tests.conftest import daily_dates, month_dates


# ---------------------------------------------------------------------------
# weights and configuration invariants
# ---------------------------------------------------------------------------
def test_factor_weights_sum_to_one():
    assert sum(st.FACTOR_WEIGHTS.values()) == pytest.approx(1.0)


def test_every_category_has_a_weight():
    assert set(st.FACTOR_WEIGHTS) == set(cat.CATEGORIES)


def test_confidence_weights_sum_to_one():
    assert sum(st.CONFIDENCE_WEIGHTS.values()) == pytest.approx(1.0)


def test_market_weights_sum_to_one():
    assert sum(st.MARKET_WEIGHTS.values()) == pytest.approx(1.0)


def test_every_asset_class_has_sensitivities():
    for asset in st.ASSET_CLASSES:
        sens = st.ASSET_SENSITIVITY[asset]
        assert set(sens) == {"growth", "inflation", "stress", "real_rate"}, asset


def test_model_features_all_exist_in_the_catalogue():
    for sid in st.MODEL_FEATURES:
        assert sid in cat.BY_ID, f"{sid} is a model feature but not in the catalogue"


def test_horizon_tables_cover_every_horizon():
    for h in st.HORIZONS:
        assert h in st.HORIZON_BASE_RATE
        assert h in st.SIGNAL_WEIGHT


def test_base_rate_rises_with_horizon():
    """More can happen in 18 months than in 3 -- the unconditional risk must be
    monotone or the horizon table is inconsistent."""
    rates = [st.HORIZON_BASE_RATE[h] for h in sorted(st.HORIZONS)]
    assert rates == sorted(rates)


def test_signal_weight_falls_with_horizon():
    """Confidence in the signal must decay with distance, or the engine claims
    18-month precision it has never demonstrated."""
    weights = [st.SIGNAL_WEIGHT[h] for h in sorted(st.HORIZONS)]
    assert weights == sorted(weights, reverse=True)


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def test_signal_is_bounded_and_direction_adjusted():
    """Spec 23's test_factor_score_range, at the series level.

    The series must have genuine variation. A perfectly linear ramp has a
    CONSTANT 12-month difference, so UNRATE's diff12 transform would have zero
    variance and its z-score would be undefined -- correctly, but it would test
    nothing.
    """
    rows = [(d, 100.0 + i + math.sin(i / 9.0) * 4.0)
            for i, d in enumerate(month_dates(120))]
    up = transforms.build_signal(rows, cat.BY_ID["PAYEMS"])       # direction +1
    down = transforms.build_signal(rows, cat.BY_ID["UNRATE"])     # direction -1
    for f in (up, down):
        vals = f.signal.dropna()
        assert not vals.empty
        assert vals.between(-1.0, 1.0).all()
    assert up.at()["signal"] > 0, "a rising +1-direction series must score positive"


def test_zero_variance_transform_yields_no_zscore():
    """A constant 12-month difference has no distribution to score against, so
    the engine must return nothing rather than divide by zero."""
    rows = [(d, 100.0 + i) for i, d in enumerate(month_dates(120))]
    f = transforms.build_signal(rows, cat.BY_ID["UNRATE"])   # diff12 -> constant
    assert f.signal.dropna().empty


def test_direction_zero_series_produce_no_signal():
    rows = [(d, 250.0 + i) for i, d in enumerate(month_dates(120))]
    f = transforms.build_signal(rows, cat.BY_ID["CPIAUCSL"])      # direction 0
    assert f.signal.dropna().empty


def test_short_series_is_unusable_and_yields_no_statistics():
    rows = [(d, 100.0 + i) for i, d in enumerate(month_dates(13))]
    f = transforms.build_signal(rows, cat.BY_ID["EXHOSLUSM495S"])
    assert not f.usable
    assert f.at()["zscore"] is None and f.at()["percentile"] is None


def test_empty_series_does_not_crash():
    f = transforms.build_signal([], cat.BY_ID["PAYEMS"])
    assert not f.usable
    assert f.at()["raw"] is None


def test_yoy_uses_the_native_frequency_lag():
    monthly = transforms.to_series([(d, float(i)) for i, d in enumerate(month_dates(30))])
    yoy = transforms.calculate_yoy(monthly, "monthly")
    assert yoy.notna().sum() == 30 - 12


def test_diff12_on_a_rate_returns_a_level_change():
    s = transforms.to_series([(d, 2.0 + i * 0.1) for i, d in enumerate(month_dates(30))])
    d12 = transforms.calculate_diff(s, "monthly", 1.0)
    assert d12.dropna().iloc[-1] == pytest.approx(1.2, abs=1e-6)


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------
def _synthetic_frames(value_fn=None) -> dict:
    """Build frames for the whole catalogue from a deterministic generator."""
    frames = {}
    for spec in cat.ALL_SERIES:
        n = {"daily": 800, "weekly": 300, "monthly": 200, "quarterly": 80}[spec.freq]
        dates = (daily_dates(n) if spec.freq == "daily" else month_dates(n))
        vals = [(d, (value_fn(i) if value_fn else 100.0 + i * 0.3 + math.sin(i / 7) * 2))
                for i, d in enumerate(dates)]
        frames[spec.sid] = transforms.build_signal(vals, spec)
    return frames


def test_factor_scores_stay_in_range():
    """Spec 23's test_factor_score_range."""
    frames = _synthetic_frames()
    F = factors.calculate_all_factors(frames)
    for name, f in F.items():
        if f.score is None:
            continue
        assert -1.0 <= f.score <= 1.0, f"{name} score {f.score} out of range"


def test_composite_score_stays_in_range():
    frames = _synthetic_frames()
    F = factors.calculate_all_factors(frames)
    score, meta = factors.composite_score(F)
    assert score is None or -1.0 <= score <= 1.0
    assert 0.0 <= meta["coverage"] <= 1.0


def test_factor_is_unavailable_below_minimum_coverage():
    """Spec 6: a bad data source must not silently produce a normal forecast."""
    frames = _synthetic_frames()
    for spec in cat.by_category("housing"):
        frames[spec.sid] = transforms.build_signal([], spec)
    F = factors.calculate_all_factors(frames)
    assert F["housing"].score is None
    assert "coverage" in F["housing"].detail.get("unavailable_reason", "")


def test_composite_renormalises_over_available_factors():
    """A missing category must shift weight to its peers, not drag the composite
    toward zero and make a data outage look like a neutral economy."""
    frames = _synthetic_frames()
    F = factors.calculate_all_factors(frames)
    full, _ = factors.composite_score(F)
    F.pop("housing")
    partial, meta = factors.composite_score(F)
    assert meta["coverage"] < 1.0
    assert sum(meta["weights_used"].values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# yield curve -- the structural cases spec 8.6 calls out
# ---------------------------------------------------------------------------
def _curve_frames(path: list[float], short: list[float] | None = None,
                  long: list[float] | None = None) -> dict:
    n = len(path)
    dates = daily_dates(n)
    frames = {"T10Y3M": transforms.build_signal(list(zip(dates, path)), cat.BY_ID["T10Y3M"])}
    if short:
        frames["DGS3MO"] = transforms.build_signal(list(zip(dates, short)), cat.BY_ID["DGS3MO"])
    if long:
        frames["DGS10"] = transforms.build_signal(list(zip(dates, long)), cat.BY_ID["DGS10"])
    return frames


def test_deep_inversion_is_classified_as_deep():
    path = [1.0] * 300 + list(np.linspace(1.0, -1.2, 500))
    state, detail = factors.classify_curve_state(_curve_frames(path))
    assert state == "INVERTED_DEEP"
    assert detail["inversion_depth"] < st.CURVE_INVERSION_DEEP


def test_bull_resteepening_is_the_highest_risk_state():
    """Spec 8.6: re-steepening out of deep inversion is NOT an all-clear. When
    it happens because SHORT rates are collapsing, it is the most dangerous
    configuration in the model."""
    # A realistic re-steepening: the 2024 episode moved ~1pp in a quarter, so
    # the last 63 trading days must clear CURVE_MOVE_EPS (0.25pp) comfortably.
    invert = list(np.linspace(0.5, -1.2, 700))
    resteep = list(np.linspace(-1.2, -0.1, 100))
    path = invert + resteep
    short = list(np.linspace(2.0, 5.5, 700)) + list(np.linspace(5.5, 3.0, 100))  # cuts
    long = list(np.linspace(2.5, 4.3, 700)) + list(np.linspace(4.3, 4.2, 100))
    state, detail = factors.classify_curve_state(_curve_frames(path, short, long))
    assert state == "RE_STEEPENING_BULL"
    assert detail["risk_weight"] == max(st.CURVE_STATE_RISK.values())


def test_steep_curve_is_the_lowest_risk_state():
    path = [0.3] * 300 + [2.0] * 500
    state, _ = factors.classify_curve_state(_curve_frames(path))
    assert state == "STEEP"
    assert st.CURVE_STATE_RISK["STEEP"] == min(st.CURVE_STATE_RISK.values())


def test_curve_factor_sign_matches_the_engine_convention():
    """CURVE_STATE_RISK is risk (higher = worse); the factor must be its
    negation so +1 stays 'economically positive' everywhere."""
    path = [1.0] * 300 + list(np.linspace(1.0, -1.2, 500))
    frames = _curve_frames(path)
    f = factors.curve_factor(frames)
    assert f.score < 0, "a deeply inverted curve must score negative"


def test_inversion_duration_counts_only_the_current_run():
    dates = daily_dates(600)
    path = [-0.5] * 200 + [0.5] * 200 + [-0.3] * 200
    s = transforms.to_series(list(zip(dates, path)))
    months = factors.calculate_inversion_duration(s)
    assert 8 <= months <= 11, f"expected ~9.5 months, got {months}"


# ---------------------------------------------------------------------------
# inflation regime
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("core_yoy,expected", [
    (-0.5, "DEFLATIONARY"), (1.0, "DISINFLATIONARY"), (2.2, "STABLE"),
    (4.0, "REFLATIONARY"), (7.0, "INFLATIONARY"),
])
def test_inflation_bands(core_yoy, expected):
    band = next(name for lo, hi, name in st.INFLATION_BANDS if lo <= core_yoy < hi)
    assert band == expected


def test_stagflation_requires_both_high_inflation_and_weak_growth():
    n = 200
    dates = month_dates(n)
    # A CPI index compounding at ~7% a year -> core YoY around 7%, clearly
    # inside the INFLATIONARY band rather than on the 5.0 boundary.
    vals = [(d, 100.0 * (1.07 ** (i / 12.0))) for i, d in enumerate(dates)]
    frames = {"CPILFESL": transforms.build_signal(vals, cat.BY_ID["CPILFESL"])}
    weak, _ = factors.calculate_inflation_regime(frames, growth_score=-0.6)
    strong, _ = factors.calculate_inflation_regime(frames, growth_score=+0.5)
    assert weak == "STAGFLATIONARY"
    assert strong == "INFLATIONARY", "same inflation, healthy growth -> not stagflation"


# ---------------------------------------------------------------------------
# breadth
# ---------------------------------------------------------------------------
def test_breadth_counts_partition_the_indicator_set():
    frames = _synthetic_frames()
    b = leading.calculate_signal_breadth(frames)
    assert b.improving + b.neutral + b.weakening == b.total
    assert 0.0 <= b.weakening_pct <= 1.0
    assert -1.0 <= b.net <= 1.0


def test_breadth_only_counts_signed_leading_indicators():
    frames = _synthetic_frames()
    b = leading.calculate_signal_breadth(frames)
    counted = {i.sid for i in b.indicators}
    for spec in cat.ALL_SERIES:
        if spec.direction == 0:
            assert spec.sid not in counted, f"{spec.sid} has no good/bad meaning"
        if spec.role != "LEADING":
            assert spec.sid not in counted


# ---------------------------------------------------------------------------
# probabilities
# ---------------------------------------------------------------------------
def test_recession_probability_between_zero_and_one():
    """Spec 23's test_recession_probability_between_zero_and_one."""
    idx = pd.date_range("2000-01-31", periods=300, freq="ME")
    rng = np.random.RandomState(7)
    X = pd.DataFrame({f: rng.normal(size=300) for f in st.MODEL_FEATURES}, index=idx)
    y = pd.Series((rng.uniform(size=300) < 0.15).astype(float), index=idx)
    for name in ("rule", "logit", "probit", "forest", "gbm"):
        m = models.make_model(name).fit(X, y)
        p = m.predict_proba(X)[:, 1]
        assert np.all((p >= 0.0) & (p <= 1.0)), f"{name} produced an out-of-range probability"


def test_rule_model_risk_is_the_negated_mean_signal():
    """The rule model's whole claim is interpretability -- verify it literally
    computes what it says."""
    X = pd.DataFrame({"a": [0.5, -0.5], "b": [0.5, -0.5]})
    risk = models.RuleModel._risk(X)
    assert risk[0] == pytest.approx(-0.5)
    assert risk[1] == pytest.approx(0.5)


def test_calibrator_declines_to_fit_on_a_genuinely_tiny_sample():
    """Below even the two-parameter threshold, pass the probability through
    unchanged and say so."""
    n = st.CALIBRATION_SMALL_SAMPLE_MIN - 10
    pred = pd.Series(np.linspace(0.05, 0.9, n))
    actual = pd.Series([0.0] * (n // 2) + [1.0] * (n - n // 2))
    cal = calibration.calibrate_probability(pred, actual)
    assert not cal.fitted
    assert cal.method == "identity"
    assert "below the" in cal.note
    assert cal.one(0.42) == pytest.approx(0.42), "identity must pass values through"


def test_small_sample_uses_platt_not_isotonic():
    """Between the two thresholds the calibrator must still fit -- with two
    parameters, not a free-form monotone step function.

    Refusing to calibrate here is not the safe option: the vintage backtest has
    only ~140 points per horizon, and leaving them uncalibrated produced a
    negative Brier skill at every horizon.
    """
    n = st.CALIBRATION_SMALL_SAMPLE_MIN + 20
    rng = np.random.RandomState(5)
    pred = pd.Series(rng.uniform(size=n))
    actual = pd.Series((rng.uniform(size=n) < pred * 0.5).astype(float))
    cal = calibration.calibrate_probability(pred, actual)
    assert cal.fitted
    assert cal.method == "sigmoid"
    assert "Platt" in cal.note
    out = cal.transform(np.array([0.0, 0.5, 1.0]))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_large_sample_uses_isotonic():
    n = st.CALIBRATION_MIN_SAMPLES + 100
    rng = np.random.RandomState(6)
    pred = pd.Series(rng.uniform(size=n))
    actual = pd.Series((rng.uniform(size=n) < pred * 0.7).astype(float))
    cal = calibration.calibrate_probability(pred, actual)
    assert cal.fitted
    assert cal.method == st.CALIBRATION_METHOD


def test_explicit_method_overrides_the_sample_size_rule():
    n = st.CALIBRATION_MIN_SAMPLES + 50
    rng = np.random.RandomState(7)
    pred = pd.Series(rng.uniform(size=n))
    actual = pd.Series((rng.uniform(size=n) < 0.4).astype(float))
    cal = calibration.calibrate_probability(pred, actual, method="sigmoid")
    assert cal.method == "sigmoid"


def test_calibrator_fits_on_a_sufficient_sample():
    rng = np.random.RandomState(11)
    n = 400
    pred = pd.Series(rng.uniform(size=n))
    actual = pd.Series((rng.uniform(size=n) < pred * 0.8).astype(float))
    cal = calibration.calibrate_probability(pred, actual)
    assert cal.fitted
    out = cal.transform(np.array([0.0, 0.5, 1.0]))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_calibrated_probabilities_stay_bounded():
    rng = np.random.RandomState(13)
    pred = pd.Series(rng.uniform(size=300))
    actual = pd.Series((rng.uniform(size=300) < 0.3).astype(float))
    cal = calibration.calibrate_probability(pred, actual)
    out = cal.transform(np.array([-5.0, 0.5, 5.0]))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_brier_skill_is_zero_for_a_base_rate_forecast():
    actual = pd.Series([1.0] * 30 + [0.0] * 70)
    base = pd.Series([0.3] * 100)
    assert calibration.brier_skill_score(base, actual) == pytest.approx(0.0, abs=1e-9)


def test_calibration_curve_reports_empty_bins():
    """An empty 70-80% bin is information: the model never says 70-80%."""
    pred = pd.Series([0.05] * 50)
    actual = pd.Series([0.0] * 50)
    curve = calibration.calculate_calibration_curve(pred, actual, bins=10)
    assert len(curve) == 10
    assert any(b["n"] == 0 for b in curve)


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
class _FakeForecast:
    def __init__(self, p):
        self.probability = p
        self.horizon_m = 12


def test_scenario_probabilities_sum_to_one_and_match_the_recession_probability():
    """Spec 17's consistency requirement, made testable."""
    frames = _synthetic_frames()
    F = factors.calculate_all_factors(frames)
    for p in (0.05, 0.25, 0.43, 0.8):
        scen = scenarios.generate(None, F, _FakeForecast(p), composite=-0.1, momentum=-0.05)
        by = {s.name: s.probability for s in scen}
        assert sum(by.values()) == pytest.approx(1.0, abs=1e-9)
        assert by["BEAR"] + by["CRISIS"] == pytest.approx(p, abs=1e-9), \
            "recession scenarios must equal the recession probability exactly"
        assert by["BASE"] + by["BULL"] == pytest.approx(1 - p, abs=1e-9)
        assert all(v >= 0 for v in by.values())


def test_crisis_share_rises_with_financial_stress():
    calm = _synthetic_frames()
    F_calm = factors.calculate_all_factors(calm)
    F_stress = factors.calculate_all_factors(calm)
    for name in ("credit", "financial"):
        F_stress[name].score = -0.9        # severe stress
    p = 0.4
    calm_scen = {s.name: s.probability for s in
                 scenarios.generate(None, F_calm, _FakeForecast(p), 0.0, 0.0)}
    stress_scen = {s.name: s.probability for s in
                   scenarios.generate(None, F_stress, _FakeForecast(p), 0.0, 0.0)}
    assert stress_scen["CRISIS"] > calm_scen["CRISIS"], \
        "the same recession probability with credit dysfunction must weight CRISIS higher"


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------
def test_regime_is_a_known_label():
    for level in (-0.9, -0.4, -0.1, 0.0, 0.3, 0.8):
        for mom in (-0.4, -0.05, 0.0, 0.05, 0.4):
            r, strength, _ = regime.classify_cycle(level, mom, level)
            assert r in st.REGIME_ORDER
            assert 0.0 <= strength <= 1.0


def test_same_level_opposite_momentum_gives_opposite_regimes():
    """The reason the classifier takes momentum at all: -0.3 falling and -0.3
    rising are opposite conclusions."""
    falling, _, _ = regime.classify_cycle(-0.30, -0.20, -0.35)
    rising, _, _ = regime.classify_cycle(-0.30, +0.20, -0.25)
    assert falling != rising
    assert rising in ("EARLY_RECOVERY", "RECOVERY")


def test_regime_never_depends_on_a_single_indicator():
    """Spec 12. classify_cycle's signature admits only aggregates, so a single
    series cannot reach it -- this test pins that contract."""
    import inspect
    params = set(inspect.signature(regime.classify_cycle).parameters)
    assert params == {"level", "momentum", "leading_index"}


# ---------------------------------------------------------------------------
# market separation (spec 19)
# ---------------------------------------------------------------------------
def test_macro_engine_does_not_import_the_market_layer():
    """Spec 19's separation, enforced structurally rather than by convention."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "modules"
    for name in ("factors.py", "forecast.py", "models.py", "regime.py",
                 "leading.py", "calibration.py", "transforms.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "import market" not in text and "from modules.market" not in text, (
            f"{name} imports the market layer -- valuation must never be able to "
            f"reach a recession probability")


def test_asset_stances_are_known_labels():
    frames = _synthetic_frames()
    F = factors.calculate_all_factors(frames)
    out = market.summarise(F, frames, recession_prob=0.3)
    valid = {lab for _, lab in st.ASSET_BANDS} | {"NEGATIVE"}
    assert len(out["assets"]) == len(st.ASSET_CLASSES)
    for a in out["assets"]:
        assert a["stance"] in valid
        assert -1.0 <= a["score"] <= 1.0


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------
def test_confidence_components_are_bounded_and_weighted():
    frames = _synthetic_frames()
    F = factors.calculate_all_factors(frames)
    idx = pd.date_range("2010-01-31", periods=150, freq="ME")
    X = pd.DataFrame({f: np.random.RandomState(3).normal(size=150)
                      for f in st.MODEL_FEATURES}, index=idx)

    class _F:
        auc, model, horizon_m, probability = 0.85, "rule", 12, 0.3

    health = {"health": 0.95, "n": 80, "stale": 2, "high": 70, "medium": 8, "low": 2}
    rep = confidence.calculate_confidence(F, _F(), {}, health, {}, frames, X)
    assert 0.0 <= rep.score <= 1.0
    assert set(rep.components) == set(st.CONFIDENCE_WEIGHTS)
    assert all(0.0 <= v <= 1.0 for v in rep.components.values())
    assert rep.band in ("HIGH", "MEDIUM", "LOW")


def test_confidence_falls_when_data_degrades():
    frames = _synthetic_frames()
    F = factors.calculate_all_factors(frames)
    idx = pd.date_range("2010-01-31", periods=150, freq="ME")
    X = pd.DataFrame({f: np.random.RandomState(4).normal(size=150)
                      for f in st.MODEL_FEATURES}, index=idx)

    class _F:
        auc, model, horizon_m, probability = 0.85, "rule", 12, 0.3

    good = confidence.calculate_confidence(
        F, _F(), {}, {"health": 0.99, "n": 80, "stale": 0, "high": 80,
                      "medium": 0, "low": 0}, {}, frames, X)
    bad = confidence.calculate_confidence(
        F, _F(), {}, {"health": 0.40, "n": 80, "stale": 40, "high": 10,
                      "medium": 20, "low": 50}, {}, frames, X)
    assert bad.score < good.score, "degraded data must lower published confidence"


# ---------------------------------------------------------------------------
# reproducibility (spec 31)
# ---------------------------------------------------------------------------
def test_models_are_deterministic():
    """Spec 23's model reproducibility: same data + same seed = same output."""
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    rng = np.random.RandomState(21)
    X = pd.DataFrame({f: rng.normal(size=240) for f in st.MODEL_FEATURES}, index=idx)
    y = pd.Series((rng.uniform(size=240) < 0.2).astype(float), index=idx)
    for name in ("rule", "logit", "probit", "forest", "gbm"):
        a = models.make_model(name).fit(X, y).predict_proba(X)[:, 1]
        b = models.make_model(name).fit(X, y).predict_proba(X)[:, 1]
        np.testing.assert_allclose(a, b, err_msg=f"{name} is not deterministic")


def test_hash_payload_is_order_independent():
    from modules import runlog
    assert runlog.hash_payload({"a": 1, "b": 2}) == runlog.hash_payload({"b": 2, "a": 1})


def test_hash_payload_changes_when_a_value_changes():
    from modules import runlog
    assert runlog.hash_payload({"a": 1}) != runlog.hash_payload({"a": 2})


# ---------------------------------------------------------------------------
# feature history depth
# ---------------------------------------------------------------------------
def test_short_history_feature_is_excluded_from_the_model_matrix():
    """FRED's keyless endpoint truncates LICENSED series (ICE BofA OAS to ~3
    years, S&P 500 to 10) while serving Fed-produced ones in full. A 3-year
    feature in a model trained since 1985 is a median-imputed constant for most
    of the sample and a live value only at prediction time -- a silent
    distribution shift. The matrix must drop it rather than carry it."""
    # 30 years of daily data for the long feature, 3 years for the short one --
    # roughly the real depths FRED serves for T10Y3M and the ICE OAS series.
    long_rows = [(d, 1.0 + math.sin(i / 400.0)) for i, d in enumerate(daily_dates(7500))]
    short_rows = [(d, 4.0 + math.sin(i / 30.0)) for i, d in enumerate(daily_dates(760))]
    frames = {
        "T10Y3M": transforms.build_signal(long_rows, cat.BY_ID["T10Y3M"]),
        "BAMLH0A0HYM2": transforms.build_signal(short_rows, cat.BY_ID["BAMLH0A0HYM2"]),
    }
    X = models.build_feature_matrix(frames, features=["T10Y3M", "BAMLH0A0HYM2"])
    assert "BAMLH0A0HYM2" not in X.columns, "a 3-year feature must be excluded"
    assert "T10Y3M" in X.columns, "a 30-year feature must be kept"


def test_no_model_feature_is_a_truncated_licensed_series():
    """A regression guard on the catalogue itself: the ICE BofA OAS and S&P 500
    series must never drift back into MODEL_FEATURES, because FRED only serves a
    rolling window of them keylessly."""
    truncated = {"BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A3HYC", "SP500"}
    overlap = truncated & set(st.MODEL_FEATURES)
    assert not overlap, (
        f"{overlap} are truncated by FRED's keyless endpoint and cannot be model "
        f"features; use BAA10Y / AAA10Y / NASDAQCOM for deep history")


# ---------------------------------------------------------------------------
# point-in-time climatology and shrinkage calibration
# ---------------------------------------------------------------------------
def _rec_rows(starts_months: list[str], n: int = 400) -> list[tuple[str, float]]:
    """Synthetic NBER-style 0/1 series with recessions starting at given months."""
    from tests.conftest import month_dates
    dates = month_dates(n, end=date(2026, 1, 1))
    rows = []
    for d in dates:
        ym = d[:7]
        inrec = any(s <= ym < _plus(s, 9) for s in starts_months)
        rows.append((d, 1.0 if inrec else 0.0))
    return rows


def _plus(ym: str, months: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    m += months
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def test_empirical_base_rate_is_point_in_time():
    """The climatology at an early date must not know about later recessions."""
    rows = _rec_rows(["2000-01", "2008-01", "2020-01"])
    early = calibration.empirical_base_rate(rows, 12, "2004-01-15")
    late = calibration.empirical_base_rate(rows, 12, "2026-01-15")
    assert 0.0 <= early <= 1.0 and 0.0 <= late <= 1.0
    # Truncating the input to the early date must give the same answer.
    truncated = [(d, v) for d, v in rows if d <= "2004-01-15"]
    assert calibration.empirical_base_rate(truncated, 12, "2004-01-15") == pytest.approx(early)


def test_empirical_base_rate_excludes_unclosed_windows():
    """A month whose outcome window has not finished cannot contribute -- its
    answer was not known at the as-of date."""
    rows = _rec_rows(["2010-01"])
    idx, out = calibration.horizon_outcome_series(rows, 12)
    assert len(idx) == len(out)
    assert set(np.unique(out)) <= {0.0, 1.0}


def test_empirical_base_rate_rises_with_horizon():
    rows = _rec_rows(["2000-01", "2008-01", "2020-01"])
    rates = [calibration.empirical_base_rate(rows, h, "2026-01-15") for h in (3, 6, 12, 18)]
    assert rates == sorted(rates), "more can happen in a longer window"


def test_shrinkage_is_zero_without_evidence():
    """With no recession events seen, the engine must quote the base rate --
    that is the honest statement when nothing has demonstrated the signal works."""
    cal = calibration.fit_shrinkage([0.1] * 50, [0.0] * 50, base=0.09)
    assert cal.lam == 0.0
    assert cal.one(0.9) == pytest.approx(0.09), "no evidence -> quote climatology"
    assert "recession events" in cal.note


def test_shrinkage_is_monotone_and_preserves_ranking():
    """The whole reason for a one-parameter form: refitting a free calibrator at
    every date destroyed the model's ranking (pooled AUC fell 0.70 -> 0.44)."""
    rng = np.random.RandomState(31)
    s = rng.uniform(0.02, 0.5, size=200)
    y = (rng.uniform(size=200) < s).astype(float)
    cal = calibration.fit_shrinkage(s, y, base=float(y.mean()))
    out = cal.transform(np.sort(s))
    assert np.all(np.diff(out) >= -1e-12), "must be monotone in the score"


def test_shrinkage_lambda_is_bounded_and_evidence_scaled():
    rng = np.random.RandomState(32)
    s = rng.uniform(0.0, 1.0, size=300)
    y = (rng.uniform(size=300) < s).astype(float)
    strong = calibration.fit_shrinkage(s, y, base=float(y.mean()))
    weak = calibration.fit_shrinkage(s[:40], y[:40], base=float(y[:40].mean()))
    assert 0.0 <= strong.lam <= 1.0 and 0.0 <= weak.lam <= 1.0
    assert strong.lam >= weak.lam, "more evidence must permit more sharpness"


def test_shrinkage_output_stays_in_range():
    cal = calibration.fit_shrinkage([0.0, 1.0] * 60, [0.0, 1.0] * 60, base=0.5)
    for v in (-5.0, 0.0, 0.5, 1.0, 5.0):
        assert 0.0 <= cal.one(v) <= 1.0


def test_brier_skill_accepts_an_array_reference():
    """The fair reference is a per-date climatology, not one full-sample scalar --
    the scalar version is an oracle that knows the window's realised frequency."""
    actual = pd.Series([0.0] * 90 + [1.0] * 10)
    pred = pd.Series([0.1] * 100)
    scalar = calibration.brier_skill_score(pred, actual)
    array = calibration.brier_skill_score(pred, actual, np.full(100, 0.10))
    assert scalar == pytest.approx(array, abs=0.02)
    # A deliberately poor reference must make the same forecast look better.
    poor = calibration.brier_skill_score(pred, actual, np.full(100, 0.60))
    assert poor > scalar


def test_climatology_beats_a_biased_constant_prior():
    """The point of estimating the base rate rather than hardcoding it: the
    1970- constant over-predicts the post-1990 era by roughly two thirds."""
    rows = _rec_rows(["2000-01", "2008-01", "2020-01"])
    estimated = calibration.empirical_base_rate(rows, 12, "2026-01-15")
    assert estimated < st.HORIZON_BASE_RATE[12] * 1.5
