"""
Model parameters -- every threshold, weight and rule the engine uses.

Nothing here touches the environment, the filesystem or the network: this file
is pure numbers and labels, so a reviewer can audit the model's judgement calls
without reading any code, and a run can be reproduced from (code version +
this file). Operational settings live in config/settings.py; the data catalogue
lives in config/series.py.

Where a number is a judgement call rather than an estimate, the comment says so.
Spec 9 warns against arbitrary thresholds, and the engine's answer is that
almost everything is expressed in historical percentiles or z-scores computed
from the data itself -- the constants below are the handful of places where a
human decision genuinely has to be made.
"""
from __future__ import annotations

# ===========================================================================
# 10. FACTOR ENGINE
# ===========================================================================
# Weight of each category in the composite macro score. Ordered by how much
# independent recession information the category has historically carried:
# credit and the curve lead by the widest margin, labour and housing next,
# inflation last because it is regime-context rather than a direction.
FACTOR_WEIGHTS: dict[str, float] = {
    "credit": 0.18,
    "curve": 0.16,
    "labor": 0.15,
    "housing": 0.13,
    "consumer": 0.12,
    "manufacturing": 0.10,
    "financial": 0.08,
    "policy": 0.05,
    "inflation": 0.03,
}

# Within a category, LEADING indicators carry more weight than coincident or
# lagging ones -- a forecast built mostly on lagging data is a description.
ROLE_WEIGHTS: dict[str, float] = {
    "LEADING": 1.0,
    "COINCIDENT": 0.6,
    "LAGGING": 0.35,
    "CONTEXT": 0.0,
}

# A category needs this share of its series present before its score is
# published. Below it the factor is reported as UNAVAILABLE rather than being
# computed from whichever two series happened to survive -- spec 6's rule that
# bad data must not silently produce a normal forecast.
MIN_FACTOR_COVERAGE = 0.5

# Months over which factor momentum (the dashboard's arrows) is measured.
MOMENTUM_MONTHS = 3

# Arrow thresholds, in units of factor score change over MOMENTUM_MONTHS.
TREND_ARROWS = [(0.30, "^^^"), (0.15, "^^"), (0.05, "^"),
                (-0.05, "->"), (-0.15, "v"), (-0.30, "vv")]   # else "vvv"

# ===========================================================================
# 8.6 YIELD CURVE -- treated structurally, not as inverted/not-inverted
# ===========================================================================
# The curve's recession signal is not the inversion itself but the sequence:
# invert -> stay inverted -> re-steepen -> recession. Re-steepening from deep
# inversion is the LATE stage, not the all-clear it superficially resembles,
# because it usually means the market has begun pricing cuts in response to
# visible damage. These weights encode that ordering.
CURVE_STATE_RISK: dict[str, float] = {
    "STEEP": -0.6,             # healthy, early-cycle
    "NORMAL": -0.2,
    "FLATTENING": 0.2,
    "INVERTED_SHALLOW": 0.5,
    "INVERTED_DEEP": 0.8,
    "RE_STEEPENING_BULL": 1.0,  # highest risk: cuts being priced after damage
    "RE_STEEPENING_BEAR": 0.6,
}
CURVE_INVERSION_DEEP = -0.50      # 10Y-3M below this (pp) counts as deep
CURVE_STEEP = 1.50                # above this (pp) the curve is steep
CURVE_FLAT = 0.50                 # below this but positive = flattening
# Change over 3 months that counts as a genuine directional move rather than noise.
CURVE_MOVE_EPS = 0.25
# Months of continuous inversion after which the signal is considered mature.
CURVE_INVERSION_MATURE_MONTHS = 6

# ===========================================================================
# 8.7 INFLATION REGIME
# ===========================================================================
# Boundaries on core YoY inflation, in percent. These ARE arbitrary in the sense
# spec 9 warns about -- but inflation regimes are defined by policy-relevant
# levels (the Fed's 2% target, the ~4% level above which policy turns
# restrictive), not by percentiles of a sample that includes the 1970s.
INFLATION_BANDS = [
    (-float("inf"), 0.0, "DEFLATIONARY"),
    (0.0, 1.5, "DISINFLATIONARY"),
    (1.5, 3.0, "STABLE"),
    (3.0, 5.0, "REFLATIONARY"),
    (5.0, float("inf"), "INFLATIONARY"),
]
# Stagflation is a joint condition, not an inflation level: high inflation AND
# a deteriorating growth picture at the same time.
STAGFLATION_INFLATION_MIN = 3.5
STAGFLATION_GROWTH_MAX = -0.20
# YoY change in inflation that counts as real momentum rather than noise (pp).
INFLATION_MOMENTUM_EPS = 0.3

# ===========================================================================
# 8.8 POLICY RESTRICTIVENESS
# ===========================================================================
# Real policy rate = fed funds - core inflation, versus an assumed neutral real
# rate. 0.5% is a mid-range r* estimate; the number is genuinely uncertain
# (published estimates run ~0.0-1.5%), so the engine reports restrictiveness as
# a distance from it rather than as a precise claim about neutral.
NEUTRAL_REAL_RATE = 0.5
POLICY_BANDS = [
    (-float("inf"), -1.0, "VERY_ACCOMMODATIVE"),
    (-1.0, -0.25, "ACCOMMODATIVE"),
    (-0.25, 0.75, "NEUTRAL"),
    (0.75, 2.0, "RESTRICTIVE"),
    (2.0, float("inf"), "VERY_RESTRICTIVE"),
]

# ===========================================================================
# 11. LEADING INDICATOR BREADTH
# ===========================================================================
# A leading indicator counts as improving/weakening when its 3-month signal
# change exceeds this; inside the band it is neutral. Without a dead band,
# breadth would flicker on rounding noise and every reading would look like 50/50.
BREADTH_EPS = 0.05
# Breadth beyond these shares is a regime-relevant condition worth alerting on.
BREADTH_WEAK_WARNING = 0.60
BREADTH_WEAK_CRITICAL = 0.75
# Consecutive readings a breadth condition must hold before it counts as
# persistent rather than a one-print wobble.
PERSISTENCE_PERIODS = 3
# Divergence: leading and coincident indices pulling apart by more than this is
# the classic turning-point tell (leaders roll over while the economy still
# looks fine).
DIVERGENCE_EPS = 0.35

# ===========================================================================
# 12. REGIME ENGINE
# ===========================================================================
# Regimes are assigned from three coordinates -- the composite level, its
# 3-month momentum, and the leading index -- rather than from any single
# indicator, per spec 12's requirement that the final regime not depend on one.
# Each rule is (name, level_min, level_max, momentum_min, momentum_max).
REGIME_RULES = [
    ("RECESSION",      -1.01, -0.45, -1.01,  0.05),
    ("PRE_RECESSION",  -0.45, -0.20, -1.01, -0.05),
    ("SLOWDOWN",       -0.45,  0.10, -1.01, -0.03),
    ("LATE_CYCLE",     -0.10,  0.35, -0.30,  0.03),
    ("EARLY_RECOVERY", -1.01, -0.20,  0.05,  1.01),
    ("RECOVERY",       -0.20,  0.25,  0.05,  1.01),
    ("EXPANSION",       0.20,  1.01, -0.05,  1.01),
]
REGIME_FALLBACK = "LATE_CYCLE"
# Sequence the dashboard's cycle timeline draws.
REGIME_ORDER = ["EXPANSION", "LATE_CYCLE", "SLOWDOWN", "PRE_RECESSION",
                "RECESSION", "EARLY_RECOVERY", "RECOVERY"]
# Composite momentum beyond this flips the headline trend label.
TREND_EPS = 0.04

# ===========================================================================
# 13/14. RECESSION FORECAST
# ===========================================================================
HORIZONS = [3, 6, 9, 12, 18]

# The rule-based model's mapping from macro risk score to a probability, before
# calibration. Deliberately a logistic curve rather than a lookup table so it is
# monotone and differentiable; the constants are fitted in modules/calibration.py
# against NBER outcomes and these are only the starting values.
RULE_LOGIT_INTERCEPT = -1.6
RULE_LOGIT_SLOPE = 2.4

# Longer horizons carry more unconditional risk simply because more can happen.
# FALLBACK ONLY. modules/calibration.empirical_base_rate() computes this from the
# NBER record actually held, point-in-time, and that computed value is what the
# engine uses; these constants apply only before enough history exists to
# estimate one. They are deliberately close to the 1970- sample values.
#
# Worth knowing: the 1970-2026 rate at 12 months is ~15%, while the realised
# 1990-2026 rate is ~9%. Recessions were markedly more frequent before the Great
# Moderation, so ANY prior built from the longer record over-predicts the modern
# era -- which is exactly why the base rate is estimated point-in-time and
# re-estimated as evidence accumulates rather than frozen here.
HORIZON_BASE_RATE = {3: 0.04, 6: 0.08, 9: 0.11, 12: 0.15, 18: 0.21}

# How many pseudo-observations of the long-run prior the expanding climatology
# starts with. Higher = the base-rate estimate moves more slowly toward the
# realised frequency of the window being scored.
CLIMATOLOGY_PRIOR_WEIGHT = 12

# Evidence shrinkage on the fitted sharpness parameter. The optimal shrinkage
# lambda is estimated from a handful of recession EPISODES, not from the row
# count, so it is badly determined early: lambda is multiplied by
# n_events / (n_events + this). With no events seen, lambda is 0 and the engine
# quotes the base rate, which is the correct thing to say when nothing yet
# demonstrates the signal adds value.
SHRINKAGE_EVIDENCE_PRIOR = 8
# Minimum scored points before any sharpness is applied at all.
SHRINKAGE_MIN_POINTS = 24

# How much of the final probability comes from the current signal versus the
# unconditional base rate, per horizon. At 3 months the signal dominates (we can
# nearly see it); at 18 months the honest answer leans much harder on the base
# rate, because no indicator has demonstrated 18-month precision.
SIGNAL_WEIGHT = {3: 0.85, 6: 0.80, 9: 0.72, 12: 0.65, 18: 0.50}

# ===========================================================================
# 15. MODEL ENGINE
# ===========================================================================
# The reduced feature set the statistical models train on. Kept to ~14 because
# there are only ~8 usable recessions since 1970: a 50-feature logit would fit
# noise and backtest beautifully for the wrong reason. Chosen for independent
# information, not for individual strength -- claims and the curve are both
# strong but tell you different things.
# NOTE ON THE CREDIT FEATURES: these are the Moody's Baa/Aaa spreads over the
# 10Y, NOT the ICE BofA OAS series. The OAS series are the better real-time read,
# but FRED's keyless endpoint serves only a ~3-year rolling window of them, and a
# feature with three years of history in a model trained since 1985 is a
# median-imputed constant for 90% of the sample and a live value only at the end
# -- which is worse than not having it. See config/series.py's CREDIT notes.
MODEL_FEATURES = [
    "T10Y3M", "T10Y2Y",              # curve level and slope
    "BAA10Y", "AAA10Y",              # credit risk pricing (deep history)
    "ICSA", "CCSA", "TEMPHELPS",     # labour turning points
    "PERMIT", "HSN1F",               # housing lead
    "NEWORDER", "ISRATIO",           # capex intent and inventory overhang
    "UMCSENT", "ALTSALES",           # consumer
    "NFCI",                          # financial conditions composite
]
MODELS = ["rule", "logit", "probit", "forest", "gbm"]
ENSEMBLE_MEMBERS = ["rule", "logit", "gbm"]
# Spec 15: "Only use an ensemble if it improves out-of-sample performance."
# The ensemble is published only when its walk-forward AUC beats the best single
# model by at least this margin; otherwise the engine falls back to that model.
ENSEMBLE_MIN_AUC_GAIN = 0.005

# Walk-forward validation: train on everything up to a date, predict the next
# window, roll forward. 60 months of initial training covers at least one full
# cycle before the first prediction is scored.
WALK_FORWARD_MIN_TRAIN_MONTHS = 120
WALK_FORWARD_STEP_MONTHS = 6

# Regularisation. Strong by default: with ~8 positive episodes, an unpenalised
# logit will happily drive a coefficient to infinity on a single separating feature.
LOGIT_C = 0.5
FOREST_TREES = 300
FOREST_MAX_DEPTH = 4
GBM_TREES = 200
GBM_LEARNING_RATE = 0.05
GBM_MAX_DEPTH = 3
RANDOM_SEED = 20260905          # fixed so a run is reproducible (spec 31)

# ===========================================================================
# 16. CALIBRATION
# ===========================================================================
CALIBRATION_METHOD = "isotonic"    # isotonic | sigmoid
CALIBRATION_BINS = 10
# Isotonic regression needs a reasonable sample; below this the engine keeps the
# uncalibrated probability and says so rather than fitting a step function to a
# handful of points and calling the result calibrated.
CALIBRATION_MIN_SAMPLES = 120

# Between these two counts, calibrate with SIGMOID (Platt) instead of isotonic.
# Isotonic is non-parametric and needs a lot of points; Platt scaling has two
# parameters and is the standard choice on small samples precisely because it
# cannot reproduce training noise the way a step function can. This matters most
# in the vintage backtest, where each horizon has only ~140 scoreable points and
# an isotonic-only rule would leave almost all of them uncalibrated.
CALIBRATION_SMALL_SAMPLE_MIN = 40

# ===========================================================================
# 17. SCENARIO ENGINE
# ===========================================================================
SCENARIOS = ["BASE", "BULL", "BEAR", "CRISIS"]
# Scenario probabilities are DERIVED from the calibrated recession probability,
# not asserted: CRISIS and BEAR together must equal it, or the dashboard would
# show a 43% recession probability beside scenarios summing to something else.
# These constants only control how that mass is split and how the remainder is
# divided between BASE and BULL.
CRISIS_SHARE_OF_RECESSION = 0.28   # of recession mass, the tail-risk portion
BULL_SHARE_OF_EXPANSION = 0.25     # of non-recession mass, the reacceleration case
# Crisis requires financial stress, not merely weak growth: a recession without
# credit dysfunction is a slowdown, not 2008.
CRISIS_STRESS_THRESHOLD = 0.60

# ===========================================================================
# 18. CONFIDENCE ENGINE
# ===========================================================================
# Confidence is NOT probability (spec 18). It answers "how much should you trust
# this number", combining agreement, data health and historical accuracy.
CONFIDENCE_WEIGHTS = {
    "indicator_agreement": 0.25,   # do the factors point the same way
    "model_agreement": 0.20,       # do the models agree with each other
    "historical_accuracy": 0.20,   # walk-forward AUC at this horizon
    "data_quality": 0.15,          # the quality engine's health score
    "data_freshness": 0.10,        # how current the inputs are
    "regime_similarity": 0.10,     # is today like anything in the training set
}
CONFIDENCE_BANDS = [(0.75, "HIGH"), (0.50, "MEDIUM")]   # else LOW

# ===========================================================================
# 19/20. MARKET REGIME + ASSET IMPLICATIONS
# ===========================================================================
# Kept strictly separate from the macro engine (spec 19): valuation must never
# enter the recession probability. These weights build a market-regime score
# that CONSUMES the macro output without feeding back into it.
MARKET_WEIGHTS = {
    "valuation": 0.20,
    "earnings": 0.15,
    "liquidity": 0.20,
    "credit": 0.20,
    "rates": 0.15,
    "risk_appetite": 0.10,
}

# Directional implications per asset. Each entry maps a macro condition to a
# stance; modules/market.py combines them. Values are scores in [-1, +1] that
# become CAUTIOUS / NEUTRAL / POSITIVE labels.
ASSET_CLASSES = ["US Equities", "Government Bonds", "Investment Grade",
                 "High Yield", "USD", "Gold", "Commodities", "REITs"]
# Sensitivity of each asset to (growth, inflation, credit stress, real rates).
# Signs encode textbook relationships; magnitudes are relative, not calibrated
# betas -- this is an implication layer, not a return forecast (spec 20).
ASSET_SENSITIVITY = {
    "US Equities":      {"growth": +0.9, "inflation": -0.2, "stress": -0.7, "real_rate": -0.4},
    "Government Bonds": {"growth": -0.7, "inflation": -0.8, "stress": +0.6, "real_rate": -0.9},
    "Investment Grade": {"growth": +0.2, "inflation": -0.5, "stress": -0.3, "real_rate": -0.7},
    "High Yield":       {"growth": +0.8, "inflation": -0.2, "stress": -0.9, "real_rate": -0.3},
    "USD":              {"growth": +0.2, "inflation": +0.1, "stress": +0.5, "real_rate": +0.7},
    "Gold":             {"growth": -0.2, "inflation": +0.5, "stress": +0.6, "real_rate": -0.8},
    "Commodities":      {"growth": +0.8, "inflation": +0.6, "stress": -0.3, "real_rate": -0.2},
    "REITs":            {"growth": +0.6, "inflation": -0.1, "stress": -0.5, "real_rate": -0.8},
}
ASSET_BANDS = [(0.35, "POSITIVE"), (0.12, "CONSTRUCTIVE"),
               (-0.12, "NEUTRAL"), (-0.35, "CAUTIOUS")]   # else "NEGATIVE"

# ===========================================================================
# 24. ALERTS
# ===========================================================================
# Spec 24 is explicit that these are configurable and not statistically
# universal, so they are stated as operator preferences rather than findings.
PROB_WARNING = 0.50
PROB_HIGH = 0.70
PROB_CRITICAL = 0.85
# Change in 12m probability over one week that is worth waking someone for.
PROB_JUMP_EPS = 0.08
# Credit spread widening (pp over 1 month) that counts as a stress event.
CREDIT_JUMP_PP = 1.0
# Factor score deterioration over 3 months that counts as a turning point.
TURNING_POINT_EPS = 0.35
DATA_HEALTH_ALERT = 0.90

# ===========================================================================
# 21. RECOMMENDATION / INVALIDATION
# ===========================================================================
# How many drivers to surface. Six is about the limit of what a reader takes in
# from a dashboard panel at a glance (spec 27's 10-second rule).
TOP_DRIVERS = 4
TOP_OFFSETS = 3
