# Zero-Cost Macro Forecasting Engine
## Developer Build Specification

**Version:** 1.0  
**Date:** 2026-09-05  
**Primary objective:** Build a production-ready U.S. macroeconomic forecasting engine using only free/public economic data APIs, with robust data handling, historical-vintage backtesting, recession forecasting, macro regime detection, scenarios, QA, and a dashboard.

---

# 1. Product Objective

Build a macroeconomic forecasting system inspired by long-cycle/leading-indicator analysis, including Gary Shilling-style emphasis on housing, credit, consumer stress, monetary conditions, and economic turning points.

The system must answer:

1. What is the current macroeconomic regime?
2. Is the economy improving or deteriorating?
3. What is the probability of recession over 3/6/9/12/18 months?
4. What are the expected directions for growth, inflation, unemployment, rates, housing, consumer demand, and credit?
5. Which indicators are driving the forecast?
6. How confident is the model?
7. What are the base, bull, bear, and crisis scenarios?
8. What are the implications for major asset classes?
9. How accurate has the model historically been?
10. Can the system prove that its historical forecasts did not use future information?

## Hard Constraints

- Data cost: **$0/month**
- No paid economic-data vendors
- No paid AI/ML APIs required
- Python-based local processing
- SQLite and/or Parquet for local storage
- API keys may be used only where the provider supplies them free
- Must support incremental updates and local caching
- Must avoid look-ahead bias
- Must preserve historical data vintages wherever possible
- Must be explainable; no black-box-only final score

---

# 2. Recommended High-Level Architecture

```text
                    FREE PUBLIC DATA
                          |
       +------------------+------------------+
       |                  |                  |
   FRED/ALFRED          BLS                BEA
       |                  |                  |
       +------------------+------------------+
                          |
                  DATA INGESTION
                          |
                  LOCAL DATA CACHE
                          |
                    DATA QA LAYER
                          |
              RELEASE/VINTAGE CONTROL
                          |
                 TRANSFORMATION ENGINE
                          |
              +-----------+-----------+
              |                       |
        MACRO FACTORS          LEADING INDICATORS
              |                       |
              +-----------+-----------+
                          |
                  SIGNAL ENGINE
                          |
                  REGIME ENGINE
                          |
                 FORECAST ENGINE
                          |
             PROBABILITY CALIBRATION
                          |
                 SCENARIO ENGINE
                          |
                 MARKET REGIME
                          |
              RECOMMENDATION ENGINE
                          |
                   QA/BACKTEST
                          |
                     DASHBOARD
```

---

# 3. Free Data Sources

## 3.1 FRED / ALFRED

Primary source.

Use for:

- Housing
- Building permits
- Mortgage rates
- Interest rates
- Yield curve
- Credit conditions
- Financial conditions
- Industrial production
- Consumer indicators
- Employment indicators
- Inflation
- Monetary indicators
- Recession labels

ALFRED must be used where historical vintage data is available.

## 3.2 BLS Public API

Use where BLS is the preferred source:

- CPI
- Employment
- Unemployment
- Wages
- Hours
- Labor-market indicators

## 3.3 BEA API

Use for:

- GDP
- Real GDP
- Personal income
- Real disposable income
- Personal consumption
- Savings
- Corporate profits
- National accounts

---

# 4. Data Frequency Architecture

Do NOT force every series into daily frequency.

Maintain the source's native frequency.

Supported frequencies:

- Daily
- Weekly
- Monthly
- Quarterly

Each observation should retain:

```text
observation_date
release_date
vintage_date
series_id
value
frequency
source
```

## Daily process

Run every day to:

- Check APIs
- Detect new observations
- Detect revisions
- Update local cache
- Run data QA
- Recalculate affected signals
- Update current macro snapshot

## Event-driven process

Immediately recalculate affected factors after important releases:

- CPI
- Employment report
- GDP
- Retail sales
- Housing data
- ISM
- Fed decisions
- Major financial-condition changes

## Weekly process

Run:

- Full forecast refresh
- Scenario refresh
- Model comparison
- Forecast confidence update
- Macro narrative generation

## Monthly process

Run:

- Historical recalculation
- Model diagnostics
- Calibration
- Backtest refresh
- Drift analysis

## Quarterly process

Run:

- Model review
- Feature review
- Weight review
- Retraining if justified
- Historical performance review

---

# 5. Data Storage

Recommended:

```text
SQLite
    |
    +-- series_metadata
    +-- observations
    +-- vintages
    +-- releases
    +-- transformations
    +-- factor_scores
    +-- forecasts
    +-- scenarios
    +-- regimes
    +-- model_runs
    +-- backtest_results
    +-- alerts
```

Use Parquet for larger historical datasets if required.

Never download the entire historical dataset on every run.

Implement incremental updates.

---

# 6. Data Quality Engine

Create:

```python
class DataQualityEngine:

    validate_schema()
    detect_missing_values()
    detect_stale_values()
    detect_duplicates()
    detect_outliers()
    detect_frequency_mismatch()
    detect_timestamp_errors()
    detect_revision()
    calculate_data_freshness()
    calculate_data_quality_score()
```

Every series should have:

```text
Data Quality: HIGH / MEDIUM / LOW
Last Observation
Last Release
Observation Age
Missing %
Revision Status
```

A bad data source must never silently produce a normal forecast.

---

# 7. Vintage and Look-Ahead Protection

This is mandatory.

Historical backtests must use only information available at that historical date.

Example:

```text
Forecast date: 2007-07-01

Allowed:
- Data released before 2007-07-01

Not allowed:
- GDP revision released in 2008
- Revised employment data
- Future CPI observations
```

Create:

```python
class VintageController:

    get_available_data(as_of_date)
    get_vintage(series_id, vintage_date)
    enforce_release_date()
    detect_future_data()
    audit_backtest_inputs()
```

Every backtest run must log the data vintage used.

---

# 8. Macro Categories

Target approximately 50–70 carefully selected series.

## 8.1 Housing

Include:

- Housing starts
- Building permits
- New home sales
- Existing home sales
- Mortgage rates
- Mortgage applications
- Housing prices
- Housing affordability
- Housing inventory

Functions:

```python
calculate_housing_factor()
calculate_housing_momentum()
calculate_housing_turning_point()
```

## 8.2 Consumer

Include:

- Real disposable income
- Personal consumption
- Real retail sales
- Savings rate
- Consumer confidence
- Consumer expectations
- Consumer credit
- Auto sales
- Debt service
- Delinquencies

Functions:

```python
calculate_consumer_factor()
calculate_consumer_stress()
calculate_consumer_momentum()
```

## 8.3 Labor

Include:

- Unemployment rate
- Payroll employment
- Initial claims
- Continuing claims
- Employment growth
- Temporary employment
- Job openings
- Hiring
- Quits
- Hours worked
- Wage growth

Functions:

```python
calculate_labor_factor()
calculate_labor_turning_point()
calculate_claims_signal()
calculate_sahm_signal()
```

## 8.4 Manufacturing / Business Cycle

Include:

- ISM Manufacturing
- ISM New Orders
- ISM Employment
- Industrial production
- Capacity utilization
- Durable goods
- Capital goods orders
- Factory orders
- Inventory/sales

Functions:

```python
calculate_manufacturing_factor()
calculate_orders_inventory_signal()
calculate_business_cycle_factor()
```

## 8.5 Credit

Include:

- High-yield spreads
- Investment-grade spreads
- Bank lending standards
- Commercial & industrial loans
- Consumer credit
- Mortgage delinquencies
- Corporate defaults
- Financial stress

Functions:

```python
calculate_credit_factor()
calculate_credit_spread_signal()
calculate_lending_conditions()
calculate_credit_turning_point()
```

## 8.6 Yield Curve

Include:

- 10Y minus 2Y
- 10Y minus 3M
- 30Y
- 10Y
- 2Y
- Short-term policy rate

Functions:

```python
calculate_curve_factor()
calculate_inversion_depth()
calculate_inversion_duration()
calculate_resteepening()
calculate_bull_steepening()
calculate_bear_steepening()
```

Do not classify the yield curve simply as:

```text
Inverted = bad
Normal = good
```

The transition and direction matter.

## 8.7 Inflation

Include:

- Headline CPI
- Core CPI
- PCE
- Core PCE
- Services inflation
- Goods inflation
- Wage growth
- Inflation expectations

Functions:

```python
calculate_inflation_factor()
calculate_inflation_momentum()
calculate_inflation_regime()
```

Regimes:

```text
DEFLATIONARY
DISINFLATIONARY
STABLE
REFLATIONARY
INFLATIONARY
STAGFLATIONARY
```

## 8.8 Monetary Policy

Include:

- Fed funds rate
- Real policy rate
- Policy-rate changes
- Balance sheet
- Financial conditions
- Mortgage rates
- Corporate borrowing costs

Functions:

```python
calculate_monetary_policy_factor()
calculate_policy_restrictiveness()
calculate_policy_momentum()
```

## 8.9 Financial Conditions

Functions:

```python
calculate_financial_conditions_factor()
calculate_liquidity_signal()
calculate_market_stress()
```

---

# 9. Transformation Engine

Implement:

```python
calculate_yoy()
calculate_mom()
calculate_qoq()
calculate_moving_average()
calculate_momentum()
calculate_acceleration()
calculate_zscore()
calculate_percentile()
calculate_historical_percentile()
normalize_signal()
```

Do not use arbitrary thresholds everywhere.

Prefer historical normalization where statistically appropriate.

Example:

```text
Housing starts YoY = -12%

Historical percentile = 8%

Interpretation = unusually weak
```

---

# 10. Factor Engine

Each category should produce a factor score.

Example:

```text
Housing       -0.72
Consumer      -0.20
Labor         -0.35
Manufacturing -0.45
Credit        -0.60
Inflation     +0.10
Policy        -0.55
Financial     -0.50
```

Standardize factor scores to:

```text
-1.0 = strongly negative
 0.0 = neutral
+1.0 = strongly positive
```

Define clearly whether positive means economic improvement or risk.

Never mix sign conventions.

---

# 11. Leading Indicator Engine

Implement:

```python
calculate_leading_indicator_index()
calculate_coincident_index()
calculate_lagging_index()
calculate_signal_breadth()
calculate_signal_persistence()
calculate_signal_acceleration()
detect_macro_divergence()
detect_turning_point()
```

Example:

```text
Leading indicators monitored: 18

Improving:  3
Neutral:    4
Weakening: 11

Weakening breadth: 61%
```

This should feed the regime engine.

---

# 12. Regime Engine

Classify:

```text
EXPANSION
LATE_CYCLE
SLOWDOWN
PRE_RECESSION
RECESSION
EARLY_RECOVERY
RECOVERY
```

Also classify independent dimensions:

```text
Growth regime
Inflation regime
Labor regime
Credit regime
Liquidity regime
Policy regime
```

The final regime should not depend on one indicator.

---

# 13. Recession Forecast

Maintain two separate outputs:

```text
RAW MACRO RISK SCORE
CALIBRATED RECESSION PROBABILITY
```

Do NOT label a simple weighted score as a probability.

Forecast horizons:

```text
3 months
6 months
9 months
12 months
18 months
```

Example:

```text
3M  = 14%
6M  = 27%
9M  = 36%
12M = 43%
18M = 48%
```

---

# 14. Forecast Engine

Forecast:

```python
forecast_recession_probability()
forecast_growth()
forecast_gdp()
forecast_inflation()
forecast_unemployment()
forecast_policy_rate()
forecast_long_term_rates()
forecast_housing()
forecast_consumer()
forecast_credit()
```

Forecast outputs should include:

```text
direction
estimate/range where appropriate
confidence
key drivers
key risks
```

---

# 15. Model Engine

Start with:

```text
1. Expert rule-based model
2. Logistic regression
3. Probit where practical
4. Random forest
5. Gradient boosting
```

Do not assume ML is superior.

Compare models through walk-forward validation.

Only use an ensemble if it improves out-of-sample performance.

---

# 16. Probability Calibration

Implement:

```python
calibrate_probability()
calculate_brier_score()
calculate_calibration_curve()
```

The objective is that:

```text
Forecast 70%
```

should historically correspond to approximately:

```text
70% occurrence rate
```

within statistical uncertainty.

---

# 17. Scenario Engine

Generate:

```text
BASE
BULL
BEAR
CRISIS
```

Each scenario must contain:

```text
Probability
Growth
Inflation
Unemployment
Rates
Housing
Consumer
Credit
Key assumptions
Key risks
```

Example:

```text
BASE
Probability: 50%
Soft landing / slow growth

BEAR
Probability: 25%
Growth deterioration

CRISIS
Probability: 10%
Recession + financial stress

BULL
Probability: 15%
Growth reacceleration
```

---

# 18. Confidence Engine

Probability and confidence are different.

Example:

```text
Recession probability: 61%
Confidence: LOW
```

or:

```text
Recession probability: 61%
Confidence: HIGH
```

Confidence should consider:

- Indicator agreement
- Model agreement
- Historical model accuracy
- Data freshness
- Data completeness
- Regime similarity
- Forecast dispersion

---

# 19. Market Regime Engine

Keep valuation separate from recession probability.

Do NOT:

```text
Recession Probability = Macro Risk + CAPE
```

Instead:

```text
MACRO ENGINE
    |
    +-- Growth
    +-- Inflation
    +-- Labor
    +-- Credit
    +-- Recession
    |
    v
MARKET REGIME ENGINE
    |
    +-- Valuation
    +-- Earnings
    +-- Liquidity
    +-- Credit
    +-- Rates
    +-- Risk appetite
```

CAPE belongs here.

---

# 20. Asset-Class Implication Engine

Produce directional implications for:

```text
US Equities
Government Bonds
Corporate Investment Grade
High Yield
USD
Gold
Commodities
REITs
```

Example:

```text
Equities       Cautious
Treasuries     Positive
HY Credit      Negative
USD            Neutral
Gold           Positive
REITs          Cautious
```

This is an implication layer, not a guarantee.

---

# 21. Recommendation Engine

Output:

```text
Current macro regime
Trend
Recession risk
Confidence
Key drivers
Key risks
Asset-class implications
What would change the view?
```

Also provide invalidation conditions.

Example:

```text
Current view: Deteriorating

Would improve if:
- Claims reverse lower
- Housing permits stabilize
- Credit spreads tighten

Would worsen if:
- Unemployment rises materially
- Credit spreads widen
- Consumer spending contracts
```

---

# 22. Backtesting Engine

Implement:

```python
run_historical_backtest()
run_walk_forward_validation()
calculate_precision()
calculate_recall()
calculate_false_positive_rate()
calculate_false_negative_rate()
calculate_auc()
calculate_brier_score()
calculate_calibration()
calculate_lead_time()
```

Test major historical cycles including:

```text
2001 recession
2008–2009 GFC
2020 recession
Other identifiable economic slowdowns
```

The backtest must use only historically available data.

---

# 23. QA / Tester Requirements

Automated tests must cover:

```text
Data schema
API failures
Missing observations
Duplicate observations
Stale data
Frequency conversion
Date alignment
Release-date alignment
Vintage selection
Future-data leakage
Signal ranges
Weight sums
Probability calibration
Model reproducibility
Backtest reproducibility
```

Example:

```python
def test_no_future_data_leakage():
    ...

def test_monthly_series_not_falsely_daily():
    ...

def test_recession_probability_between_zero_and_one():
    ...

def test_factor_score_range():
    ...

def test_vintage_date_is_respected():
    ...
```

---

# 24. Alert Engine

Alert when:

```text
Macro regime changes
Recession probability crosses thresholds
Leading-indicator breadth deteriorates sharply
Credit stress jumps
Labor turning point detected
Housing turning point detected
Yield curve changes regime
Data quality deteriorates
```

Suggested thresholds:

```text
Probability >= 50%  → Warning
Probability >= 70%  → High Risk
Probability >= 85%  → Critical
```

These thresholds should remain configurable and should not be interpreted as statistically universal.

---

# 25. Run Frequency

## Daily

```text
06:00
    Pull free APIs
    Validate data
    Update cache
    Check releases
    Recalculate affected signals
    Update macro snapshot
```

## After major release

```text
New CPI
    → Inflation
    → Policy
    → Macro regime
    → Forecast

New Employment
    → Labor
    → Consumer
    → Recession probability
```

## Weekly

```text
Full forecast refresh
Scenario refresh
Model comparison
Confidence calculation
```

## Monthly

```text
Backtest refresh
Calibration
Model diagnostics
Drift monitoring
```

## Quarterly

```text
Model review
Feature review
Weight review
Retraining decision
```

---

# 26. Dashboard Requirements

Dashboard should be optimized for quick decision-making.

## Top section

```text
MACRO REGIME
LATE CYCLE / SLOWDOWN

12M RECESSION PROBABILITY
43%

CONFIDENCE
74%

TREND
DETERIORATING
```

## Recession probability chart

Display:

```text
3M     14%
6M     27%
9M     36%
12M    43%
18M    48%
```

Include historical probability line if available.

## Factor heatmap

```text
Factor             Score       Trend
Housing            -0.72       ↓↓↓
Consumer           -0.20       ↓
Labor              -0.35       ↓↓
Manufacturing      -0.45       ↓↓
Credit             -0.60       ↓↓↓
Inflation           0.10       →
Policy             -0.55       ↓
Financial          -0.50       ↓↓
```

## Leading-indicator breadth

```text
18 indicators

Improving      3
Neutral        4
Weakening     11

Weakening breadth: 61%
```

## Macro regime timeline

Show:

```text
Expansion → Late Cycle → Slowdown → Pre-Recession → Recession → Recovery
```

with current position highlighted.

## Scenario panel

```text
BASE     50%
BULL     15%
BEAR     25%
CRISIS   10%
```

## Driver panel

```text
TOP NEGATIVE DRIVERS
1. Credit conditions
2. Housing permits
3. Labor claims
4. Manufacturing orders

TOP POSITIVE DRIVERS
1. Real income
2. Inflation moderation
```

## Market implication panel

```text
Equities        CAUTIOUS
Treasuries      POSITIVE
HY Credit       NEGATIVE
Gold            POSITIVE
USD             NEUTRAL
REITs           CAUTIOUS
```

## "What changes the forecast?" panel

```text
IMPROVES IF:
✓ Housing stabilizes
✓ Claims decline
✓ Credit spreads tighten

WORSENS IF:
⚠ Unemployment rises
⚠ Credit spreads widen
⚠ Consumer spending contracts
```

---

# 27. Dashboard UX Principle

The dashboard should answer the following within 10 seconds:

```text
1. Where are we?
2. Where are we going?
3. How confident are we?
4. Why?
5. What could invalidate the forecast?
6. What does it imply for markets?
```

Do not overwhelm the main dashboard with raw economic series.

Raw data should be available in drill-down pages.

---

# 28. Recommended Dashboard Pages

```text
PAGE 1 — EXECUTIVE MACRO DASHBOARD

PAGE 2 — RECESSION FORECAST

PAGE 3 — FACTOR ANALYSIS

PAGE 4 — LEADING INDICATORS

PAGE 5 — GROWTH / INFLATION / LABOR

PAGE 6 — CREDIT / LIQUIDITY / RATES

PAGE 7 — SCENARIOS

PAGE 8 — MARKET IMPLICATIONS

PAGE 9 — MODEL PERFORMANCE

PAGE 10 — DATA QUALITY / SYSTEM HEALTH
```

---

# 29. API Failure Handling

Never fail silently.

Implement:

```python
retry_with_backoff()
record_api_error()
use_cached_value_if_allowed()
mark_series_stale()
raise_data_quality_alert()
```

Dashboard should show:

```text
Data Health: 96%
2 series stale
0 critical failures
Last successful update: 06:04
```

---

# 30. Explainability

Every major forecast must have a machine-readable explanation.

Example:

```json
{
  "regime": "SLOWDOWN",
  "recession_probability_12m": 0.43,
  "confidence": 0.74,
  "drivers": [
    "Credit conditions deteriorating",
    "Housing permits weakening",
    "Labor claims increasing"
  ],
  "offsetting_factors": [
    "Real income remains positive",
    "Inflation is moderating"
  ]
}
```

The dashboard should convert this into human-readable language.

---

# 31. Logging

Every model run must record:

```text
run_id
run_timestamp
data_timestamp
model_version
feature_version
data_vintage
forecast_horizon
forecast_probability
confidence
regime
input_hash
output_hash
```

This makes forecasts reproducible.

---

# 32. Security / Secrets

Never hard-code API keys.

Use:

```text
.env
environment variables
secret configuration
```

Example:

```python
FRED_API_KEY = os.getenv("FRED_API_KEY")
BEA_API_KEY = os.getenv("BEA_API_KEY")
```

Never commit secrets to Git.

---

# 33. Definition of Done

V1 is complete only when:

- [ ] Free APIs successfully ingest required data
- [ ] Local cache works
- [ ] Incremental updates work
- [ ] Native frequencies are preserved
- [ ] Release dates are stored
- [ ] Historical vintages are supported
- [ ] No look-ahead leakage exists
- [ ] 50–70 core indicators are operational
- [ ] Factor scores are generated
- [ ] Leading-indicator breadth is generated
- [ ] Macro regime is generated
- [ ] 3/6/9/12/18M recession probabilities are generated
- [ ] Probability calibration is implemented
- [ ] Scenario engine works
- [ ] Market regime is separate from macro recession probability
- [ ] Historical walk-forward backtest works
- [ ] QA suite passes
- [ ] Dashboard works
- [ ] Alerts work
- [ ] Every forecast is explainable
- [ ] Model run is reproducible
- [ ] Data failures are visible
- [ ] No paid data/API dependency exists

---

# 34. Recommended Development Order

Do NOT build everything simultaneously.

## Phase 1 — Data Foundation

```text
FRED/ALFRED
BLS
BEA
↓
SQLite
↓
Data QA
↓
Caching
```

## Phase 2 — Core Factors

```text
Housing
Consumer
Labor
Credit
Yield Curve
Inflation
Manufacturing
Financial Conditions
```

## Phase 3 — Signal Engine

```text
Z-scores
Momentum
Breadth
Divergence
Turning points
```

## Phase 4 — Forecast Engine

```text
Rule-based model
Logistic model
Gradient boosting
Calibration
```

## Phase 5 — Validation

```text
Vintage backtest
Walk-forward testing
Historical recession testing
```

## Phase 6 — Scenario + Market Layer

```text
Scenarios
Market regime
Asset implications
```

## Phase 7 — Dashboard

```text
Executive view
Drill-down
Model performance
Data health
```

---

# 35. Final Product Output

The final daily report should look approximately like:

```text
========================================================
                 MACRO FORECAST
                 05 SEP 2026
========================================================

REGIME
LATE CYCLE / SLOWDOWN

12M RECESSION PROBABILITY
43%

CONFIDENCE
74%

TREND
DETERIORATING
--------------------------------------------------------

RECESSION PROBABILITY

3M       14%
6M       27%
9M       36%
12M      43%
18M      48%

--------------------------------------------------------

MACRO FACTORS

Housing          -0.72   ↓↓↓
Consumer         -0.20   ↓
Labor            -0.35   ↓↓
Manufacturing    -0.45   ↓↓
Credit           -0.60   ↓↓↓
Inflation         0.10   →
Policy           -0.55   ↓
Financial        -0.50   ↓↓

--------------------------------------------------------

LEADING INDICATORS

Weakening:       11 / 18
Neutral:          4 / 18
Improving:        3 / 18

Breadth:         61% weakening

--------------------------------------------------------

SCENARIOS

BASE             50%
BULL             15%
BEAR             25%
CRISIS           10%

--------------------------------------------------------

TOP RISKS

1. Credit deterioration
2. Housing weakness
3. Labor-market weakening

OFFSETS

1. Inflation moderation
2. Real income resilience

--------------------------------------------------------

MARKET IMPLICATION

Equities         CAUTIOUS
Treasuries       POSITIVE
HY Credit        NEGATIVE
Gold             POSITIVE
USD              NEUTRAL
REITs            CAUTIOUS

--------------------------------------------------------

FORECAST INVALIDATION

IMPROVES IF:
- Housing stabilizes
- Claims decline
- Credit spreads tighten

WORSENS IF:
- Unemployment accelerates
- Credit spreads widen
- Consumer spending contracts

--------------------------------------------------------

DATA HEALTH
96%

MODEL VERSION
macro-v1.0.0

LAST UPDATE
06:04

========================================================
```
