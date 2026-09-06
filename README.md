# Zero-Cost Macro Forecasting Engine

A US macroeconomic forecasting system built entirely on free, **keyless** public
data. Produces a macro regime, calibrated recession probabilities at 3/6/9/12/18
months, factor scores, leading-indicator breadth, scenarios, market implications
and a vintage-true historical backtest — with an explicit, tested guarantee that
no historical forecast used future information.

Implements [`docs/spec.md`](docs/spec.md) v1.0. The dashboard design it was
built against is in [`docs/design-mockup.png`](docs/design-mockup.png).

**Cost: $0/month. No API key. No signup. No paid dependency.**

---

## What it answers

| Question | Where |
|---|---|
| What is the current macro regime? | Executive page · `main.py report` |
| Improving or deteriorating? | Composite momentum + breadth |
| Recession probability at 3/6/9/12/18m? | Recession page |
| Direction for growth, inflation, unemployment, rates, housing, consumer, credit? | Growth · Credit pages |
| Which indicators drive the forecast? | Drivers panel, per-factor detail |
| How confident is the model? | Confidence engine (six components) |
| Base / bull / bear / crisis? | Scenarios page |
| Implications for asset classes? | Market page |
| How accurate has it been historically? | Model Performance page |
| **Can it prove no look-ahead bias?** | `tests/test_no_leakage.py` + the backtest audit trail |

---

## Quick start

```bash
pip install -r requirements.txt
python tools/backfill.py --observations      # ~2 min: full history, 84 series
python main.py daily                         # snapshot + dashboard feed
python main.py report                        # the text report
```

Then, once (slow, resumable, and required before the backtest means anything):

```bash
python tools/backfill.py --vintages
```

That downloads real historical vintages from ALFRED — roughly 3,800 keyless
requests, about 90 minutes, self-throttled. It is resumable: every completed
(series, vintage) pair is recorded and skipped on the next run, so interrupting
it is safe.

---

## Data sources — all keyless

| Source | Use | Auth |
|---|---|---|
| **FRED** `fredgraph.csv?id=A,B,C` | 84 series, batched ~12 per request | none |
| **ALFRED** `alfredgraph.csv?id=X&vintage_date=D` | true historical vintages | none |
| **BLS** API v2 | CPI / employment cross-check | none (25 req/day) |
| Yahoo chart API | gold only, when FRED's LBMA fixings 404 | none |

A `FRED_API_KEY` is read if present but is **never required** — it only makes the
vintage backfill faster by returning all vintages in one call instead of one per
request. **BEA is deliberately absent**: it needs a UserID, and every BEA series
the spec asks for (GDP, PCE, income, savings) is mirrored on FRED under an id
already fetched keylessly.

---

## Architecture

```
config/
  settings.py   paths, env, network, secrets      (operational knobs)
  series.py     the 84-series catalogue           (what data, and its meaning)
  strategy.py   weights, thresholds, model params (every judgement call, in one file)

modules/
  sources.py     every outbound HTTP call; fetch/parse split so parsers are pure
  ingestion.py   incremental update, revision detection, release-date estimation
  quality.py     DataQualityEngine — 10 checks, HIGH/MEDIUM/LOW per series
  vintage.py     VintageController — the look-ahead guarantee
  transforms.py  YoY/momentum/EXPANDING z-score & percentile -> [-1,+1] signal
  factors.py     9 category factors (curve/inflation/policy handled structurally)
  leading.py     breadth, persistence, divergence, turning points
  regime.py      7 cycle regimes + 6 dimension regimes
  models.py      rule · logit · probit · forest · GBM, purged walk-forward
  calibration.py isotonic/Platt/shrinkage + point-in-time climatology, Brier, ECE
  forecast.py    recession probabilities + 10 numeric variable forecasts
  confidence.py  6-component confidence (NOT probability)
  scenarios.py   BASE/BULL/BEAR/CRISIS, derived from the 12m probability
  market.py      market regime + asset implications (one-way from macro)
  recommend.py   drivers, offsets, invalidation conditions, narrative
  backtest.py    vintage-true historical backtest + metrics + sanity flags
  alerts.py      8 alert kinds, deduplicated per (kind, date)
  runlog.py      input/output hashing for reproducibility
  snapshot.py    the single computation everything else renders
  report.py      the ASCII daily report

tools/    backfill.py (history + vintages)
tests/    114 tests, fully offline
main.py   daily | event | weekly | monthly | quarterly | backtest | report | health | export
```

### Front end

The dashboard is the **Macro tab of the sibling `Personal Dynamic Dashboard`**
(static HTML/JS, zero dependencies). The KLSE Monitor tab was removed to make
room; AlphaSpike keeps its own Streamlit app.

- `macro-engine.js` — the ten spec pages, plus the original series monitor as an
  eleventh ("Series Monitor") so nothing that already worked was discarded.
- Feed: `data/macro_engine.json`, written atomically by `main.py`.

```bash
cd "../Personal Dynamic Dashboard" && python -m http.server 8099
# then open http://localhost:8099/#macro
```

---

## The three design decisions that matter most

### 1. Look-ahead protection has three channels, not one

A release-date filter alone is not enough. The engine closes all three:

| Leak | Example | Closed by |
|---|---|---|
| Future **observations** | Q2 GDP not yet published | release-date filter (`vintage.py`) |
| Future **revisions** | published, but at a later-revised number | real ALFRED vintages (tier A) |
| Future **statistics** | a 2007 z-score computed against 1970–2026 | **expanding** z-scores (`transforms.py`) |

The third is the one most systems miss. A full-sample z-score at 2007 knows
about 2008; `tests/test_no_leakage.py` demonstrates both the correct behaviour
and the failure mode it avoids.

Series are tiered by how their revisions are handled, and the tier is reported,
never assumed:

- **tier A `alfred`** (23 series) — real vintages. Revisions reproduced exactly.
- **tier B `final`** — market prices, yields, spreads. Never revised, so today's
  value *is* the historical one. Exact, not an approximation.
- **tier C `lag`** — revised but without vintage coverage. Publication timing is
  respected; revisions are **not** reproduced, and every result touching one is
  flagged `vintage_true = 0`.

ALFRED holds no vintages before roughly 1997–2000, so pre-2000 backtest dates are
necessarily tier C. The backtest says so rather than quietly averaging it in.

### 2. ML is not assumed superior — it is measured

Spec §15 says "do not assume ML is superior". On this data, purged walk-forward
at the 12-month horizon gives:

| Model | AUC | Brier |
|---|---|---|
| **rule** (expert, 2 parameters) | **0.853** | 0.101 |
| logit | 0.819 | 0.165 |
| probit | 0.802 | 0.111 |
| random forest | 0.792 | 0.132 |
| gradient boosting | 0.652 | 0.134 |
| ensemble | 0.830 | 0.107 |

The interpretable rule-based model wins, and the ensemble fails to beat it by the
configured margin, so it is not published. With ~8 usable recessions since 1970,
parsimony is a real advantage, not a handicap.

Every training fold **purges** the last `horizon` months. Without that, adjacent
rows share most of their outcome window and the reported AUC would be a leak.

Isotonic calibration on those out-of-sample predictions moves the Brier score
from 0.101 to 0.071, expected calibration error from 0.083 to ~0.000, and gives a
Brier skill of **+0.192** against the base rate.

### 3. Probability, confidence and score are three different things

- **Raw macro risk score** — the model's uncalibrated output.
- **Calibrated probability** — that score blended toward the horizon's historical
  base rate and mapped through a calibrator fitted on *out-of-sample*
  predictions. Only this should be read as a frequency.
- **Confidence** — how much to trust the probability at all, from indicator
  agreement, model agreement, historical accuracy, data quality, data freshness
  and how similar today is to anything in the training record.

All three are stored and displayed. A 61% probability at LOW confidence and a 61%
probability at HIGH confidence are opposite conclusions, and the dashboard can
express both.

The engine also **declines to calibrate** when the out-of-sample sample is too
small, and reports the probability as UNCALIBRATED rather than fitting a step
function to 40 points and calling the result calibrated.

---

## Measured performance

Two numbers, deliberately reported side by side, because they answer different
questions and only one of them is honest about real time.

**Walk-forward on revised data** — "which model structure generalises best". Many
points, enough to compare five models. AUC **0.853** at 12 months (table above).

**Vintage-true backtest** — "what would this engine actually have said at the
time". 147 quarterly as-of dates from 1990, each one refitting the model on data
filtered through the vintage controller. 695 scored points, **0 leaks** in the
audit trail.

| Horizon | n | Base rate | AUC | Brier | Brier skill | Inputs vintage-true |
|---|---|---|---|---|---|---|
| 3M | 138 | 2.2% | 0.721 | 0.022 | −0.004 | 63% |
| 6M | 137 | 4.4% | 0.714 | 0.043 | −0.004 | 63% |
| 9M | 136 | 6.6% | 0.710 | 0.065 | −0.004 | 63% |
| 12M | 135 | 8.9% | 0.684 | 0.087 | −0.009 | 63% |
| 18M | 133 | 13.5% | 0.622 | 0.128 | **+0.004** | 63% |

**The 0.853 → 0.701 gap at 12 months is the cost of real time** — revisions and
publication lags. A system reporting only the first number would be overstating
what it could actually have done.

### Lead time

All three recessions in the window are detected, on the raw score at 2× the
realised base rate:

| Recession | First signal | Lead |
|---|---|---|
| 2001-04 | 2001-01 | 2.5 months |
| 2008-01 | 2006-07 | **17.6 months** |
| 2020-03 | 2019-07 | 7.6 months |

Median **7.6 months**. The 2008 signal is the strongest and earliest, which is
what a credit-and-housing-weighted model should do. The 2019 signal ahead of 2020
reflects the yield-curve inversion and late-cycle stress that were genuinely
present — it is not a claim to have forecast a pandemic.

### Brier skill: what "good" actually is here

Skill is now **−0.009 to +0.004** across all five horizons — flat zero to within
noise, and that is the ceiling, not a shortfall.

The reason is worth stating precisely. With three recessions in the scored
window, the best possible single sharpness parameter chosen **with hindsight**
scores only about **+0.01 to +0.02**. The engine gets within 0.015 of that using
no future information. There is essentially nothing left to extract on squared
error, and any system claiming a strongly positive Brier skill on 1990–2026 US
recession forecasting is either using future information or got lucky on three
events.

Two fixes got it from −0.085 to −0.009 at 12 months:

1. **The base rate is estimated point-in-time, not hardcoded.** It was a fixed
   15% at 12 months, taken from the full NBER record and applied at historical
   dates — both a mild look-ahead and biased high, because recessions were far
   more frequent before the Great Moderation (1970– rate 15%, realised 1990–2026
   rate 9%). It is now computed from the record available at each as-of date.
2. **A one-parameter shrinkage calibrator replaced free Platt/isotonic refits.**
   Refitting a two-parameter calibrator at every as-of date on three recession
   events was unstable enough to destroy the ranking — pooled AUC on the
   calibrated series fell to 0.44 while the raw score held 0.70. A single
   shrinkage parameter is monotone, so ranking survives, and λ is scaled by the
   number of recession **events** seen, so with no evidence it goes to zero and
   the engine simply quotes climatology.

**Skill is measured against a point-in-time climatology.** The conventional
reference — the realised frequency of the window being scored — is an oracle: a
forecaster in 1995 could not know how many recessions 1990–2026 would contain.
Both numbers are reported (`brier_skill` and `brier_skill_vs_full_sample`); the
gap between them is the cost of not knowing the future base rate, which is not a
forecasting error.

Calibration is doing very heavy lifting: raw Brier skill at 3 months is **−1.50**,
calibrated it is **−0.004**.

### What is still NOT good here, stated plainly

- **The model's value is ranking and lead time, not squared error.** AUC 0.62–0.72
  and a 4–18 month lead are the usable outputs. It does not beat climatology on
  Brier, and it is not going to.
- **AUC decays with horizon** — 0.72 at 3 months down to 0.62 at 18. The
  18-month number is close enough to chance that the forecast there leans mostly
  on the base rate by design (`SIGNAL_WEIGHT` = 0.50).
- **63% of model inputs are vintage-true.** Four features (ISRATIO, UMCSENT,
  ALTSALES, NFCI) have no ALFRED coverage and are lag-adjusted — publication
  timing respected, revisions not reproduced. Named explicitly on the Model
  Performance page rather than hidden behind a blanket disclaimer.
- **Precision/recall are mostly null**, because the calibrated probability rarely
  crosses 0.5. On a rare event that is a demanding cut; AUC and Brier skill are
  the primary metrics.
- **Three recessions is a tiny sample.** Every number above has wide error bars
  that no amount of method fixes.

### Three measurement traps this repo fell into and fixed

Worth recording, because each produced a plausible-looking wrong number:

1. **AUC on the calibrated probability** collapsed to 0.41 — below chance — and
   looked like model failure. A single monotone transform cannot change AUC, but
   the point-in-time calibrator is *refitted at every as-of date*, so pooling
   across dates scrambles the ranking. AUC is now computed on the raw score,
   Brier/ECE on the calibrated probability.
2. **Lead time at a fixed 0.35 threshold** reported zero detections after
   calibration, because calibrated probabilities against a 9% base rate rarely
   reach 0.35. It now triggers at a multiple of the horizon's own base rate.
3. **Stale release dates.** After the quality engine caught four series declared
   at the wrong frequency, the leak detector immediately flagged all four —
   correctly, because `release_date` is derived data and had not been recomputed.
   `recompute_release_dates()` now runs on every ingestion pass.
4. **A refit-per-date calibrator destroying the ranking.** Free Platt/isotonic
   refits at each as-of date fixed over-sharpness but scrambled the pooled
   ordering (AUC 0.70 → 0.44). Replaced with a monotone one-parameter shrinkage.
5. **Benchmarking a real-time engine against an oracle.** Brier skill against the
   scored window's realised frequency charges the engine for not knowing the
   future base rate. Now measured against a point-in-time climatology, with the
   conventional number reported alongside.
6. **A `datetime64[s]` vs nanosecond mismatch** in the climatology helper put
   every base rate at ~0.9 instead of ~0.03. pandas 3.0 changed the default
   resolution; the fix uses `DatetimeIndex.searchsorted` and has no resolution
   assumption to get wrong.

---

## Run cadence (spec §25)

| Command | When | Does |
|---|---|---|
| `main.py daily` | 06:00 | pull, QA, recalculate, snapshot, alerts (uses cached fits — seconds) |
| `main.py event --release cpi` | after a major release | refresh only the affected series, recalculate |
| `main.py weekly` | weekly | full forecast + scenario refresh, retrain |
| `main.py monthly` | monthly | + vintage refresh, backtest, calibration, diagnostics |
| `main.py quarterly` | quarterly | + full model/feature/weight review |

Releases understood by `--release`: `cpi`, `employment`, `claims`, `gdp`,
`retail`, `housing`, `ism`, `fed`, `credit`.

A full walk-forward across five models and five horizons takes minutes, so the
daily run reuses a cached fit keyed by a content hash of the feature matrix — any
change to the data or the feature set invalidates it automatically.

---

## Honest limitations

- **ISM is not free.** ISM withdrew its PMI from FRED's public catalogue. The
  engine substitutes the Empire State and Philadelphia Fed manufacturing surveys
  — genuinely leading, published earlier, and free.
- **CAPE is not included.** No keyless source carries it, and the engine will not
  put a scraped page on the critical path. Valuation is proxied from the equity
  market's own change percentile and the real 10-year yield. Per spec §19, CAPE
  would belong in the *market* layer regardless — never in the recession
  probability.
- **Existing home sales** is retained but truncated to ~12 months on FRED (NAR
  licensing). The transform layer refuses to z-score it, so it contributes to
  breadth and never to a percentile it cannot support.
- **~8 usable recessions.** Every metric here rests on a single-digit number of
  independent episodes. `backtest.sanity_flags()` explicitly flags any AUC above
  0.95 as more likely a leak than skill.
- **NBER dates lag by up to a year**, so the most recent months of the historical
  record are provisional.
- **Not investment advice.** The asset panel is an implication layer. Macro
  relationships are regime-dependent and do invert.

---

## Tests

```bash
python -m pytest tests/ -q
```

114 tests, no network, no production database. Covering spec §23 in full: schema,
API failures, missing/duplicate/stale observations, frequency conversion, date
and release-date alignment, vintage selection, **future-data leakage**, signal
ranges, weight sums, probability bounds, factor score ranges, calibration
behaviour, and model reproducibility.

The leakage tests are the ones that matter. They check all three leak channels
separately, and they test the *failure mode* as well as the fix — a guard that
has never been seen to fail is not a guard.

---

## Reproducibility

Every run records `input_hash` (a hash of every value consumed) and `output_hash`
(a hash of the results), alongside `model_version` and `feature_version`. Two runs
with the same `input_hash` saw identical data, so any difference in `output_hash`
is a code or configuration change, not a data change.

`runlog.reproducibility_check(conn, run_a, run_b)` turns "the number changed" into
a specific attribution.

---

## Configuration

Everything a reviewer would want to audit is in `config/strategy.py`: factor
weights, curve-state risk ordering, inflation bands, regime rules, horizon base
rates, model hyperparameters, calibration settings, scenario splits, confidence
weights, asset sensitivities and alert thresholds. No environment access, no
filesystem, no network — pure numbers, so a run is reproducible from
(code version + that file).

Secrets, if you ever add any, go in `.env` (git-ignored). See `.env.example`.
