"""
Macro Forecasting Engine — environment / path / runtime settings.

Operational knobs that are NOT modelling parameters live here. Pure model
thresholds and weights live in config/strategy.py; the series catalogue lives
in config/series.py. Secrets are loaded from .env (never committed) via
python-dotenv when available.

Design note: this engine is keyless by default. FRED, ALFRED and BLS all serve
the data we need without an API key, which is what keeps the running cost at
$0/month with no signup. Keys are read if present and unlock strictly optional
upgrades (see FRED_API_KEY / BEA_API_KEY below) — nothing breaks without them.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("MACRO_DB", PROJECT_ROOT / "macro_engine.db"))
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "data_cache"

# Where the static dashboard picks the engine feed up. Defaults to the sibling
# Personal Dynamic Dashboard repo so `tools/export_dashboard.py` works with no
# arguments; override with MACRO_DASHBOARD_DIR when the site lives elsewhere.
DASHBOARD_DIR = Path(os.environ.get(
    "MACRO_DASHBOARD_DIR",
    PROJECT_ROOT.parent / "Personal Dynamic Dashboard"))
DASHBOARD_FEED = DASHBOARD_DIR / "data" / "macro_engine.json"

# --- Versioning (stamped onto every model_run for reproducibility, spec 31) --
MODEL_VERSION = "macro-v1.0.0"
FEATURE_VERSION = "features-v1.0.0"

# When a new training month starts, a daily run may keep scoring with LAST
# month's fitted models instead of paying the ~4-5 minute refit inline. The
# models are flagged as stale (meta.model_stale + RETRAIN_MARKER) and the
# refit runs separately -- see the dashboard repo's refresh workflow. Never
# reused across a feature-set or model-version change: those are different
# models, not older ones.
ALLOW_STALE_MODEL = os.environ.get("MACRO_ALLOW_STALE_MODEL", "") == "1"
RETRAIN_MARKER = CACHE_DIR / "RETRAIN_NEEDED"

# --- Timezone -------------------------------------------------------------
# US macro releases are stamped in US Eastern; the engine's "today" must follow
# the data calendar, not the operator's laptop locale, or a run from Malaysia
# (UTC+8) would look a day ahead of the release calendar.
DATA_TIMEZONE = "America/New_York"
DAILY_RUN_HHMM = (6, 0)          # spec 25: 06:00 daily pull

# --- Network --------------------------------------------------------------
HTTP_TIMEOUT = int(os.environ.get("MACRO_HTTP_TIMEOUT", "60"))
HTTP_RETRIES = int(os.environ.get("MACRO_HTTP_RETRIES", "3"))
HTTP_BACKOFF_BASE = float(os.environ.get("MACRO_HTTP_BACKOFF", "1.5"))  # seconds
USER_AGENT = "Mozilla/5.0 (macro-forecasting-engine; keyless public data)"

# Politeness delay between ALFRED vintage requests. ALFRED publishes no rate
# limit for the keyless graph endpoint, but a vintage backfill issues thousands
# of calls, so we self-throttle rather than risk being cut off mid-backfill.
ALFRED_REQUEST_DELAY = float(os.environ.get("MACRO_ALFRED_DELAY", "0.35"))

# --- Data freshness / staleness ------------------------------------------
# A series is STALE when its newest observation is older than its native
# frequency allows plus the publisher's typical lag. Multipliers are applied to
# the series' own expected_lag_days in config/series.py.
STALE_MULTIPLIER = float(os.environ.get("MACRO_STALE_MULTIPLIER", "2.0"))
CRITICAL_STALE_MULTIPLIER = float(os.environ.get("MACRO_CRITICAL_STALE_MULTIPLIER", "4.0"))

# --- History window -------------------------------------------------------
# Earliest observation the engine keeps. 1970 gives ~8 NBER recessions, enough
# for walk-forward validation without dragging in the pre-1960 data that many
# modern series simply do not have.
HISTORY_START = os.environ.get("MACRO_HISTORY_START", "1970-01-01")
# Backtests start here: by 1985 the credit-spread and survey series that the
# model leans on actually exist, so earlier forecast dates would be scored on a
# materially different (thinner) feature set than the live model uses.
BACKTEST_START = os.environ.get("MACRO_BACKTEST_START", "1985-01-01")

# --- Secrets (loaded from .env) ------------------------------------------
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


_load_dotenv()

# Optional. Present only to unlock upgrades over the keyless path:
#   FRED_API_KEY -> ALFRED's JSON API returns EVERY vintage of a series in one
#                   call, versus one HTTP request per vintage keylessly. Pure
#                   speed; the resulting data is identical.
#   BEA_API_KEY  -> BEA national-accounts detail beyond what FRED mirrors.
# Neither is required and the engine never warns about their absence.
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
BEA_API_KEY = os.environ.get("BEA_API_KEY", "")
BLS_API_KEY = os.environ.get("BLS_API_KEY", "")  # raises BLS daily cap 25 -> 500
