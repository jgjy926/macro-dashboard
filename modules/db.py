"""
Database layer -- the single source of truth for the SQLite schema (spec 5).

All tables are created with CREATE TABLE IF NOT EXISTS so init() is idempotent.
connect() returns a sqlite3.Connection with a row factory and foreign keys on.
Pass ":memory:" for tests.

Storage split (spec 5, "Use Parquet for larger historical datasets if required"):
  observations  -- current best value per (series_id, observation_date). Small:
                   ~81 series x at most daily since 1970 is a few hundred
                   thousand rows, which SQLite handles comfortably.
  vintages      -- (series_id, vintage_date, observation_date) -> value. This is
                   the one table that grows combinatorially, so it is written
                   only for series flagged vintage='alfred' in the catalogue and
                   only on the quarterly vintage grid the backtest actually asks
                   for. Parquet is not needed at this scale; if it ever is, the
                   read path is confined to modules/vintage.py.

Modelled on the sibling KLSE_Monitor project's modules/db.py (ordered
SCHEMA_STATEMENTS list, idempotent init, connect() helper).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from config import settings

SCHEMA_STATEMENTS: list[str] = [
    # --- catalogue / metadata -------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS series_metadata (
        series_id          TEXT PRIMARY KEY,
        label              TEXT NOT NULL,
        category           TEXT NOT NULL,
        frequency          TEXT NOT NULL,
        direction          INTEGER NOT NULL,
        role               TEXT NOT NULL,
        transform          TEXT NOT NULL,
        expected_lag_days  INTEGER NOT NULL,
        vintage_policy     TEXT NOT NULL,
        source             TEXT NOT NULL,
        unit               TEXT,
        is_core            INTEGER NOT NULL DEFAULT 0,
        note               TEXT,
        resolved_id        TEXT,       -- which candidate id actually returned data
        first_obs          TEXT,
        last_obs           TEXT,
        obs_count          INTEGER,
        last_fetch_ok      TIMESTAMP,
        last_fetch_attempt TIMESTAMP
    )
    """,
    # --- observations: current (latest-vintage) values -------------------
    # release_date is the date the value first became public. Known exactly for
    # ALFRED-backed series; estimated as observation_date + expected_lag_days
    # for the rest, with release_estimated flagging which is which so no caller
    # can mistake an estimate for a fact.
    """
    CREATE TABLE IF NOT EXISTS observations (
        series_id         TEXT NOT NULL,
        observation_date  TEXT NOT NULL,
        value             REAL,
        release_date      TEXT,
        release_estimated INTEGER NOT NULL DEFAULT 1,
        frequency         TEXT NOT NULL,
        source            TEXT NOT NULL,
        fetched_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (series_id, observation_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_obs_date ON observations(observation_date)",
    "CREATE INDEX IF NOT EXISTS idx_obs_release ON observations(series_id, release_date)",
    # --- vintages: what the number LOOKED LIKE on a past date ------------
    """
    CREATE TABLE IF NOT EXISTS vintages (
        series_id         TEXT NOT NULL,
        vintage_date      TEXT NOT NULL,
        observation_date  TEXT NOT NULL,
        value             REAL,
        fetched_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (series_id, vintage_date, observation_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_vintage_lookup ON vintages(series_id, vintage_date)",
    # Which (series, vintage_date) pairs we have actually pulled. Without this
    # we could not tell "ALFRED had no data for that vintage" from "we never
    # asked", and the backtest would silently re-request the same empty vintage
    # on every run.
    """
    CREATE TABLE IF NOT EXISTS vintage_coverage (
        series_id     TEXT NOT NULL,
        vintage_date  TEXT NOT NULL,
        obs_count     INTEGER NOT NULL,
        fetched_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (series_id, vintage_date)
    )
    """,
    # --- release calendar -------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS releases (
        series_id        TEXT NOT NULL,
        release_date     TEXT NOT NULL,
        observation_date TEXT NOT NULL,
        value            REAL,
        is_revision      INTEGER NOT NULL DEFAULT 0,
        prior_value      REAL,
        PRIMARY KEY (series_id, release_date, observation_date)
    )
    """,
    # --- revisions detected on ingest ------------------------------------
    """
    CREATE TABLE IF NOT EXISTS revisions (
        series_id        TEXT NOT NULL,
        observation_date TEXT NOT NULL,
        detected_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        old_value        REAL,
        new_value        REAL,
        pct_change       REAL,
        PRIMARY KEY (series_id, observation_date, detected_at)
    )
    """,
    # --- data quality -----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS data_quality (
        series_id        TEXT NOT NULL,
        assessed_at      TIMESTAMP NOT NULL,
        grade            TEXT NOT NULL,          -- HIGH | MEDIUM | LOW
        score            REAL NOT NULL,          -- 0..1
        last_observation TEXT,
        last_release     TEXT,
        observation_age  INTEGER,                -- days
        missing_pct      REAL,
        is_stale         INTEGER NOT NULL DEFAULT 0,
        revision_status  TEXT,
        issues           TEXT,                   -- JSON array of issue strings
        PRIMARY KEY (series_id, assessed_at)
    )
    """,
    # --- transformed / normalised signals --------------------------------
    """
    CREATE TABLE IF NOT EXISTS transformations (
        series_id        TEXT NOT NULL,
        observation_date TEXT NOT NULL,
        transform        TEXT NOT NULL,
        raw_value        REAL,
        transformed      REAL,
        zscore           REAL,
        percentile       REAL,
        signal           REAL,                   -- direction-adjusted, -1..+1
        PRIMARY KEY (series_id, observation_date, transform)
    )
    """,
    # --- factor scores ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS factor_scores (
        as_of_date   TEXT NOT NULL,
        factor       TEXT NOT NULL,
        score        REAL NOT NULL,              -- -1..+1, + = economically positive
        momentum     REAL,                       -- 3m change in score
        n_inputs     INTEGER,
        coverage     REAL,                       -- share of category series available
        detail       TEXT,                       -- JSON: per-series contributions
        run_id       INTEGER,
        PRIMARY KEY (as_of_date, factor, run_id)
    )
    """,
    # --- regimes ----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS regimes (
        as_of_date       TEXT NOT NULL,
        regime           TEXT NOT NULL,
        growth_regime    TEXT,
        inflation_regime TEXT,
        labor_regime     TEXT,
        credit_regime    TEXT,
        liquidity_regime TEXT,
        policy_regime    TEXT,
        strength         REAL,
        trend            TEXT,
        detail           TEXT,
        run_id           INTEGER,
        PRIMARY KEY (as_of_date, run_id)
    )
    """,
    # --- forecasts --------------------------------------------------------
    # raw_score and probability are stored side by side precisely because spec 13
    # forbids presenting a weighted score AS a probability. Both are kept so the
    # dashboard can show the uncalibrated risk score and the calibrated
    # probability as the different objects they are.
    """
    CREATE TABLE IF NOT EXISTS forecasts (
        as_of_date   TEXT NOT NULL,
        horizon_m    INTEGER NOT NULL,
        target       TEXT NOT NULL,              -- recession | gdp | cpi | unrate | ...
        model        TEXT NOT NULL,
        raw_score    REAL,
        probability  REAL,
        point        REAL,
        lo           REAL,
        hi           REAL,
        direction    TEXT,
        confidence   REAL,
        drivers      TEXT,                       -- JSON
        run_id       INTEGER,
        PRIMARY KEY (as_of_date, horizon_m, target, model, run_id)
    )
    """,
    # --- scenarios --------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS scenarios (
        as_of_date  TEXT NOT NULL,
        scenario    TEXT NOT NULL,               -- BASE | BULL | BEAR | CRISIS
        probability REAL NOT NULL,
        detail      TEXT NOT NULL,               -- JSON: growth/inflation/rates/...
        run_id      INTEGER,
        PRIMARY KEY (as_of_date, scenario, run_id)
    )
    """,
    # --- model runs (spec 31: reproducibility) ---------------------------
    """
    CREATE TABLE IF NOT EXISTS model_runs (
        run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        run_type        TEXT NOT NULL,           -- daily|weekly|monthly|quarterly|backtest
        status          TEXT NOT NULL DEFAULT 'RUNNING',
        data_timestamp  TEXT,
        model_version   TEXT,
        feature_version TEXT,
        data_vintage    TEXT,
        input_hash      TEXT,
        output_hash     TEXT,
        regime          TEXT,
        prob_12m        REAL,
        confidence      REAL,
        data_health     REAL,
        series_ok       INTEGER,
        series_stale    INTEGER,
        series_failed   INTEGER,
        api_errors      TEXT,
        duration_sec    REAL,
        error           TEXT
    )
    """,
    # --- backtest ---------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS backtest_results (
        backtest_id   TEXT NOT NULL,
        as_of_date    TEXT NOT NULL,
        horizon_m     INTEGER NOT NULL,
        model         TEXT NOT NULL,
        probability   REAL,                      -- published, point-in-time calibrated
        uncalibrated  REAL,                      -- before calibration; AUC is scored on THIS
        raw_score     REAL,                      -- the model's own output, pre-blending
        actual        INTEGER,                   -- 1 if recession within horizon
        vintage_true  INTEGER NOT NULL DEFAULT 0,-- 1 = every model input vintage-true
        vintage_share REAL,                      -- share of model inputs that are
        climatology   REAL,                      -- point-in-time base rate it competed against
        PRIMARY KEY (backtest_id, as_of_date, horizon_m, model)
    )
    """,
    # The full metrics structure for one backtest. backtest_metrics holds the
    # scalars (queryable, one row per metric); this holds the whole nested result
    # -- calibration curves, lead-time episodes, sanity flags -- so the dashboard
    # can render the last backtest on ANY run, not only the monthly one that
    # produced it.
    """
    CREATE TABLE IF NOT EXISTS backtest_summary (
        backtest_id  TEXT PRIMARY KEY,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        start_date   TEXT,
        end_date     TEXT,
        model        TEXT,
        n_points     INTEGER,
        metrics_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_metrics (
        backtest_id  TEXT NOT NULL,
        horizon_m    INTEGER NOT NULL,
        model        TEXT NOT NULL,
        metric       TEXT NOT NULL,
        value        REAL,
        PRIMARY KEY (backtest_id, horizon_m, model, metric)
    )
    """,
    # Audit trail proving a backtest saw no future data (spec 7).
    """
    CREATE TABLE IF NOT EXISTS backtest_audit (
        backtest_id      TEXT NOT NULL,
        as_of_date       TEXT NOT NULL,
        series_id        TEXT NOT NULL,
        vintage_used     TEXT,
        source_kind      TEXT,                   -- alfred | final | lag
        last_obs_used    TEXT,
        latest_release   TEXT,
        leak_detected    INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (backtest_id, as_of_date, series_id)
    )
    """,
    # --- alerts -----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        raised_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        as_of_date   TEXT NOT NULL,
        kind         TEXT NOT NULL,
        severity     TEXT NOT NULL,              -- INFO | WARNING | HIGH | CRITICAL
        title        TEXT NOT NULL,
        detail       TEXT,
        acknowledged INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(as_of_date)",
    # Dedup key so a persistent condition (e.g. curve still inverted) raises one
    # alert per day, not one per run.
    """
    CREATE TABLE IF NOT EXISTS alert_dedup (
        kind       TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        PRIMARY KEY (kind, as_of_date)
    )
    """,
    # --- API error log (spec 29: never fail silently) --------------------
    """
    CREATE TABLE IF NOT EXISTS api_errors (
        logged_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source     TEXT NOT NULL,
        series_id  TEXT,
        url        TEXT,
        error      TEXT NOT NULL,
        attempt    INTEGER
    )
    """,
    # --- fitted model coefficients (so a run is reproducible offline) ----
    """
    CREATE TABLE IF NOT EXISTS model_artifacts (
        model        TEXT NOT NULL,
        horizon_m    INTEGER NOT NULL,
        trained_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        train_end    TEXT,
        features     TEXT,                       -- JSON list
        params       TEXT,                       -- JSON
        metrics      TEXT,                       -- JSON: in/out-of-sample scores
        PRIMARY KEY (model, horizon_m, trained_at)
    )
    """,
]


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = str(db_path if db_path is not None else settings.DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets the dashboard exporter read while a run is writing. Harmless on
    # :memory: (SQLite ignores it there).
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init(conn: sqlite3.Connection) -> None:
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    conn.commit()


def sync_series_metadata(conn: sqlite3.Connection) -> int:
    """Push the config/series.py catalogue into series_metadata.

    Catalogue-owned columns are overwritten; runtime columns (resolved_id,
    obs counts, fetch timestamps) are preserved via COALESCE on the existing
    row, so editing a label never wipes ingestion history.
    """
    from config import series as cat

    rows = 0
    for s in cat.ALL_SERIES:
        conn.execute(
            """
            INSERT INTO series_metadata
                (series_id, label, category, frequency, direction, role,
                 transform, expected_lag_days, vintage_policy, source, unit,
                 is_core, note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(series_id) DO UPDATE SET
                label=excluded.label, category=excluded.category,
                frequency=excluded.frequency, direction=excluded.direction,
                role=excluded.role, transform=excluded.transform,
                expected_lag_days=excluded.expected_lag_days,
                vintage_policy=excluded.vintage_policy, source=excluded.source,
                unit=excluded.unit, is_core=excluded.is_core, note=excluded.note
            """,
            (s.sid, s.label, s.category, s.freq, s.direction, s.role,
             s.transform, s.expected_lag_days, s.vintage, s.source, s.unit,
             1 if s.core else 0, s.note),
        )
        rows += 1
    conn.commit()
    return rows


def get_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """connect + init + catalogue sync, in one call. The normal entry point."""
    conn = connect(db_path)
    init(conn)
    sync_series_metadata(conn)
    return conn
