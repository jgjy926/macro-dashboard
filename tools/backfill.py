#!/usr/bin/env python3
"""
One-time (and resumable) historical backfill.

Two independent jobs, either of which can be run alone:

    python tools/backfill.py --observations     # full history, all series
    python tools/backfill.py --vintages         # the ALFRED vintage grid

The vintage job is the slow one: ~23 heavily-revised series across a quarterly
grid since 1985 is a few thousand keyless ALFRED requests, self-throttled by
settings.ALFRED_REQUEST_DELAY. It is resumable -- every (series, vintage) pair
that completes is recorded in vintage_coverage and skipped on the next run -- so
interrupting it is safe and re-running it costs only what is left.

Run this once before trusting the backtest. Without it, backtests fall back to
lag-adjusted data (tier C) and every result is flagged vintage_true=0.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings                      # noqa: E402
from modules import db, ingestion, quality, vintage   # noqa: E402
from modules.log import log_info                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observations", action="store_true", help="full-history observation pull")
    ap.add_argument("--vintages", action="store_true", help="ALFRED vintage grid")
    ap.add_argument("--start", default=settings.BACKTEST_START, help="vintage grid start")
    ap.add_argument("--end", default=None, help="vintage grid end (default: today)")
    ap.add_argument("--months", type=int, default=3, help="vintage grid spacing in months")
    ap.add_argument("--series", default="", help="comma-separated series ids to limit to")
    ap.add_argument("--limit", type=int, default=None, help="stop after N vintage requests")
    args = ap.parse_args()

    if not (args.observations or args.vintages):
        args.observations = args.vintages = True

    conn = db.get_db()
    sids = [s.strip() for s in args.series.split(",") if s.strip()] or None
    t0 = time.time()

    if args.observations:
        report = ingestion.update(conn, full=True, only=sids)
        ingestion.rebuild_release_calendar(conn)
        q = quality.assess_all(conn)
        h = quality.health_summary(q)
        log_info(f"[backfill] observations: {report.summary()}")
        log_info(f"[backfill] data health {h['health']:.1%} "
                 f"(HIGH {h['high']} / MEDIUM {h['medium']} / LOW {h['low']})")

    if args.vintages:
        stats = vintage.backfill_vintages(conn, start=args.start, end=args.end,
                                          months=args.months, sids=sids, limit=args.limit)
        log_info(f"[backfill] vintages: {stats}")
        cov = vintage.coverage_report(conn)
        log_info(f"[backfill] vintage store: {cov['total_vintage_rows']:,} rows, "
                 f"{cov['tier_a_covered']}/{cov['tier_a_series']} tier-A series covered")

    log_info(f"[backfill] done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
