#!/usr/bin/env python3
"""
Macro Forecasting Engine -- command line entry point.

The run cadence follows spec 25 exactly, and the split matters for cost: the
daily run reuses cached model fits and finishes in seconds, while the expensive
walk-forward retraining happens on the monthly and quarterly cadences the spec
prescribes.

    python main.py daily        06:00 pull, QA, recalculate, snapshot, alerts
    python main.py event --release cpi   recalc after a major release
    python main.py weekly       full forecast + scenario refresh
    python main.py monthly      + backtest refresh, calibration, diagnostics
    python main.py quarterly    + model/feature/weight review, retraining
    python main.py backtest     vintage-true historical backtest
    python main.py report       print the daily text report
    python main.py health       source reachability + data health
    python main.py export       write the dashboard JSON feed

Every command is safe to re-run. Nothing here deletes data.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings                                     # noqa: E402
from config import strategy as st                               # noqa: E402
from modules import (alerts, backtest, db, ingestion, quality,   # noqa: E402
                     report, runlog, snapshot, sources, vintage)
from modules.log import log_error, log_info, log_warning        # noqa: E402

# Which factors a given release moves. Spec 25's event-driven map, made explicit
# so an unrecognised release name fails loudly instead of silently recalculating
# nothing.
RELEASE_IMPACTS = {
    "cpi": ["CPIAUCSL", "CPILFESL", "CUSR0000SASLE", "CUSR0000SACL1E", "PPIACO"],
    "employment": ["PAYEMS", "UNRATE", "U6RATE", "CES0500000003", "AWHMAN",
                   "TEMPHELPS", "CIVPART"],
    "claims": ["ICSA", "CCSA"],
    "gdp": ["PCEC96", "DSPIC96", "PSAVERT"],
    "retail": ["RRSFS", "TOTALSL"],
    "housing": ["HOUST", "PERMIT", "HOUST1F", "HSN1F", "MSACSR", "CSUSHPINSA"],
    "ism": ["GACDISA066MSFRBNY", "GACDFSA066MSFRBPHI", "INDPRO", "IPMAN", "TCU"],
    "fed": ["DFF", "FEDFUNDS", "WALCL", "DGS10", "DGS2", "DGS3MO", "T10Y2Y", "T10Y3M"],
    "credit": ["BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A3HYC", "NFCI", "ANFCI", "STLFSI4"],
}


def _run_engine(conn, run_type: str, *, retrain: bool, export: bool = True,
                print_report: bool = False) -> dict:
    """The shared body of every scheduled command."""
    with runlog.RunContext(conn, run_type) as run:
        snap, artifacts = snapshot.build(conn, retrain=retrain, run_type=run_type)
        snapshot.persist(conn, snap, run, artifacts)

        raised = alerts.persist(conn, snap["meta"]["as_of"], alerts.evaluate(
            conn, snap["meta"]["as_of"], artifacts["regime"], artifacts["factors"],
            artifacts["horizons"], snap["leading"]["breadth"], snap["health"],
            snap.get("revisions")))
        snap["alerts_raised"] = [
            {"kind": a.kind, "severity": a.severity, "title": a.title, "detail": a.detail}
            for a in raised]

    if export:
        write_feed(snap)
    if print_report:
        print(report.render(snap))
    else:
        log_info("[main] " + report.render_compact(snap))
    return snap


def write_feed(snap: dict) -> Path:
    """Write the dashboard JSON feed to both the engine's data/ and, when the
    sibling dashboard repo is present, straight into its data/ directory.

    Writing to a temporary file and replacing keeps the feed atomic: the
    dashboard fetches this file on a timer and must never read a half-written
    JSON document.
    """
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = settings.DATA_DIR / "macro_engine.json"
    # Indented locally (a human reads this one when debugging); COMPACT for the
    # dashboard, which is downloaded by every visitor on every load. indent=2 and
    # the default ", " separators cost roughly 40% of the payload for whitespace
    # the browser discards.
    pretty = json.dumps(snap, indent=2, ensure_ascii=False, default=str)
    compact = json.dumps(snap, ensure_ascii=False, default=str, separators=(",", ":"))

    targets = [(local, pretty)]
    if settings.DASHBOARD_DIR.exists():
        settings.DASHBOARD_FEED.parent.mkdir(parents=True, exist_ok=True)
        targets.append((settings.DASHBOARD_FEED, compact))
    else:
        log_warning(f"[main] dashboard dir not found at {settings.DASHBOARD_DIR}; "
                    f"wrote the feed locally only")

    for path, payload in targets:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
        log_info(f"[main] wrote feed -> {path} ({len(payload):,} bytes)")
    return local


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_daily(conn, args) -> int:
    """Spec 25: pull, validate, update cache, check releases, recalculate."""
    rep = ingestion.update(conn)
    ingestion.rebuild_release_calendar(conn)
    if rep.failed:
        log_warning(f"[daily] {len(rep.failed)} series failed: "
                    + ", ".join(r.sid for r in rep.failed))
    _run_engine(conn, "daily", retrain=False, print_report=args.print_report)
    return 0


def cmd_event(conn, args) -> int:
    """Spec 25: immediately recalculate the factors a major release moves."""
    key = (args.release or "").lower()
    if key not in RELEASE_IMPACTS:
        log_error(f"[event] unknown release {args.release!r}; "
                  f"expected one of {', '.join(sorted(RELEASE_IMPACTS))}")
        return 2
    sids = RELEASE_IMPACTS[key]
    log_info(f"[event] {key} release -> refreshing {len(sids)} series")
    rep = ingestion.update(conn, only=sids)
    log_info(f"[event] {rep.summary()}")
    ingestion.rebuild_release_calendar(conn)
    _run_engine(conn, "event", retrain=False, print_report=args.print_report)
    return 0


def cmd_weekly(conn, args) -> int:
    """Spec 25: full forecast + scenario refresh + model comparison."""
    ingestion.update(conn)
    ingestion.rebuild_release_calendar(conn)
    _run_engine(conn, "weekly", retrain=True, print_report=args.print_report)
    return 0


def cmd_monthly(conn, args) -> int:
    """Spec 25: backtest refresh, calibration, diagnostics, drift."""
    ingestion.update(conn)
    ingestion.rebuild_release_calendar(conn)
    log_info("[monthly] refreshing the ALFRED vintage grid")
    vintage.backfill_vintages(conn, limit=args.vintage_limit)
    snap = _run_engine(conn, "monthly", retrain=True, print_report=False)

    log_info("[monthly] running the vintage-true backtest")
    result = backtest.run_historical_backtest(conn, months=args.months)
    # persist_result() stored the full metrics, so re-reading the snapshot picks
    # the backtest up through the normal path rather than a special case here.
    snap["backtest"] = backtest.latest_backtest_summary(conn)
    write_feed(snap)
    print(json.dumps(result.metrics.get("by_horizon", {}), indent=2, default=str))
    for flag in result.metrics.get("sanity", []):
        log_info(f"[monthly] sanity: {flag}")
    return 0


def cmd_quarterly(conn, args) -> int:
    """Spec 25: model review, feature review, weight review, retraining."""
    ingestion.update(conn, full=True)
    ingestion.rebuild_release_calendar(conn)
    vintage.backfill_vintages(conn, limit=args.vintage_limit)

    log_info("[quarterly] walk-forward model comparison across all horizons")
    comparison = backtest.run_walk_forward_validation(conn)
    print(json.dumps({h: {"selected": v["selected"],
                          "models": {m: d["auc"] for m, d in v["models"].items()}}
                      for h, v in comparison["horizons"].items()}, indent=2))

    snap = _run_engine(conn, "quarterly", retrain=True, print_report=False)
    snap["walk_forward"] = comparison
    write_feed(snap)
    return 0


def cmd_backtest(conn, args) -> int:
    result = backtest.run_historical_backtest(
        conn, start=args.start, end=args.end, months=args.months,
        model_name=args.model)
    starts = backtest.recession_start_dates(quality.load_rows(conn, "USREC"))
    out = {
        "backtest_id": result.backtest_id,
        "points": len(result.points),
        "metrics": result.metrics,
        "cycles": backtest.test_historical_cycles(result, starts),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_report(conn, args) -> int:
    snap, _ = snapshot.build(conn, retrain=False, run_type="report",
                             include_history=False)
    print(report.render(snap))
    return 0


def cmd_health(conn, args) -> int:
    probe = sources.probe()
    q = quality.assess_all(conn, persist=False)
    h = quality.health_summary(q)
    cov = vintage.coverage_report(conn)
    print(json.dumps({
        "sources": probe,
        "data_health": h,
        "vintage_store": {"rows": cov["total_vintage_rows"],
                          "tier_a_covered": cov["tier_a_covered"],
                          "tier_a_series": cov["tier_a_series"]},
        "problem_series": [
            {"sid": s.sid, "grade": s.grade, "score": s.score, "issues": s.issues}
            for s in sorted(q.values(), key=lambda x: x.score) if s.grade != "HIGH"],
        "recent_runs": runlog.latest_runs(conn, 5),
    }, indent=2, default=str))
    return 0 if all(probe.values()) and h["critical"] == 0 else 1


def cmd_export(conn, args) -> int:
    snap, _ = snapshot.build(conn, retrain=False, run_type="export")
    write_feed(snap)
    print(report.render_compact(snap))
    return 0


def cmd_ingest(conn, args) -> int:
    sids = [s.strip() for s in args.series.split(",") if s.strip()] or None
    rep = ingestion.update(conn, full=args.full, only=sids)
    ingestion.rebuild_release_calendar(conn)
    print(rep.summary())
    return 0 if not rep.failed else 1


COMMANDS = {
    "daily": cmd_daily, "event": cmd_event, "weekly": cmd_weekly,
    "monthly": cmd_monthly, "quarterly": cmd_quarterly, "backtest": cmd_backtest,
    "report": cmd_report, "health": cmd_health, "export": cmd_export,
    "ingest": cmd_ingest,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--release", help="event command: which release fired "
                                      f"({', '.join(sorted(RELEASE_IMPACTS))})")
    ap.add_argument("--start", default=settings.BACKTEST_START, help="backtest start")
    ap.add_argument("--end", default=None, help="backtest end")
    ap.add_argument("--months", type=int, default=3, help="backtest grid spacing")
    ap.add_argument("--model", default="rule", help="backtest model")
    ap.add_argument("--series", default="", help="ingest: limit to these series ids")
    ap.add_argument("--full", action="store_true", help="ingest: full history")
    ap.add_argument("--vintage-limit", type=int, default=400,
                    help="max ALFRED vintage requests per scheduled run")
    ap.add_argument("--print-report", action="store_true",
                    help="print the full text report instead of the one-line summary")
    args = ap.parse_args(argv)

    conn = db.get_db()
    try:
        return COMMANDS[args.command](conn, args)
    except Exception as e:
        log_error(f"[main] {args.command} failed: {type(e).__name__}: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
