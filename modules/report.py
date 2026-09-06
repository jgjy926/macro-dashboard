"""
Text report (spec 35) -- the daily terminal/email digest.

Pure formatting: it takes an already-computed snapshot dict and renders it.
Nothing here computes a number, so the report can never disagree with the
dashboard or the database -- they all read the same snapshot.

ASCII only. This runs on Windows consoles that default to cp1252, where a stray
box-drawing character raises UnicodeEncodeError and kills the run -- the exact
failure the sibling KLSE_Monitor project documents in its fetch_macro.py.
"""
from __future__ import annotations

WIDTH = 72
RULE = "=" * WIDTH
THIN = "-" * WIDTH

ARROWS = {"^^^": "UP UP UP", "^^": "UP UP", "^": "UP", "->": "FLAT",
          "v": "DN", "vv": "DN DN", "vvv": "DN DN DN", "?": "?"}


def _center(text: str) -> str:
    return text.center(WIDTH)


def _kv(label: str, value: str, width: int = 22) -> str:
    return f"{label:<{width}}{value}"


def render(snap: dict) -> str:
    """Render the spec 35 daily report from a snapshot dict."""
    L: list[str] = []
    meta = snap.get("meta", {})
    headline = snap.get("headline", {})

    L += [RULE, _center("MACRO FORECAST"),
          _center(meta.get("as_of_display", meta.get("as_of", ""))), RULE, ""]

    L += ["REGIME", f"  {headline.get('regime_display', '?')}", ""]
    L += ["12M RECESSION PROBABILITY",
          f"  {headline.get('recession_12m_pct', '?')}", ""]
    L += ["CONFIDENCE",
          f"  {headline.get('confidence_pct', '?')}  ({headline.get('confidence_band', '?')})", ""]
    L += ["TREND", f"  {headline.get('trend', '?')}", THIN, ""]

    # -- recession probabilities ------------------------------------------
    L += ["RECESSION PROBABILITY", ""]
    L.append(f"  {'HORIZON':<10}{'CALIBRATED':<14}{'RAW SCORE':<13}{'MODEL':<10}AUC")
    for h in snap.get("recession", {}).get("horizons", []):
        auc = h.get("auc")
        L.append(f"  {str(h['horizon_m']) + 'M':<10}"
                 f"{h['probability'] * 100:>6.1f}%       "
                 f"{h['raw_score'] * 100:>6.1f}%      "
                 f"{h['model']:<10}{auc if auc is not None else '-'}")
    if not snap.get("recession", {}).get("calibrated_all", True):
        L += ["", "  NOTE: at least one horizon is UNCALIBRATED -- see the caveats below."]
    L += ["", THIN, ""]

    # -- factors ------------------------------------------------------------
    L += ["MACRO FACTORS", ""]
    for f in snap.get("factors", []):
        score = f.get("score")
        L.append(f"  {f['label']:<18}{score:>+7.2f}   {ARROWS.get(f.get('arrow', '?'), '?')}"
                 if score is not None else
                 f"  {f['label']:<18}{'N/A':>7}   (coverage {f.get('coverage', 0):.0%})")
    L += ["", THIN, ""]

    # -- breadth -------------------------------------------------------------
    b = snap.get("leading", {}).get("breadth", {})
    L += ["LEADING INDICATORS", ""]
    L += [f"  Weakening:  {b.get('weakening', 0):>3} / {b.get('total', 0)}",
          f"  Neutral:    {b.get('neutral', 0):>3} / {b.get('total', 0)}",
          f"  Improving:  {b.get('improving', 0):>3} / {b.get('total', 0)}",
          "",
          f"  Breadth:    {b.get('weakening_pct', 0):.0%} weakening"]
    div = snap.get("leading", {}).get("divergence", {})
    if div.get("diverging"):
        L += ["", f"  DIVERGENCE ({div.get('kind')}): leading {div.get('leading'):+.2f} "
                  f"vs coincident {div.get('coincident'):+.2f}"]
    L += ["", THIN, ""]

    # -- scenarios -----------------------------------------------------------
    L += ["SCENARIOS", ""]
    for s in snap.get("scenarios", {}).get("scenarios", []):
        L.append(f"  {s['scenario']:<10}{s['probability'] * 100:>5.0f}%   {s['headline'][:44]}")
    L += ["", THIN, ""]

    # -- drivers -------------------------------------------------------------
    rec = snap.get("recommendation", {})
    L += ["TOP RISKS", ""]
    for i, d in enumerate(rec.get("drivers", []), 1):
        L.append(f"  {i}. {d['label']} ({d['signal']:+.2f})")
    L += ["", "OFFSETS", ""]
    for i, d in enumerate(rec.get("offsetting", []), 1):
        L.append(f"  {i}. {d['label']} ({d['signal']:+.2f})")
    L += ["", THIN, ""]

    # -- market --------------------------------------------------------------
    L += ["MARKET IMPLICATION", ""]
    for a in snap.get("market", {}).get("assets", []):
        L.append(f"  {a['asset']:<22}{a['stance']}")
    L += ["", THIN, ""]

    # -- invalidation --------------------------------------------------------
    L += ["FORECAST INVALIDATION", "", "IMPROVES IF:"]
    for c in rec.get("improves_if", []):
        L.append(f"  + {c}")
    L += ["", "WORSENS IF:"]
    for c in rec.get("worsens_if", []):
        L.append(f"  ! {c}")
    L += ["", THIN, ""]

    # -- caveats -------------------------------------------------------------
    if rec.get("caveats"):
        L += ["CAVEATS", ""]
        for c in rec["caveats"]:
            L.append(f"  * {c}")
        L += ["", THIN, ""]

    # -- health --------------------------------------------------------------
    h = snap.get("health", {})
    L += ["DATA HEALTH",
          f"  {h.get('health', 0):.0%}   "
          f"({h.get('high', 0)} HIGH / {h.get('medium', 0)} MEDIUM / {h.get('low', 0)} LOW"
          f", {h.get('stale', 0)} stale)"]
    if h.get("stale_series"):
        L.append(f"  Stale: {', '.join(h['stale_series'][:6])}")
    # `main.py report` renders without opening a RunContext, so there is no run
    # id or hash to show. Say that, rather than printing a bare "?" that reads
    # like a missing value.
    run_id = meta.get("run_id")
    L += ["",
          _kv("MODEL VERSION", meta.get("model_version", "?")),
          _kv("RUN ID", str(run_id) if run_id else "not recorded (read-only report)"),
          _kv("INPUT HASH", (meta.get("input_hash") or "-")[:16]),
          _kv("LAST UPDATE", meta.get("generated_at", "?")),
          "", RULE]
    return "\n".join(L)


def render_compact(snap: dict) -> str:
    """One-line summary, for a cron log or a notification."""
    hl = snap.get("headline", {})
    h = snap.get("health", {})
    return (f"{snap.get('meta', {}).get('as_of', '?')} | {hl.get('regime', '?')} | "
            f"12m recession {hl.get('recession_12m_pct', '?')} | "
            f"confidence {hl.get('confidence_pct', '?')} ({hl.get('confidence_band', '?')}) | "
            f"trend {hl.get('trend', '?')} | data health {h.get('health', 0):.0%}")
