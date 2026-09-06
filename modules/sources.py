"""
Network layer -- every outbound HTTP call in the engine lives here (spec 3, 29).

Everything below is keyless. FRED, ALFRED and BLS all serve the data this engine
needs without an API key, which is what keeps the running cost at $0/month with
no signup:

    FRED    https://fred.stlouisfed.org/graph/fredgraph.csv?id=A,B,C
            Multiple series in ONE request, comma-separated. 81 series become
            ~10 calls instead of 81.
    ALFRED  https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=X&vintage_date=D
            The series exactly as it looked on date D -- real historical
            vintages, keyless. Verified against GDPC1/PAYEMS/HOUST.
            One request per (series, vintage): the comma-separated multi-vintage
            form is silently ignored by this endpoint (it returns only the first
            vintage's column), so callers must not batch vintage dates.
    BLS     https://api.bls.gov/publicAPI/v2/timeseries/data/  (POST, no key)
            Keyless tier: 25 requests/day, 25 series/request, 10-year window.
            Used only as a cross-check on CPI/employment, never as the primary
            path, so hitting the cap degrades a QA check and nothing else.

The fetch/parse split is deliberate and copied from the sibling Personal Dynamic
Dashboard's tools/fetch_macro.py: every function that touches the network is a
thin one-liner, and every parser is pure. That is what makes the parsers
unit-testable without a network, and it is why the tests in tests/ never mock
sockets.

BEA is intentionally absent. It requires a free UserID, and every BEA series the
spec asks for (GDP, real GDP, personal income, PCE, savings, corporate profits)
is mirrored on FRED under an id we already fetch keylessly. Adding BEA would buy
sub-component detail at the cost of the engine's no-signup property.
"""
from __future__ import annotations

import csv
import io
import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

from config import settings
from modules.log import log_error, log_warning

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ALFRED_CSV = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
BLS_API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={rng}&interval=1d"

_HEADERS = {"User-Agent": settings.USER_AGENT, "Accept": "text/csv,application/json,*/*"}


class FetchError(RuntimeError):
    """Raised after all retries are exhausted. Callers catch this per series so
    one dead source can never abort a whole ingestion run (spec 29)."""


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
def _http(url: str, *, data: bytes | None = None, headers: dict | None = None,
          timeout: int | None = None, encoding: str = "utf-8") -> str:
    req = urllib.request.Request(url, data=data, headers={**_HEADERS, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout or settings.HTTP_TIMEOUT) as r:
        return r.read().decode(encoding, "replace")


def retry_with_backoff(fn, *, what: str, conn=None, source: str = "", series_id: str = "",
                       url: str = "", retries: int | None = None):
    """Call fn(), retrying transient failures with exponential backoff.

    A 404 is NOT transient -- it means the series id is wrong or retired -- so it
    fails immediately rather than burning three timeouts on a URL that will never
    work. Every attempt is recorded to api_errors when a connection is supplied,
    because spec 29's rule is that failures must be visible, not merely handled.
    """
    n = retries if retries is not None else settings.HTTP_RETRIES
    last: Exception | None = None
    for attempt in range(1, n + 1):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            last = e
            record_api_error(conn, source, series_id, url, f"HTTP {e.code}", attempt)
            if e.code in (400, 404):
                break
        except Exception as e:  # timeouts, DNS, resets, malformed payloads
            last = e
            record_api_error(conn, source, series_id, url, f"{type(e).__name__}: {e}", attempt)
        if attempt < n:
            time.sleep(settings.HTTP_BACKOFF_BASE ** attempt)
    raise FetchError(f"{what}: {type(last).__name__}: {last}")


def record_api_error(conn, source: str, series_id: str, url: str, error: str,
                     attempt: int) -> None:
    if conn is None:
        return
    try:
        conn.execute(
            "INSERT INTO api_errors (source, series_id, url, error, attempt) VALUES (?,?,?,?,?)",
            (source, series_id or None, url or None, error, attempt))
        conn.commit()
    except Exception:
        # Error logging must never be the thing that breaks a run.
        pass


# ---------------------------------------------------------------------------
# FRED / ALFRED
# ---------------------------------------------------------------------------
def fetch_fred_csv(series_ids: list[str] | str, *, start: str | None = None,
                   conn=None) -> str:
    """One or many FRED series as CSV. Pass a list to batch them into one call.

    Note on `start`: FRED honours cosd for a single-series request but ignores it
    when several ids are batched, so callers must not assume the response is
    trimmed -- parse_fred_csv returns whatever came back and ingestion filters
    by settings.HISTORY_START itself.
    """
    ids = series_ids if isinstance(series_ids, str) else ",".join(series_ids)
    url = f"{FRED_CSV}?id={ids}"
    if start:
        url += f"&cosd={start}"
    return retry_with_backoff(lambda: _http(url), what=f"FRED {ids}", conn=conn,
                              source="FRED", series_id=ids if isinstance(series_ids, str) else "",
                              url=url)


def fetch_alfred_vintage(series_id: str, vintage_date: str, *, start: str | None = None,
                         conn=None) -> str:
    """The series as it stood on `vintage_date` -- a true historical vintage.

    One vintage per call: this endpoint accepts a comma-separated vintage_date
    list without erroring but returns only the first column, so batching here
    would silently produce wrong (single-vintage) data for every date after the
    first. Verified 2026-09-06.
    """
    url = f"{ALFRED_CSV}?id={series_id}&vintage_date={vintage_date}"
    if start:
        url += f"&cosd={start}"
    txt = retry_with_backoff(lambda: _http(url), what=f"ALFRED {series_id}@{vintage_date}",
                             conn=conn, source="ALFRED", series_id=series_id, url=url)
    if settings.ALFRED_REQUEST_DELAY:
        time.sleep(settings.ALFRED_REQUEST_DELAY)
    return txt


def parse_fred_csv(text: str) -> dict[str, list[tuple[str, float]]]:
    """FRED/ALFRED CSV -> {column_name: [(YYYY-MM-DD, value), ...]} ascending.

    Layout is `observation_date,SERIES_A,SERIES_B,...` with missing values as
    '.' (FRED) or '' (ALFRED batches). Column names carry a vintage suffix in
    ALFRED responses (PAYEMS_20080115); callers strip it, this parser does not,
    so nothing here has to know which endpoint produced the text.

    Pure -- no network, no clock. Unit-tested in tests/test_sources.py.
    """
    # newline="" is required: FRED serves CRLF, and StringIO's default universal
    # newline translation leaves a bare \r inside fields that csv then rejects.
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if len(rows) < 2:
        return {}
    header = [h.strip() for h in rows[0]]
    out: dict[str, list[tuple[str, float]]] = {h: [] for h in header[1:]}
    for row in rows[1:]:
        if not row or len(row) < 2:
            continue
        d = row[0].strip()
        if not d or len(d) < 8:
            continue
        for i, col in enumerate(header[1:], start=1):
            if i >= len(row):
                continue
            v = row[i].strip()
            if v in ("", ".", "NA", "null"):
                continue
            try:
                out[col].append((d, float(v)))
            except ValueError:
                continue
    return out


def parse_fred_single(text: str) -> list[tuple[str, float]]:
    """Convenience for a one-series response: the first (only) data column."""
    cols = parse_fred_csv(text)
    for v in cols.values():
        return v
    return []


# ---------------------------------------------------------------------------
# BLS -- keyless cross-check only
# ---------------------------------------------------------------------------
def fetch_bls(series_ids: list[str], start_year: int, end_year: int, *, conn=None) -> dict:
    """BLS API v2 without a key: 25 requests/day, 25 series and 10 years each.

    Used to cross-check FRED's CPI/employment mirrors, so exhausting the daily
    cap costs a QA signal and nothing on the forecast path.
    """
    if end_year - start_year > 9:
        start_year = end_year - 9
    payload = {"seriesid": series_ids[:25],
               "startyear": str(start_year), "endyear": str(end_year)}
    if settings.BLS_API_KEY:
        payload["registrationkey"] = settings.BLS_API_KEY
    body = json.dumps(payload).encode()
    txt = retry_with_backoff(
        lambda: _http(BLS_API, data=body,
                      headers={"Content-Type": "application/json"}),
        what="BLS", conn=conn, source="BLS", url=BLS_API)
    return json.loads(txt)


def parse_bls(payload: dict) -> dict[str, list[tuple[str, float]]]:
    """BLS JSON -> {series_id: [(YYYY-MM-DD, value)]} ascending.

    BLS periods are M01..M12 (monthly), Q01..Q04 (quarterly) and M13/Q05 (annual
    averages). Annual-average rows are dropped: folding a year's average into a
    monthly series would put a smoothed value on a real month's date.
    """
    out: dict[str, list[tuple[str, float]]] = {}
    for s in payload.get("Results", {}).get("series", []):
        sid = s.get("seriesID", "")
        rows: list[tuple[str, float]] = []
        for item in s.get("data", []):
            period, year = item.get("period", ""), item.get("year", "")
            if period in ("M13", "Q05") or not period or not year:
                continue
            try:
                if period.startswith("M"):
                    d = f"{year}-{int(period[1:]):02d}-01"
                elif period.startswith("Q"):
                    d = f"{year}-{(int(period[1:]) - 1) * 3 + 1:02d}-01"
                else:
                    continue
                rows.append((d, float(item["value"])))
            except (ValueError, KeyError):
                continue
        out[sid] = sorted(rows)
    return out


# ---------------------------------------------------------------------------
# Yahoo -- gold fallback only
# ---------------------------------------------------------------------------
def fetch_yahoo_chart(symbol: str, rng: str = "10y", *, conn=None) -> str:
    """FRED retired its LBMA gold fixings, so gold falls back to Yahoo's public
    keyless chart endpoint. Unofficial -- which is exactly why it is a fallback
    and not a primary source, and why the caller treats a failure here as a
    missing optional series rather than a run-ending error."""
    url = YAHOO_CHART.format(symbol=symbol, rng=rng)
    return retry_with_backoff(lambda: _http(url, headers={"Accept": "application/json"}),
                              what=f"Yahoo {symbol}", conn=conn, source="YAHOO",
                              series_id=symbol, url=url)


def parse_yahoo_chart(text: str) -> list[tuple[str, float]]:
    """Yahoo chart JSON: parallel timestamp / close arrays, nulls skipped."""
    data = json.loads(text)
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        return []
    r0 = result[0]
    ts = r0.get("timestamp") or []
    closes = ((r0.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    out = []
    for t, v in zip(ts, closes):
        if v is None:
            continue
        out.append((datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d"), float(v)))
    return out


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------
def probe() -> dict[str, bool]:
    """Quick reachability check for each source. Used by main.py's health
    command and by the dashboard's system-health page so a network problem is
    reported as a network problem, not as a mysteriously flat forecast."""
    results: dict[str, bool] = {}
    checks = {
        "FRED": lambda: parse_fred_single(fetch_fred_csv("DGS10")),
        "ALFRED": lambda: parse_fred_csv(fetch_alfred_vintage("PAYEMS", "2020-01-15")),
        "BLS": lambda: fetch_bls(["LNS14000000"], date.today().year - 1, date.today().year),
    }
    for name, fn in checks.items():
        try:
            results[name] = bool(fn())
        except Exception as e:
            log_warning(f"[probe] {name} unreachable: {e}")
            results[name] = False
    return results
