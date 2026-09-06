"""
The series catalogue -- the single place that says what data this engine uses.

Every downstream module (ingestion, quality, transforms, factors, models) reads
this file rather than hard-coding series ids, so adding an indicator is a
one-entry change here.

SIGN CONVENTION (spec 10 -- "never mix sign conventions")
---------------------------------------------------------
`direction` answers exactly one question: does a HIGHER raw value mean a
STRONGER economy?
    +1  higher = stronger   (payrolls, permits, new orders)
    -1  higher = weaker     (jobless claims, credit spreads, delinquencies)
The signal engine multiplies every normalised value by `direction`, so from the
factor layer onward +1.0 always means "strongly positive for the economy" and
-1.0 always means "strongly negative", with no per-indicator special cases.

Inflation is the one category where "stronger" is genuinely ambiguous -- 3% CPI
is not obviously good or bad without knowing the regime -- so inflation series
carry direction=0 and are handled by regime classification in modules/factors.py
rather than being folded into a good/bad score. See INFLATION_NOTE below.

`role`      LEADING / COINCIDENT / LAGGING -- feeds the breadth engine (spec 11).
`transform` the default normalisation applied by modules/transforms.py:
    "yoy"    year-over-year % change      (levels that trend, e.g. payrolls)
    "yoy3"   3-month-avg YoY              (noisy levels, e.g. housing starts)
    "level"  the value itself             (already a rate/spread/diffusion index)
    "diff12" 12-month change in the level (rates: the CHANGE is the signal)
    "logyoy" YoY of log level             (indices spanning orders of magnitude)
`expected_lag_days` typical publication lag after the observation period ends.
    Used by the quality engine for staleness and by the vintage controller to
    simulate release timing for series ALFRED has no vintages for.
`vintage`  "alfred"  -> pull true historical vintages from ALFRED (revised hard)
           "final"   -> never revised (market prices, rates, spreads); the
                        real-time value IS the final value, so a vintage pull
                        would be wasted requests returning identical numbers.
           "lag"     -> revised, but ALFRED coverage is thin/absent; the backtest
                        applies expected_lag_days and flags it as lag-adjusted
                        rather than vintage-true.
"""
from __future__ import annotations

from dataclasses import dataclass

# Categories, in the order the dashboard shows them.
CATEGORIES = ["housing", "consumer", "labor", "manufacturing",
              "credit", "curve", "inflation", "policy", "financial"]

INFLATION_NOTE = (
    "Inflation series carry direction=0: a higher CPI print is not inherently "
    "'good' or 'bad' for growth, it depends on the regime (disinflation from 9% "
    "is positive; reflation from 2% to 5% is negative). modules/factors.py maps "
    "inflation to a regime label first and only then to a signed contribution."
)


@dataclass(frozen=True)
class Series:
    sid: str                      # our stable id (usually the FRED id)
    label: str
    category: str
    freq: str                     # daily | weekly | monthly | quarterly
    direction: int                # +1 higher=stronger, -1 higher=weaker, 0 regime-handled
    role: str                     # LEADING | COINCIDENT | LAGGING | CONTEXT
    transform: str
    expected_lag_days: int
    vintage: str                  # alfred | final | lag
    source: str = "FRED"
    fred_ids: tuple[str, ...] = ()   # candidates tried in order (renames/retirements)
    unit: str = ""
    core: bool = False            # part of the reduced feature set the models use
    note: str = ""

    @property
    def candidates(self) -> tuple[str, ...]:
        return self.fred_ids or (self.sid,)


def _s(sid, label, category, freq, direction, role, transform, lag, vintage,
       unit="", core=False, fred_ids=(), source="FRED", note="") -> Series:
    return Series(sid=sid, label=label, category=category, freq=freq,
                  direction=direction, role=role, transform=transform,
                  expected_lag_days=lag, vintage=vintage, unit=unit, core=core,
                  fred_ids=tuple(fred_ids), source=source, note=note)


# ---------------------------------------------------------------------------
# 8.1 HOUSING -- the classic long-lead sector. Housing turns first, both down
# and up, because it is the most rate-sensitive part of the economy.
# ---------------------------------------------------------------------------
HOUSING = [
    _s("PERMIT", "Building Permits", "housing", "monthly", +1, "LEADING", "yoy3", 19, "alfred", "k units", core=True,
       note="The best-documented housing lead; a Conference Board LEI component."),
    _s("HOUST", "Housing Starts", "housing", "monthly", +1, "LEADING", "yoy3", 19, "alfred", "k units", core=True),
    _s("HOUST1F", "Single-Family Starts", "housing", "monthly", +1, "LEADING", "yoy3", 19, "alfred", "k units"),
    _s("HSN1F", "New Home Sales", "housing", "monthly", +1, "LEADING", "yoy3", 24, "alfred", "k units", core=True),
    _s("EXHOSLUSM495S", "Existing Home Sales", "housing", "monthly", +1, "COINCIDENT", "yoy3", 22, "lag", "units",
       note="Spec-named, but FRED's public catalogue retains only ~12 months of it "
            "(NAR licenses the history, the same restriction that removed ISM). "
            "Kept because current momentum is still informative; modules/transforms.py "
            "refuses to z-score a series this short, so it contributes to breadth "
            "and never to a percentile it has no history to support."),
    _s("MSPUS", "Median Home Sale Price", "housing", "quarterly", +1, "LAGGING", "yoy", 30, "lag", "$",
       note="Census median sales price, continuous since 1963 -- the long housing-price "
            "history that Case-Shiller (1987-) and the truncated NAR series cannot give."),
    _s("MSACSR", "Months Supply of New Homes", "housing", "monthly", -1, "LEADING", "level", 24, "alfred", "months", core=True,
       note="Inventory overhang leads price weakness; rising supply = weakening."),
    _s("MORTGAGE30US", "30Y Mortgage Rate", "housing", "weekly", -1, "LEADING", "diff12", 0, "final", "%", core=True,
       note="Freddie Mac PMMS. Never revised, so vintage='final'."),
    _s("CSUSHPINSA", "Case-Shiller Home Prices", "housing", "monthly", +1, "LAGGING", "yoy", 60, "lag", "index",
       fred_ids=("CSUSHPINSA", "CSUSHPISA"),
       note="~2-month publication lag; the slowest-moving housing series here."),
    _s("RHORUSQ156N", "Rental Vacancy Rate", "housing", "quarterly", -1, "LAGGING", "level", 30, "lag", "%"),
]

# ---------------------------------------------------------------------------
# 8.2 CONSUMER -- ~68% of GDP. Watch income vs spending vs the savings buffer.
# ---------------------------------------------------------------------------
CONSUMER = [
    _s("DSPIC96", "Real Disposable Income", "consumer", "monthly", +1, "COINCIDENT", "yoy", 30, "alfred", "$bn", core=True),
    _s("PCEC96", "Real Personal Consumption", "consumer", "monthly", +1, "COINCIDENT", "yoy", 30, "alfred", "$bn", core=True),
    _s("RRSFS", "Real Retail Sales", "consumer", "monthly", +1, "COINCIDENT", "yoy", 16, "alfred", "$mn", core=True,
       fred_ids=("RRSFS", "RSAFS")),
    _s("PSAVERT", "Personal Savings Rate", "consumer", "monthly", +1, "LEADING", "level", 30, "alfred", "%",
       note="Ambiguous alone -- a FALLING rate can mean confidence or distress -- so "
            "modules/factors.py reads it jointly with real income growth."),
    _s("UMCSENT", "Consumer Sentiment (UMich)", "consumer", "monthly", +1, "LEADING", "level", 10, "lag", "index", core=True),
    _s("TOTALSL", "Consumer Credit Outstanding", "consumer", "monthly", +1, "LAGGING", "yoy", 37, "lag", "$bn"),
    _s("ALTSALES", "Light Vehicle Sales", "consumer", "monthly", +1, "LEADING", "yoy3", 3, "lag", "mn saar", core=True,
       note="Big-ticket credit-financed purchase -- turns early in a consumer downturn."),
    _s("TDSP", "Household Debt Service Ratio", "consumer", "quarterly", -1, "LAGGING", "level", 75, "lag", "%"),
    _s("DRCCLACBS", "Credit Card Delinquency Rate", "consumer", "quarterly", -1, "LAGGING", "level", 60, "lag", "%", core=True),
    _s("DRALACBS", "All Loans Delinquency Rate", "consumer", "quarterly", -1, "LAGGING", "level", 60, "lag", "%"),
]

# ---------------------------------------------------------------------------
# 8.3 LABOR -- coincident by construction, but claims and temp help lead.
# ---------------------------------------------------------------------------
LABOR = [
    _s("UNRATE", "Unemployment Rate", "labor", "monthly", -1, "LAGGING", "diff12", 5, "alfred", "%", core=True),
    _s("PAYEMS", "Nonfarm Payrolls", "labor", "monthly", +1, "COINCIDENT", "yoy", 5, "alfred", "k", core=True,
       note="Heavily revised (benchmark + birth-death); a prime case for true ALFRED vintages."),
    _s("ICSA", "Initial Jobless Claims", "labor", "weekly", -1, "LEADING", "yoy", 5, "final", "k", core=True,
       note="Revised only trivially one week later; treated as final for backtests."),
    _s("CCSA", "Continuing Claims", "labor", "weekly", -1, "LEADING", "yoy", 12, "final", "k", core=True),
    _s("TEMPHELPS", "Temporary Help Services", "labor", "monthly", +1, "LEADING", "yoy", 5, "alfred", "k", core=True,
       note="Firms cut temps before permanent staff -- one of the cleanest labour leads."),
    _s("AWHMAN", "Avg Weekly Hours, Mfg", "labor", "monthly", +1, "LEADING", "diff12", 5, "alfred", "hrs", core=True,
       note="Hours are cut before heads; a Conference Board LEI component."),
    _s("JTSJOL", "Job Openings (JOLTS)", "labor", "monthly", +1, "LEADING", "yoy", 40, "lag", "k", core=True),
    _s("JTSHIL", "Hires (JOLTS)", "labor", "monthly", +1, "LEADING", "yoy", 40, "lag", "k"),
    _s("JTSQUL", "Quits (JOLTS)", "labor", "monthly", +1, "LEADING", "yoy", 40, "lag", "k",
       note="A confidence proxy -- workers stop quitting before layoffs start."),
    _s("CES0500000003", "Avg Hourly Earnings", "labor", "monthly", +1, "LAGGING", "yoy", 5, "alfred", "$"),
    _s("U6RATE", "U-6 Underemployment", "labor", "monthly", -1, "LAGGING", "diff12", 5, "lag", "%"),
    _s("CIVPART", "Labor Force Participation", "labor", "monthly", +1, "LAGGING", "diff12", 5, "lag", "%"),
]

# ---------------------------------------------------------------------------
# 8.4 MANUFACTURING / BUSINESS CYCLE
# Note on ISM: the ISM PMI is NOT freely redistributable and was pulled from
# FRED's public catalogue, so this engine uses the regional Federal Reserve
# manufacturing surveys (Empire State, Philadelphia) as free diffusion-index
# stand-ins. They are genuinely leading, published earlier than ISM, and free.
# ---------------------------------------------------------------------------
MANUFACTURING = [
    _s("INDPRO", "Industrial Production", "manufacturing", "monthly", +1, "COINCIDENT", "yoy", 16, "alfred", "index", core=True),
    _s("IPMAN", "Manufacturing Production", "manufacturing", "monthly", +1, "COINCIDENT", "yoy", 16, "alfred", "index"),
    _s("TCU", "Capacity Utilization", "manufacturing", "monthly", +1, "COINCIDENT", "diff12", 16, "alfred", "%", core=True),
    _s("NEWORDER", "Core Capital Goods Orders", "manufacturing", "monthly", +1, "LEADING", "yoy3", 26, "alfred", "$mn", core=True,
       note="Nondefence capital goods ex-aircraft -- the cleanest free capex-intent series."),
    _s("DGORDER", "Durable Goods Orders", "manufacturing", "monthly", +1, "LEADING", "yoy3", 26, "alfred", "$mn"),
    _s("AMTMNO", "Total Factory Orders", "manufacturing", "monthly", +1, "LEADING", "yoy3", 35, "lag", "$mn"),
    _s("ISRATIO", "Inventory / Sales Ratio", "manufacturing", "monthly", -1, "LEADING", "level", 45, "lag", "ratio", core=True,
       note="Rising inventories against flat sales precede production cuts."),
    _s("GACDISA066MSFRBNY", "Empire State Mfg Index", "manufacturing", "monthly", +1, "LEADING", "level", 0, "final", "diffusion", core=True,
       note="Free ISM substitute #1. Diffusion index, 0 = no change; never revised."),
    _s("GACDFSA066MSFRBPHI", "Philadelphia Fed Mfg Index", "manufacturing", "monthly", +1, "LEADING", "level", 0, "final", "diffusion", core=True,
       note="Free ISM substitute #2. Published mid-month, ahead of ISM."),
]

# ---------------------------------------------------------------------------
# 8.5 CREDIT -- where recessions become financial events. Spreads are daily and
# never revised, which makes them the highest-quality inputs in the whole model.
# ---------------------------------------------------------------------------
CREDIT = [
    # THE LONG-HISTORY CREDIT SPREADS. Moody's Baa/Aaa minus the 10Y Treasury:
    # daily since 1986, never revised, and public domain -- so FRED serves the
    # FULL history keylessly. These carry the credit signal for the models,
    # because the ICE BofA OAS series below cannot (see their note).
    _s("BAA10Y", "Baa Corporate Spread over 10Y", "credit", "daily", -1, "LEADING", "level", 1, "final", "%", core=True,
       note="Moody's Baa yield minus the 10Y Treasury -- the classic free credit "
            "spread, continuous since 1986. This is the model's credit feature; the "
            "ICE OAS series are richer but only three years deep on the keyless "
            "endpoint, which is not enough to z-score across cycles."),
    _s("AAA10Y", "Aaa Corporate Spread over 10Y", "credit", "daily", -1, "LEADING", "level", 1, "final", "%", core=True,
       note="The investment-grade counterpart, continuous since 1983."),
    # THE ICE BofA OAS SERIES. Genuinely the most responsive credit-stress reads
    # available, and they stay in the catalogue for exactly that reason -- but
    # FRED's public keyless endpoint serves only a ~3-year rolling window of them
    # (ICE licenses the history, the same restriction that removed ISM and
    # truncates SP500 to ten years). Verified 2026-09-06: 787 observations from
    # 2023-09-05. So they are NOT model features and NOT core: their percentiles
    # mean "versus the last three years", never "versus history", and the
    # dashboard's percentile column would otherwise imply a depth they lack.
    _s("BAMLH0A0HYM2", "High Yield OAS", "credit", "daily", -1, "LEADING", "level", 1, "final", "%",
       note="ICE BofA option-adjusted spread. The sharpest current read on credit "
            "stress, but only ~3 years deep on the keyless endpoint -- so it informs "
            "breadth and the current-conditions panels, never a long-run percentile "
            "or a model trained since 1985. Use BAA10Y for history."),
    _s("BAMLC0A0CM", "Investment Grade OAS", "credit", "daily", -1, "LEADING", "level", 1, "final", "%",
       note="Same ~3-year keyless truncation as the high-yield OAS above."),
    _s("BAMLH0A3HYC", "CCC & Lower OAS", "credit", "daily", -1, "LEADING", "level", 1, "final", "%",
       note="The low-quality tail moves first when credit turns. Same ~3-year "
            "keyless truncation."),
    _s("DRTSCILM", "C&I Lending Standards (SLOOS)", "credit", "quarterly", -1, "LEADING", "level", 30, "lag", "% net tightening", core=True,
       note="Senior Loan Officer Survey. Quarterly and slow, but historically one of "
            "the highest-signal recession precursors."),
    _s("DRTSCLCC", "Credit Card Lending Standards", "credit", "quarterly", -1, "LEADING", "level", 30, "lag", "%"),
    _s("BUSLOANS", "C&I Loans Outstanding", "credit", "monthly", +1, "LAGGING", "yoy", 20, "lag", "$bn", core=True,
       note="H.8 monthly aggregate. Declared weekly in an early draft; the quality "
            "engine's frequency check caught the mismatch (median gap 31d)."),
    _s("DRSFRMACBS", "Mortgage Delinquency Rate", "credit", "quarterly", -1, "LAGGING", "level", 60, "lag", "%"),
    _s("STLFSI4", "St. Louis Financial Stress", "credit", "weekly", -1, "LEADING", "level", 4, "final", "index", core=True,
       fred_ids=("STLFSI4", "STLFSI3", "STLFSI2")),
    _s("TOTBKCR", "Total Bank Credit", "credit", "weekly", +1, "LAGGING", "yoy", 8, "lag", "$bn"),
]

# ---------------------------------------------------------------------------
# 8.6 YIELD CURVE -- spec 8.6 is explicit that "inverted = bad" is too crude.
# We store the raw spreads here; modules/factors.py derives inversion DEPTH,
# DURATION and the direction of travel (bull/bear steepening), because a curve
# re-steepening out of deep inversion is a late-stage recession signal, not an
# all-clear.
# ---------------------------------------------------------------------------
CURVE = [
    _s("T10Y2Y", "10Y - 2Y Spread", "curve", "daily", +1, "LEADING", "level", 1, "final", "%", core=True),
    _s("T10Y3M", "10Y - 3M Spread", "curve", "daily", +1, "LEADING", "level", 1, "final", "%", core=True,
       note="Estrella-Mishkin's preferred spread; the best single-variable recession "
            "predictor in the published literature."),
    _s("DGS10", "10Y Treasury Yield", "curve", "daily", 0, "CONTEXT", "level", 1, "final", "%", core=True),
    _s("DGS2", "2Y Treasury Yield", "curve", "daily", 0, "CONTEXT", "level", 1, "final", "%", core=True),
    _s("DGS30", "30Y Treasury Yield", "curve", "daily", 0, "CONTEXT", "level", 1, "final", "%"),
    _s("DGS3MO", "3M Treasury Yield", "curve", "daily", 0, "CONTEXT", "level", 1, "final", "%", core=True),
]

# ---------------------------------------------------------------------------
# 8.7 INFLATION -- direction=0 throughout; see INFLATION_NOTE.
# ---------------------------------------------------------------------------
INFLATION = [
    _s("CPIAUCSL", "Headline CPI", "inflation", "monthly", 0, "LAGGING", "yoy", 13, "alfred", "%", core=True),
    _s("CPILFESL", "Core CPI", "inflation", "monthly", 0, "LAGGING", "yoy", 13, "alfred", "%", core=True),
    _s("PCEPI", "PCE Price Index", "inflation", "monthly", 0, "LAGGING", "yoy", 30, "alfred", "%"),
    _s("PCEPILFE", "Core PCE Price Index", "inflation", "monthly", 0, "LAGGING", "yoy", 30, "alfred", "%", core=True,
       note="The Fed's target measure -- drives the policy-reaction function."),
    _s("CUSR0000SASLE", "Core Services CPI", "inflation", "monthly", 0, "LAGGING", "yoy", 13, "lag", "%",
       note="The sticky half of inflation; services disinflate slowly."),
    _s("CUSR0000SACL1E", "Core Goods CPI", "inflation", "monthly", 0, "LEADING", "yoy", 13, "lag", "%",
       note="The flexible half; goods prices turn first."),
    _s("T10YIE", "10Y Breakeven Inflation", "inflation", "daily", 0, "LEADING", "level", 1, "final", "%", core=True),
    _s("T5YIFR", "5y5y Forward Inflation", "inflation", "daily", 0, "LEADING", "level", 1, "final", "%", core=True,
       note="Market-implied long-run expectations -- the anchor the Fed watches."),
    _s("MICH", "UMich 1Y Inflation Expectations", "inflation", "monthly", 0, "LEADING", "level", 10, "lag", "%"),
]

# ---------------------------------------------------------------------------
# 8.8 MONETARY POLICY -- restrictiveness is about the REAL rate versus a neutral
# estimate, not the nominal level. 5% nominal at 8% inflation is easy money.
# ---------------------------------------------------------------------------
POLICY = [
    _s("DFF", "Effective Fed Funds Rate", "policy", "daily", 0, "CONTEXT", "level", 1, "final", "%", core=True),
    _s("FEDFUNDS", "Fed Funds Rate (monthly)", "policy", "monthly", 0, "CONTEXT", "level", 5, "final", "%"),
    _s("REAINTRATREARAT10Y", "10Y Real Interest Rate", "policy", "monthly", -1, "LEADING", "level", 30, "lag", "%",
       note="Cleveland Fed model-based real rate; the fallback real-rate estimate for "
            "the pre-2003 era, since TIPS (DFII10) only start in 2003."),
    _s("DFII10", "10Y TIPS Real Yield", "policy", "daily", -1, "LEADING", "level", 1, "final", "%", core=True),
    _s("WALCL", "Fed Balance Sheet", "policy", "weekly", +1, "LEADING", "yoy", 3, "final", "$mn", core=True),
    _s("M2SL", "M2 Money Supply", "policy", "monthly", +1, "LEADING", "yoy", 30, "lag", "$bn", core=True),
    _s("BAA", "Moody's Baa Corporate Yield", "policy", "daily", -1, "LEADING", "diff12", 1, "final", "%",
       fred_ids=("DBAA",),
       note="DBAA only -- the DAILY Baa yield. An early draft listed the monthly BAA "
            "as a fallback, which is a frequency trap: the fallback would store "
            "monthly data under a daily declaration, and every downstream lag "
            "(YoY, momentum, staleness) would then be wrong by a factor of 21 "
            "without anything erroring. One id, one frequency."),
]

# ---------------------------------------------------------------------------
# 8.9 FINANCIAL CONDITIONS
# ---------------------------------------------------------------------------
FINANCIAL = [
    _s("NFCI", "Chicago Fed NFCI", "financial", "weekly", -1, "LEADING", "level", 4, "lag", "index", core=True,
       note="Positive = tighter than average. Revised weekly as inputs update, hence 'lag'."),
    _s("ANFCI", "Adjusted NFCI", "financial", "weekly", -1, "LEADING", "level", 4, "lag", "index", core=True,
       note="NFCI purged of the growth/inflation cycle -- the part of tightening not "
            "explained by where the economy already is."),
    _s("VIXCLS", "VIX", "financial", "daily", -1, "LEADING", "level", 1, "final", "index", core=True),
    _s("SP500", "S&P 500", "financial", "daily", +1, "LEADING", "logyoy", 1, "final", "index",
       note="Equities are a genuine LEI component, but S&P licenses its index history "
            "and FRED's keyless endpoint holds only ten years -- enough for the "
            "current-conditions display, not for a cross-cycle percentile. NASDAQCOM "
            "(continuous since 1971) carries the equity signal instead."),
    _s("NASDAQCOM", "Nasdaq Composite (deep history)", "financial", "daily", +1, "LEADING", "logyoy", 1, "final", "index", core=True,
       note="The deep-history equity series. FRED retired every Wilshire 5000 id "
            "(WILL5000IND/INDFC/PR/PRFC all 404 as of 2026-09) and keeps only 10 "
            "years of SP500, so Nasdaq Composite -- continuous since 1971-02-05 -- "
            "is the only free index long enough to z-score across eight cycles. "
            "More tech-weighted than the S&P; used for the equity SIGNAL (12m "
            "log change), never as a level."),
    _s("DTWEXBGS", "Broad Dollar Index", "financial", "daily", 0, "CONTEXT", "logyoy", 7, "final", "index", core=True,
       note="Daily observations, but the H.10 release publishes them weekly in arrears, "
            "so the freshness budget is 7 days rather than the 1 a daily series implies."),
]

# ---------------------------------------------------------------------------
# GROUND TRUTH + market-layer context (not scored as macro factors)
# ---------------------------------------------------------------------------
TRUTH = [
    _s("USREC", "NBER Recession Indicator", "truth", "monthly", 0, "CONTEXT", "level", 0, "final", "0/1",
       note="NBER dates are announced with a long lag (up to a year). The backtest "
            "uses them ONLY as the label to score against, never as a model input."),
]

MARKET = [
    _s("GOLD", "Gold (LBMA / spot)", "market", "daily", 0, "CONTEXT", "logyoy", 1, "final", "$",
       fred_ids=("GOLDPMGBD228NLBM", "GOLDAMGBD228NLBM"),
       note="FRED retired the LBMA fixings; ingestion falls back to Yahoo's keyless "
            "chart endpoint (GC=F), the same fallback proven in the sibling "
            "Personal Dynamic Dashboard's fetch_macro.py."),
    _s("DCOILWTICO", "WTI Crude Oil", "market", "daily", 0, "CONTEXT", "logyoy", 4, "final", "$",
       note="EIA posts to FRED with a few days' lag; 1 day was too tight and flagged "
            "a healthy series as stale."),
    _s("PPIACO", "Producer Price Index", "market", "monthly", 0, "CONTEXT", "yoy", 13, "lag", "index"),
]


ALL_SERIES: list[Series] = (HOUSING + CONSUMER + LABOR + MANUFACTURING + CREDIT
                            + CURVE + INFLATION + POLICY + FINANCIAL + TRUTH + MARKET)

BY_ID: dict[str, Series] = {s.sid: s for s in ALL_SERIES}

# Sanity: ids must be unique, or ingestion would silently overwrite one with
# another and the factor engine would score a series it never fetched.
assert len(BY_ID) == len(ALL_SERIES), "duplicate series id in catalogue"


def by_category(category: str) -> list[Series]:
    return [s for s in ALL_SERIES if s.category == category]


def factor_series() -> list[Series]:
    """Series that participate in factor scoring (everything except the NBER
    label and the market-layer context, which are handled separately)."""
    return [s for s in ALL_SERIES if s.category in CATEGORIES]


def leading_series() -> list[Series]:
    """The breadth panel (spec 11). Only LEADING series with a real sign --
    direction=0 series have no 'improving/weakening' meaning to count."""
    return [s for s in factor_series() if s.role == "LEADING" and s.direction != 0]


def core_series() -> list[Series]:
    """Reduced feature set the statistical models train on. Kept small on
    purpose: ~8 recessions since 1970 means a 60-feature logit would fit noise."""
    return [s for s in factor_series() if s.core]


def alfred_vintage_series() -> list[Series]:
    """Series worth spending ALFRED requests on -- heavily revised AND used by
    the models. Everything else is either never revised (vintage='final', where
    real-time == final) or too peripheral to justify thousands of calls."""
    return [s for s in ALL_SERIES if s.vintage == "alfred"]


FREQ_ORDER = {"daily": 0, "weekly": 1, "monthly": 2, "quarterly": 3}
