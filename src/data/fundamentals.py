"""Point-in-time fundamentals from SEC EDGAR — raw facts with restatement history.

EDGAR is the upstream source of truth: every public company files 10-K/10-Q in
XBRL, and the ``companyfacts`` API exposes every figure with the actual ``filed``
date. We store **raw facts with full restatement history** — one row per
value-change-point ``(ticker, concept, period_end, filed_date)`` — so any
point-in-time query is correct even when a company restates a prior period.

Design split (see plans/architecture.md):
- **This module** emits faithful raw facts: discrete quarterly values (flows) and
  balance-sheet instants (stocks), each with the filing date at which that value
  became public. A restatement is a new row at a later ``filed_date``.
- **The feature layer** does as-of selection + derivation (TTM, ROE, accruals,
  growth) — because a restatement of one quarter ripples into later TTM/growth,
  derivation must happen against the as-of snapshot, not be pre-baked here.

Output columns match the ``fundamental_facts`` table:
``ticker, cik, concept, gaap_tag, period_end, fiscal_period, duration_days,
filed_date, form, value, unit``.
"""
from __future__ import annotations

import time
from itertools import groupby

import pandas as pd
import requests

from src.config import load_config

# GAAP concept candidates (first match wins; tags vary across filers/years).
_FLOW_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "dna": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ],
    "op_cashflow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}
_STOCK_CONCEPTS: dict[str, list[str]] = {
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "assets": ["Assets"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "debt_cur": ["LongTermDebtCurrent", "DebtCurrent"],
    "shares": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
}
_ALL_CONCEPTS = {**{k: v for k, v in _FLOW_CONCEPTS.items()},
                 **{k: v for k, v in _STOCK_CONCEPTS.items()}}
_IS_FLOW = {k: True for k in _FLOW_CONCEPTS} | {k: False for k in _STOCK_CONCEPTS}

_FACT_COLUMNS = [
    "ticker", "cik", "concept", "gaap_tag", "period_end", "fiscal_period",
    "duration_days", "filed_date", "form", "value", "unit",
]


# --------------------------------------------------------------------------- #
# SEC session / CIK map / submissions
# --------------------------------------------------------------------------- #
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": load_config("data")["fundamentals"]["user_agent"]})
    return s


_CIK_MAP: dict[str, str] | None = None


def get_cik_map(session: requests.Session | None = None) -> dict[str, str]:
    """ticker -> zero-padded 10-digit CIK, from SEC's company_tickers.json."""
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP
    cfg = load_config("data")["fundamentals"]
    s = session or _session()
    data = s.get(cfg["cik_map_url"], timeout=30).json()
    _CIK_MAP = {
        row["ticker"].upper().replace(".", "-"): f"{int(row['cik_str']):010d}"
        for row in data.values()
    }
    return _CIK_MAP


def _fetch_submissions(cik: str, session: requests.Session) -> dict:
    """SEC submissions JSON (light): has SIC and the recent-filings list."""
    cfg = load_config("data")["fundamentals"]
    try:
        return session.get(cfg["submissions_url"].format(cik=cik), timeout=30).json()
    except Exception:  # noqa: BLE001
        return {}


def latest_filing_date(sub_json: dict) -> pd.Timestamp | None:
    """Most recent filing date from a submissions JSON, if any."""
    dates = sub_json.get("filings", {}).get("recent", {}).get("filingDate", [])
    return max((pd.Timestamp(d) for d in dates), default=None) if dates else None


# --------------------------------------------------------------------------- #
# Concept extraction with restatement history
# --------------------------------------------------------------------------- #
def _match_concept(facts: dict, candidates: list[str]) -> tuple[list[dict], str, str] | None:
    """Return (records, unit, gaap_tag) for the first candidate present."""
    for ns in ("us-gaap", "dei"):
        ns_facts = facts.get(ns, {})
        for tag in candidates:
            node = ns_facts.get(tag)
            if not node:
                continue
            units = node.get("units", {})
            for unit_key in ("USD", "shares", "USD/shares"):
                if unit_key in units:
                    return units[unit_key], unit_key, tag
    return None


def _snap_q(ts: pd.Timestamp) -> pd.Timestamp:
    """Snap a date to the NEAREST calendar quarter-end (aligns concept period-ends).

    Snapping to the *containing* quarter's end would shove a fiscal period that
    ends early in a calendar quarter — off-calendar filers ending ~Jan/Apr/Jul/Oct
    (HD, CSCO, GAP, …) — up to ~2 months into the future, fabricating a period_end
    later than the filing date and misaligning the fact into the wrong calendar
    quarter. Nearest-quarter snapping shifts each off-calendar quarter to its
    closest calendar quarter-end: monotonic and collision-free for every
    fiscal-month pattern (consecutive fiscal quarters are ~91 days apart, as are
    calendar quarter-ends, so they map one-to-one).
    """
    q = ts.to_period("Q")
    this_end = q.to_timestamp("Q").normalize()
    prev_end = (q - 1).to_timestamp("Q").normalize()
    return prev_end if (ts - prev_end) < (this_end - ts) else this_end


def _quarter_label(period_end: pd.Timestamp) -> str | None:
    return {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}.get(period_end.month)


def _flow_changepoints(records: list[dict]) -> list[tuple[pd.Timestamp, pd.Timestamp, float, str]]:
    """Discrete-quarter value-change-points for a flow concept, across filings.

    Event-sourced over ``filed`` date: as each filing arrives we update the known
    3-month and cumulative-YTD values, recompute discrete quarters (YTD
    differencing yields Q4 = annual - 9mo automatically), and emit a point
    whenever a quarter's value changes. Returns (period_end, filed, value, form).
    """
    parsed = []
    for r in records:
        if r.get("val") is None or "start" not in r or "end" not in r:
            continue
        filed = pd.Timestamp(r["filed"])
        start, end = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        if end > filed:  # can't report actuals for a period ending after filing —
            continue     # forward guidance / forecast facts; drop (no lookahead)
        parsed.append((filed, start, end, float(r["val"]), r.get("form", "")))
    parsed.sort(key=lambda x: (x[0], x[2]))

    # State, event-sourced over filings:
    #  cumulative[fy_start][quarter_end] = value cumulative from fy_start
    #    (includes the Q1 3-month, which IS the cumulative-to-Q1, so YTD
    #     differencing yields each discrete quarter incl. Q4 = annual - 9mo)
    #  three_mo[quarter_end] = directly-reported 3-month value (overrides diff)
    cumulative: dict[pd.Timestamp, dict[pd.Timestamp, float]] = {}
    three_mo: dict[pd.Timestamp, float] = {}
    emitted: dict[pd.Timestamp, float] = {}
    out: list[tuple[pd.Timestamp, pd.Timestamp, float, str]] = []

    for filed, group in groupby(parsed, key=lambda x: x[0]):
        form = ""
        for _f, start, end, val, frm in group:
            form = frm or form
            qend = _snap_q(end)
            days = (end - start).days
            fy_start = pd.Timestamp(start).to_period("Q").to_timestamp(how="start")
            if 0 <= days <= 380:
                cumulative.setdefault(fy_start, {})[qend] = val
            if 80 <= days <= 100:
                three_mo[qend] = val

        # Recompute discrete quarters: difference each cumulative chain, then
        # let directly-reported 3-month values override (more reliable).
        quarters: dict[pd.Timestamp, float] = {}
        for _fy, ends in cumulative.items():
            seq = sorted(ends)
            for i, qend in enumerate(seq):
                quarters[qend] = ends[qend] - (ends[seq[i - 1]] if i > 0 else 0.0)
        quarters.update(three_mo)

        for qend, val in quarters.items():
            if emitted.get(qend) != val:
                out.append((qend, filed, val, form))
                emitted[qend] = val
    return out


def _instant_changepoints(records: list[dict]) -> list[tuple[pd.Timestamp, pd.Timestamp, float, str]]:
    """Value-change-points for a balance-sheet (instant) concept, across filings."""
    parsed = []
    for r in records:
        if r.get("val") is None or "end" not in r:
            continue
        filed, end = pd.Timestamp(r["filed"]), pd.Timestamp(r["end"])
        if end > filed:  # future-dated instant (e.g. debt-maturity schedule) — drop
            continue
        parsed.append((filed, end, float(r["val"]), r.get("form", "")))
    parsed.sort(key=lambda x: (x[0], x[1]))

    latest: dict[pd.Timestamp, float] = {}
    emitted: dict[pd.Timestamp, float] = {}
    out: list[tuple[pd.Timestamp, pd.Timestamp, float, str]] = []

    for filed, group in groupby(parsed, key=lambda x: x[0]):
        form = ""
        for _f, end, val, frm in group:
            form = frm or form
            latest[_snap_q(end)] = val
        for qend, val in latest.items():
            if emitted.get(qend) != val:
                out.append((qend, filed, val, form))
                emitted[qend] = val
    return out


def _extract_facts(ticker: str, cik: str, facts: dict) -> pd.DataFrame:
    """Long raw-facts table for one ticker: every value-change-point per concept."""
    rows: list[dict] = []
    for concept, candidates in _ALL_CONCEPTS.items():
        matched = _match_concept(facts, candidates)
        if matched is None:
            continue
        records, unit, gaap_tag = matched
        is_flow = _IS_FLOW[concept]
        points = (_flow_changepoints if is_flow else _instant_changepoints)(records)
        for period_end, filed, value, form in points:
            rows.append({
                "ticker": ticker,
                "cik": cik,
                "concept": concept,
                "gaap_tag": gaap_tag,
                "period_end": period_end,
                "fiscal_period": _quarter_label(period_end),
                "duration_days": 91 if is_flow else 0,
                "filed_date": filed,
                "form": form,
                "value": value,
                "unit": unit,
            })
    return pd.DataFrame(rows, columns=_FACT_COLUMNS)


def fetch_company_facts(cik: str, session: requests.Session) -> dict:
    cfg = load_config("data")["fundamentals"]
    resp = session.get(cfg["companyfacts_url"].format(cik=cik), timeout=30)
    if resp.status_code != 200:
        return {}
    return resp.json().get("facts", {})


def fetch_fundamentals(
    tickers: list[str],
    *,
    existing_latest: dict[str, pd.Timestamp] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch EDGAR raw facts (restatement history) for many tickers.

    Returns a long DataFrame matching ``fundamental_facts``. When
    ``existing_latest`` (ticker -> max filed_date already stored) is given, the
    light submissions endpoint is checked first and the heavy companyfacts pull
    is skipped for tickers with no new filing — the incremental fast path.

    Sector (from SIC) is returned separately via :func:`fetch_sectors` since it
    belongs to the ``universe`` table, not the facts.
    """
    cfg = load_config("data")["fundamentals"]
    sleep = cfg.get("request_sleep", 0.12)
    session = _session()
    cik_map = get_cik_map(session)

    frames, no_cik, failed, skipped = [], [], [], 0
    for i, t in enumerate(tickers):
        if verbose and i % 100 == 0:
            print(f"  EDGAR {i}/{len(tickers)}...", flush=True)
        cik = cik_map.get(t)
        if not cik:
            no_cik.append(t)
            continue
        try:
            if existing_latest is not None and t in existing_latest:
                sub = _fetch_submissions(cik, session)
                time.sleep(sleep)
                latest = latest_filing_date(sub)
                if latest is not None and latest <= existing_latest[t]:
                    skipped += 1
                    continue
            facts = fetch_company_facts(cik, session)
            time.sleep(sleep)
            if not facts:
                failed.append(t)
                continue
            df = _extract_facts(t, cik, facts)
            if df.empty:
                failed.append(t)
            else:
                frames.append(df)
        except Exception:  # noqa: BLE001
            failed.append(t)

    if verbose:
        msg = f"  done: {len(frames)} fetched, {len(no_cik)} no-CIK, {len(failed)} failed"
        if existing_latest is not None:
            msg += f", {skipped} unchanged-skipped"
        print(msg)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_FACT_COLUMNS)
    # Library-boundary DQ gate: shape/dtype/PIT contract before the facts are
    # ever cached or written to the warehouse. Lazy import breaks the cycle
    # (schemas imports the concept set from this module).
    from src.data.schemas import validate_fundamental_facts
    return validate_fundamental_facts(out)


# --------------------------------------------------------------------------- #
# Sector (SIC) — belongs to the universe table, fetched alongside facts.
# --------------------------------------------------------------------------- #
# Ordered SIC -> GICS-like sector ranges (first match wins). Specific 4-digit
# refinements first, then 2-digit major-group fallbacks covering 0100-9999 so
# any valid SIC maps to a sector (approximate: SIC != GICS, but ~complete).
_SIC_RANGES: list[tuple[int, int, str]] = [
    # --- specific refinements ---
    (1311, 1389, "Energy"),            # oil & gas extraction/services
    (1200, 1299, "Energy"),            # coal
    (2911, 2999, "Energy"),            # petroleum refining
    (4610, 4619, "Energy"),            # pipelines
    (2833, 2836, "Health Care"),       # pharma / biological products
    (3840, 3851, "Health Care"),       # medical/surgical/ophthalmic instruments
    (8000, 8099, "Health Care"),       # health services
    (8731, 8731, "Health Care"),       # commercial biological research (biotech)
    (3570, 3579, "Information Technology"),  # computer & office equipment
    (3670, 3679, "Information Technology"),  # semiconductors & electronic components
    (3661, 3669, "Information Technology"),  # communications equipment
    (7370, 7379, "Information Technology"),  # software & data services
    (3820, 3829, "Information Technology"),  # measuring/controlling instruments
    (7310, 7319, "Communication Services"),  # advertising
    (4800, 4899, "Communication Services"),  # communications (telecom/broadcast/cable)
    (2700, 2799, "Communication Services"),  # publishing/printing
    (7800, 7841, "Communication Services"),  # motion pictures
    (3711, 3716, "Consumer Discretionary"),  # motor vehicles
    (3630, 3639, "Consumer Discretionary"),  # household appliances
    (5400, 5499, "Consumer Staples"),        # food stores
    (5912, 5912, "Consumer Staples"),        # drug stores
    (2000, 2199, "Consumer Staples"),        # food & tobacco
    (2080, 2085, "Consumer Staples"),        # beverages
    (100,  299,  "Consumer Staples"),        # agricultural production
    (6798, 6798, "Real Estate"),             # REITs
    (6500, 6599, "Real Estate"),             # real estate
    # --- 2-digit major-group fallbacks (cover everything) ---
    (300,  999,  "Materials"),         # ag services / forestry / fishing
    (1000, 1099, "Materials"),         # metal mining
    (1400, 1499, "Materials"),         # nonmetallic mining
    (1500, 1799, "Industrials"),       # construction
    (2200, 2399, "Consumer Discretionary"),  # textiles & apparel
    (2400, 2499, "Materials"),         # lumber & wood
    (2500, 2599, "Consumer Discretionary"),  # furniture
    (2600, 2699, "Materials"),         # paper
    (2800, 2899, "Materials"),         # chemicals (non-pharma)
    (3000, 3399, "Materials"),         # rubber/plastics/leather/stone/metals
    (3400, 3499, "Industrials"),       # fabricated metal
    (3500, 3599, "Industrials"),       # industrial machinery (non-computer)
    (3600, 3699, "Industrials"),       # electrical equipment (non-IT)
    (3700, 3799, "Industrials"),       # transportation equipment (aerospace etc.)
    (3800, 3899, "Industrials"),       # instruments (non-medical/non-IT)
    (3900, 3999, "Consumer Discretionary"),  # misc manufacturing
    (4000, 4799, "Industrials"),       # transportation
    (4900, 4999, "Utilities"),         # electric/gas/sanitary
    (5000, 5199, "Industrials"),       # wholesale trade
    (5200, 5999, "Consumer Discretionary"),  # retail trade
    (6000, 6499, "Financials"),        # finance & insurance
    (6600, 6799, "Financials"),        # holding/investment offices
    (7000, 7099, "Consumer Discretionary"),  # hotels & lodging
    (7100, 7399, "Industrials"),       # business services (non-IT/non-advertising)
    (7400, 7799, "Industrials"),       # other business/repair services
    (7900, 7999, "Consumer Discretionary"),  # amusement & recreation
    (8100, 8199, "Industrials"),       # legal services
    (8200, 8399, "Consumer Discretionary"),  # education & social services
    (8400, 8999, "Industrials"),       # other services
    (9000, 9999, "Industrials"),       # public administration (rare)
]


def _sic_to_sector(sic: int | str | None) -> str | None:
    if sic is None or sic == "":
        return None
    try:
        s = int(sic)
    except (ValueError, TypeError):
        return None
    for lo, hi, sector in _SIC_RANGES:
        if lo <= s <= hi:
            return sector
    return None


# yfinance's sector vocabulary -> our GICS labels. Fallback for names whose SEC
# submissions endpoint has a blank `sic` (regional banks, BDCs/closed-end funds,
# some foreign issuers all return sic='' from EDGAR even with a valid CIK).
_YF_SECTOR_MAP = {
    "Financial Services": "Financials",
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
    "Utilities": "Utilities",
}


def _yf_sector(ticker: str) -> str | None:
    """Best-effort GICS sector from yfinance .info (fallback when SEC SIC blank)."""
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    return _YF_SECTOR_MAP.get(info.get("sector"))


def fetch_sectors(
    tickers: list[str], *, verbose: bool = False, yf_fallback: bool = True
) -> pd.DataFrame:
    """Map tickers -> (sic, gics_sector, industry, sector_source).

    Primary source is the SEC submissions SIC code. When that is blank/unmapped
    and ``yf_fallback`` is set, fall back to yfinance's sector (mapped to our
    GICS labels) so banks, BDCs/closed-end funds, and foreign issuers that EDGAR
    leaves unclassified still get a sector instead of forming a NULL group that
    sector-neutral normalization would drop.
    """
    session = _session()
    cik_map = get_cik_map(session)
    sleep = load_config("data")["fundamentals"].get("request_sleep", 0.12)
    rows = []
    for i, t in enumerate(tickers):
        if verbose and i % 200 == 0:
            print(f"  sectors {i}/{len(tickers)}...", flush=True)
        cik = cik_map.get(t)
        sub = _fetch_submissions(cik, session) if cik else None
        if cik:
            time.sleep(sleep)
        sic = sub.get("sic") if sub else None
        gics = _sic_to_sector(sic)
        source = "sec_sic" if gics else None
        if gics is None and yf_fallback:
            gics = _yf_sector(t)
            source = "yfinance" if gics else None
        rows.append({
            "ticker": t,
            "sic": str(sic) if sic not in (None, "") else None,
            "gics_sector": gics,
            "industry": sub.get("sicDescription") if sub else None,
            "sector_source": source,
        })
    return pd.DataFrame(rows)
