"""Point-in-time quarterly fundamentals from SEC EDGAR.

EDGAR is the upstream source of truth for US fundamentals: every public company
files 10-K/10-Q in XBRL, and the ``companyfacts`` API exposes the full history
with the actual ``filed`` date of each figure. That filing date is what makes
the data true point-in-time (no lookahead).

For each ticker we:
  1. map ticker -> CIK,
  2. pull companyfacts JSON,
  3. extract the GAAP concepts we need (with fallbacks, since tags drift),
  4. build a clean quarterly series (deriving Q4 = annual - 9mo where needed),
  5. compute TTM flows, point-in-time stocks, and price-independent ratios,
  6. stamp each quarter with ``availability_date`` = first filing date,
  7. attach sector from the company's SIC code.

Valuation ratios needing market price (earnings yield, book-to-price, EV/EBITDA)
are built later in the panel stage from these building blocks + monthly price.
"""
from __future__ import annotations

import time

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
}


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


def _concept_records(facts: dict, concept: str) -> list[dict] | None:
    """Return the USD/shares unit records for a us-gaap or dei concept."""
    for ns in ("us-gaap", "dei"):
        node = facts.get(ns, {}).get(concept)
        if node:
            units = node.get("units", {})
            for unit_key in ("USD", "shares", "USD/shares"):
                if unit_key in units:
                    return units[unit_key]
    return None


def _first_match(facts: dict, candidates: list[str]) -> list[dict] | None:
    for c in candidates:
        recs = _concept_records(facts, c)
        if recs:
            return recs
    return None


def _quarterly_flow(records: list[dict]) -> pd.DataFrame:
    """Clean quarterly series for a flow concept (as-first-reported).

    Handles two filing styles uniformly:
      - income-statement items tagged as discrete 3-month periods, and
      - cash-flow items tagged only as year-to-date cumulatives (3/6/9/12-mo).

    Records sharing a fiscal-period ``start`` are cumulative; differencing them
    by ascending ``end`` yields the discrete quarterly increment (so Q4 falls
    out of annual − 9-month automatically). Discrete 3-month records, when
    present, take precedence. Returns DataFrame indexed by period_end with
    columns ``val``, ``filed``.
    """
    three_m: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    by_start: dict[pd.Timestamp, list[tuple[pd.Timestamp, float, pd.Timestamp]]] = {}
    for r in records:
        if "start" not in r or "end" not in r or r.get("val") is None:
            continue
        start, end = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        filed = pd.Timestamp(r["filed"])
        days = (end - start).days
        if 80 <= days <= 100:                       # discrete quarter
            prev = three_m.get(end)
            if prev is None or filed < prev[1]:     # earliest filing wins (PIT)
                three_m[end] = (float(r["val"]), filed)
        elif 100 < days <= 380:                     # YTD cumulative (incl. annual)
            by_start.setdefault(start, []).append((end, float(r["val"]), filed))

    # Difference each cumulative (same-start) group to get quarterly increments.
    derived: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    for _start, items in by_start.items():
        items.sort(key=lambda x: x[0])
        prev_val = 0.0
        for end, val, filed in items:
            inc = val - prev_val
            prev_val = val
            if end not in derived or filed < derived[end][1]:
                derived[end] = (inc, filed)

    # Discrete 3-month observations override derived ones.
    quarter = {**derived, **three_m}
    if not quarter:
        return pd.DataFrame(columns=["val", "filed"])
    return (
        pd.DataFrame(
            [(e, v, f) for e, (v, f) in quarter.items()],
            columns=["period_end", "val", "filed"],
        )
        .set_index("period_end")
        .sort_index()
    )


def _quarterly_stock(records: list[dict]) -> pd.DataFrame:
    """Point-in-time series for a balance-sheet (instant) concept."""
    by_end: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    for r in records:
        if r.get("val") is None or "end" not in r:
            continue
        end = pd.Timestamp(r["end"])
        filed = pd.Timestamp(r["filed"])
        prev = by_end.get(end)
        if prev is None or filed < prev[1]:
            by_end[end] = (float(r["val"]), filed)
    if not by_end:
        return pd.DataFrame(columns=["val", "filed"])
    return (
        pd.DataFrame(
            [(e, v, f) for e, (v, f) in by_end.items()],
            columns=["period_end", "val", "filed"],
        )
        .set_index("period_end")
        .sort_index()
    )


# Coarse SIC -> sector mapping (approximate GICS-like buckets for neutralization).
def _sic_to_sector(sic: int | None) -> str | None:
    if sic is None:
        return None
    s = int(sic)
    if 100 <= s <= 999:
        return "Materials"
    if 1000 <= s <= 1499:
        return "Energy" if s >= 1300 else "Materials"
    if 1500 <= s <= 1799:
        return "Industrials"
    if 2000 <= s <= 2199 or 2080 <= s <= 2090:
        return "Consumer Staples"
    if 2830 <= s <= 2836 or 8000 <= s <= 8099 or s == 2835:
        return "Health Care"
    if 2800 <= s <= 2899:
        return "Materials"
    if 2900 <= s <= 2999 or 1300 <= s <= 1399:
        return "Energy"
    if 3570 <= s <= 3579 or 3670 <= s <= 3679 or 7370 <= s <= 7379:
        return "Information Technology"
    if 3000 <= s <= 3999:
        return "Industrials"
    if 4000 <= s <= 4799:
        return "Industrials"
    if 4800 <= s <= 4899:
        return "Communication Services"
    if 4900 <= s <= 4999:
        return "Utilities"
    if 5200 <= s <= 5999:
        return "Consumer Discretionary"
    if 5000 <= s <= 5199:
        return "Consumer Discretionary"
    if 6000 <= s <= 6499:
        return "Financials"
    if 6500 <= s <= 6799:
        return "Real Estate"
    if 7000 <= s <= 8999:
        return "Communication Services" if 7800 <= s <= 7899 else "Consumer Discretionary"
    return None


def _extract_ticker(ticker: str, cik: str, session: requests.Session) -> pd.DataFrame:
    cfg = load_config("data")["fundamentals"]
    facts_url = cfg["companyfacts_url"].format(cik=cik)
    resp = session.get(facts_url, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame()
    facts = resp.json().get("facts", {})
    if not facts:
        return pd.DataFrame()

    flows, stocks = {}, {}
    for field, cands in _FLOW_CONCEPTS.items():
        recs = _first_match(facts, cands)
        if recs is not None:
            flows[field] = _quarterly_flow(recs)
    for field, cands in _STOCK_CONCEPTS.items():
        recs = _first_match(facts, cands)
        if recs is not None:
            stocks[field] = _quarterly_stock(recs)

    if "revenue" not in flows or "equity" not in stocks:
        return pd.DataFrame()

    # Align everything on the union of quarter-ends.
    ends = sorted(
        set().union(*[d.index for d in flows.values()], *[d.index for d in stocks.values()])
    )
    out = pd.DataFrame(index=pd.DatetimeIndex(ends, name="period_end"))
    filed_dates = pd.Series(pd.NaT, index=out.index)

    for field, d in flows.items():
        out[field] = d["val"].reindex(out.index)
        filed_dates = filed_dates.fillna(d["filed"].reindex(out.index))
    for field, d in stocks.items():
        out[field] = d["val"].reindex(out.index)
        filed_dates = filed_dates.fillna(d["filed"].reindex(out.index))

    # TTM for flows (trailing 4 quarters).
    for field in flows:
        out[f"{field}_ttm"] = out[field].rolling(4, min_periods=4).sum()

    # Total debt = long-term + current portion (either may be absent).
    debt = out.get("debt_lt", pd.Series(index=out.index, dtype=float)).fillna(0) + out.get(
        "debt_cur", pd.Series(index=out.index, dtype=float)
    ).fillna(0)

    result = pd.DataFrame(index=out.index)
    result["revenue_ttm"] = out.get("revenue_ttm")
    result["net_income_ttm"] = out.get("net_income_ttm")
    result["gross_profit_ttm"] = out.get("gross_profit_ttm")
    result["op_cashflow_ttm"] = out.get("op_cashflow_ttm")
    # EBITDA ≈ operating income + D&A (EDGAR does not tag EBITDA directly).
    result["ebitda_ttm"] = out.get("operating_income_ttm", pd.NA)
    if "dna_ttm" in out:
        result["ebitda_ttm"] = result["ebitda_ttm"].add(out["dna_ttm"], fill_value=0)
    result["equity"] = out.get("equity")
    result["assets"] = out.get("assets")
    result["debt"] = debt.replace(0, pd.NA)
    result["cash"] = out.get("cash")

    # Price-independent ratios.
    result["roe"] = result["net_income_ttm"] / result["equity"].replace(0, pd.NA)
    result["gross_margin"] = result["gross_profit_ttm"] / result["revenue_ttm"].replace(0, pd.NA)
    result["profit_margin"] = result["net_income_ttm"] / result["revenue_ttm"].replace(0, pd.NA)
    result["accruals"] = (result["net_income_ttm"] - result["op_cashflow_ttm"]) / result[
        "assets"
    ].replace(0, pd.NA)
    result["revenue_growth_yoy"] = result["revenue_ttm"].pct_change(4)
    result["asset_growth_yoy"] = result["assets"].pct_change(4)

    result["availability_date"] = filed_dates.values
    result["ticker"] = ticker

    # Sector from SIC (one extra lightweight call).
    try:
        sub = session.get(cfg["submissions_url"].format(cik=cik), timeout=30).json()
        result["gics_sector"] = _sic_to_sector(sub.get("sic"))
        result["industry"] = sub.get("sicDescription")
    except Exception:  # noqa: BLE001
        result["gics_sector"] = None
        result["industry"] = None

    return result.reset_index()


def fetch_fundamentals(tickers: list[str], *, verbose: bool = True) -> pd.DataFrame:
    """Fetch EDGAR quarterly fundamentals for many tickers (resilient)."""
    cfg = load_config("data")["fundamentals"]
    sleep = cfg.get("request_sleep", 0.12)
    session = _session()
    cik_map = get_cik_map(session)

    frames, no_cik, failed = [], [], []
    for i, t in enumerate(tickers):
        if verbose and i % 100 == 0:
            print(f"  EDGAR {i}/{len(tickers)}...", flush=True)
        cik = cik_map.get(t)
        if not cik:
            no_cik.append(t)
            continue
        try:
            df = _extract_ticker(t, cik, session)
            if not df.empty:
                frames.append(df)
            else:
                failed.append(t)
            time.sleep(sleep)  # be polite to SEC
        except Exception:  # noqa: BLE001
            failed.append(t)

    if verbose:
        print(f"  done: {len(frames)} ok, {len(no_cik)} no-CIK, {len(failed)} failed")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
