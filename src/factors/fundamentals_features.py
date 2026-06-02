"""Derive the quarterly fundamental *feature* table from raw EDGAR facts.

The warehouse stores ``fundamental_facts`` as raw, restatement-versioned XBRL
change-points (one row per ``ticker, concept, period_end, filed_date``). This
module rolls them up into the per-quarter feature row the panel consumes —
trailing-twelve-month flows, balance-sheet levels, and the price-independent
ratios (ROE, margins, accruals, growth) — each stamped with an
``availability_date`` so the panel's point-in-time as-of join stays correct.

This is the medallion *gold/feature* layer over the silver raw facts. It is a
pure function (no DB / network), materialized by the ``fundamental_features``
Dagster asset and unit-tested directly.

History: this logic previously lived in ``src/data/fundamentals.py``
(``_extract_ticker``, removed in commit ``fc04b5c``) and parsed raw SEC JSON.
Because the ingester now stores *discrete* quarterly change-points (YTD
cumulatives already differenced by ``_flow_changepoints``), TTM here is a plain
rolling 4-quarter sum — no re-differencing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.fundamentals import _FLOW_CONCEPTS, _STOCK_CONCEPTS

_FLOWS = list(_FLOW_CONCEPTS)   # revenue, net_income, gross_profit, operating_income, dna, op_cashflow
_STOCKS = list(_STOCK_CONCEPTS)  # equity, assets, cash, debt_lt, debt_cur, shares

# Output column contract (matches what panel.pit_join_fundamentals selects, plus
# the TTM aggregates used by compute_valuation_ratios).
OUTPUT_COLUMNS = [
    "ticker", "period_end", "availability_date", "gics_sector",
    "revenue_ttm", "net_income_ttm", "gross_profit_ttm", "op_cashflow_ttm",
    "ebitda_ttm", "equity", "assets", "debt", "cash", "shares",
    "roe", "gross_margin", "profit_margin", "accruals",
    "revenue_growth_yoy", "asset_growth_yoy",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def derive_fundamental_features(
    facts: pd.DataFrame, sectors: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Roll raw quarterly facts up into the per-quarter feature table.

    Parameters
    ----------
    facts
        Long raw facts: ``ticker, concept, period_end, filed_date, value``
        (extra columns ignored). Restatement history (multiple ``filed_date``
        per period) is collapsed to the **as-first-reported** value.
    sectors
        Optional ``ticker -> gics_sector`` frame (the ``universe`` table); joined
        on ``ticker`` so each feature row carries its sector for neutralization.

    Returns a frame with :data:`OUTPUT_COLUMNS`, one row per
    ``(ticker, period_end)``, sorted by ``(ticker, period_end)``.
    """
    if facts is None or facts.empty:
        return _empty()

    f = facts[["ticker", "concept", "period_end", "filed_date", "value"]].copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    f["filed_date"] = pd.to_datetime(f["filed_date"])

    # As-first-reported (PIT): earliest filing wins per (ticker, concept, period).
    f = f.sort_values("filed_date")
    first = f.groupby(["ticker", "concept", "period_end"], as_index=False).first()

    # Wide value matrix: index (ticker, period_end), one column per concept.
    wide = (
        first.pivot_table(
            index=["ticker", "period_end"], columns="concept", values="value",
            aggfunc="first",
        )
        .sort_index()
    )
    # Availability = when the whole row was public: the latest of the per-concept
    # earliest filings present in that period (conservative, no lookahead).
    avail = first.groupby(["ticker", "period_end"])["filed_date"].max()

    # Drop tickers that never report revenue or equity (junk / non-operating).
    have = wide.reset_index().groupby("ticker").apply(
        lambda g: g.get("revenue", pd.Series(dtype=float)).notna().any()
        and g.get("equity", pd.Series(dtype=float)).notna().any(),
        include_groups=False,
    )
    keep_tickers = have[have].index
    wide = wide[wide.index.get_level_values("ticker").isin(keep_tickers)]
    if wide.empty:
        return _empty()

    g = wide.groupby(level="ticker")
    out = pd.DataFrame(index=wide.index)

    # TTM (trailing 4 quarters) for flow concepts present.
    for c in _FLOWS:
        if c in wide.columns:
            out[f"{c}_ttm"] = g[c].transform(lambda s: s.rolling(4, min_periods=4).sum())

    # Balance-sheet levels (point-in-time, no TTM).
    for c in ("equity", "assets", "cash", "shares"):
        out[c] = wide[c] if c in wide.columns else np.nan
    debt_lt = wide["debt_lt"] if "debt_lt" in wide.columns else pd.Series(np.nan, index=wide.index)
    debt_cur = wide["debt_cur"] if "debt_cur" in wide.columns else pd.Series(np.nan, index=wide.index)
    out["debt"] = debt_lt.fillna(0) + debt_cur.fillna(0)
    out["debt"] = out["debt"].replace(0, np.nan)

    # EBITDA ≈ operating income + D&A (EDGAR tags no EBITDA directly).
    out["ebitda_ttm"] = out.get("operating_income_ttm", pd.Series(np.nan, index=out.index))
    if "dna_ttm" in out.columns:
        out["ebitda_ttm"] = out["ebitda_ttm"].add(out["dna_ttm"], fill_value=0)

    # Price-independent ratios.
    rev = out.get("revenue_ttm")
    ni = out.get("net_income_ttm")
    out["roe"] = ni / out["equity"].replace(0, np.nan)
    out["gross_margin"] = out.get("gross_profit_ttm") / rev.replace(0, np.nan)
    out["profit_margin"] = ni / rev.replace(0, np.nan)
    out["accruals"] = (ni - out.get("op_cashflow_ttm")) / out["assets"].replace(0, np.nan)
    out["revenue_growth_yoy"] = g["revenue"].transform(lambda s: s.pct_change(4)) if "revenue" in wide.columns else np.nan
    out["asset_growth_yoy"] = g["assets"].transform(lambda s: s.pct_change(4)) if "assets" in wide.columns else np.nan

    out["availability_date"] = avail.reindex(out.index).values
    out = out.reset_index()

    if sectors is not None and not sectors.empty:
        sec = sectors[["ticker", "gics_sector"]].drop_duplicates("ticker")
        out = out.merge(sec, on="ticker", how="left")
    else:
        out["gics_sector"] = np.nan
    # Coerce junk sector labels (e.g. a stale "NaN"/"None" string from an earlier
    # remap) to real NaN so they never form a spurious sector-neutralization group.
    out["gics_sector"] = out["gics_sector"].replace(
        {"NaN": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan, "": np.nan}
    )

    # Ensure every contract column exists, then order.
    for c in OUTPUT_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[OUTPUT_COLUMNS].sort_values(["ticker", "period_end"]).reset_index(drop=True)
