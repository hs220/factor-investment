"""Assemble the (ticker, month) feature panel.

Pipeline:
  1. Base long frame from monthly returns -> (date, ticker, ret).
  2. Price/technical features (momentum, volatility) computed per ticker.
  3. Point-in-time fundamental join: for each (ticker, month) pick the most
     recent quarter whose ``availability_date <= month-end`` (merge_asof).
  3b. Sector tag from the universe table (ticker-level), independent of the
     fundamentals join, so sector-neutral normalization keeps price/macro
     features on names without a fundamental filing.
  4. Valuation ratios from fundamentals + month-end market cap (so they update
     monthly while accounting inputs update on filing cadence).
  5. Macro regime columns broadcast across all stocks at each date.
  6. Cross-sectional, sector-neutral normalization of all features.
  7. Target: 1-month forward return as a within-sector cross-sectional rank.

No lookahead: features at month t use only data known by end of t; the target
covers (t, t+1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import cache
from src.factors.normalize import normalize_features


def _melt(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Wide (date x ticker) -> long (date, ticker, value)."""
    long = wide.stack().rename(value_name).reset_index()
    long.columns = ["date", "ticker", value_name]
    return long


def compute_price_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Momentum and volatility from monthly returns (wide -> long).

    Definitions (all use only information available at month t):
    - momentum_12_2: return over [t-12, t-2] (skip the most recent month).
    - momentum_6_1:  return over [t-6, t-1].
    - volatility_12m: stdev of the trailing 12 monthly returns.
    """
    lr = np.log1p(returns)
    mom_12_2 = np.expm1(lr.shift(2).rolling(11).sum())
    mom_6_1 = np.expm1(lr.shift(1).rolling(6).sum())
    vol_12m = returns.rolling(12).std()

    out = _melt(mom_12_2, "momentum_12_2")
    out = out.merge(_melt(mom_6_1, "momentum_6_1"), on=["date", "ticker"], how="outer")
    out = out.merge(_melt(vol_12m, "volatility_12m"), on=["date", "ticker"], how="outer")
    return out


def pit_join_fundamentals(panel: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time merge: latest quarter with availability_date <= date."""
    fund = fund.copy()
    panel = panel.copy()
    # merge_asof requires identical datetime resolution on both keys.
    fund["availability_date"] = fund["availability_date"].astype("datetime64[ns]")
    panel["date"] = panel["date"].astype("datetime64[ns]")
    fund = fund.sort_values("availability_date")
    panel = panel.sort_values("date")
    keep = [
        "ticker", "availability_date",
        "net_income_ttm", "ebitda_ttm", "equity", "assets", "debt", "cash", "shares",
        "roe", "gross_margin", "profit_margin", "accruals",
        "revenue_growth_yoy", "asset_growth_yoy",
    ]
    merged = pd.merge_asof(
        panel,
        fund[keep],
        left_on="date",
        right_on="availability_date",
        by="ticker",
        direction="backward",
    )
    return merged


def compute_valuation_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Price-dependent value features from fundamentals + month-end market cap."""
    out = panel.copy()
    mktcap = out["price"] * out["shares"]
    out["market_cap"] = mktcap
    out["earnings_yield"] = out["net_income_ttm"] / mktcap.replace(0, np.nan)
    out["book_to_price"] = out["equity"] / mktcap.replace(0, np.nan)
    ev = mktcap + out["debt"].fillna(0) - out["cash"].fillna(0)
    out["ev_ebitda_inv"] = out["ebitda_ttm"] / ev.replace(0, np.nan)
    out["size_log_mktcap"] = np.log(mktcap.replace(0, np.nan))
    return out


def _feature_list() -> list[str]:
    f = load_config("features")["features"]
    cols = list(f["price"])
    for grp in f["fundamental"].values():
        cols += list(grp)
    cols += list(f["macro"])
    return cols


def build_target(returns: pd.DataFrame) -> pd.DataFrame:
    """1-month forward return per (date, ticker). date t -> return over (t, t+1]."""
    fwd = returns.shift(-1)
    return _melt(fwd, "forward_return")


def _load_inputs(source: str):
    """Return (returns, prices, fund, macro, sectors) from warehouse or cache.

    ``sectors`` is a ticker -> gics_sector map used to tag every row's sector
    independently of the fundamentals join, so price/macro features survive on
    names that have a sector in the universe but no fundamental filing yet.
    """
    if source == "db":
        from src.data import warehouse
        return (
            warehouse.load_returns_wide(),
            warehouse.load_prices_wide(),
            warehouse.load_fundamental_features(),
            warehouse.load_macro(),
            warehouse.load_sectors(),
        )
    if source == "cache":
        fund = cache.load("fundamentals_quarterly.parquet")
        # Legacy offline path: derive the ticker->sector map from fundamentals
        # (preserves the cache path's historical behavior).
        sectors = fund[["ticker", "gics_sector"]] if "gics_sector" in fund else None
        return (
            cache.load("stock_returns_monthly.parquet"),
            cache.load("stock_prices_monthly.parquet"),
            fund,
            cache.load("macro_monthly.parquet"),
            sectors,
        )
    raise ValueError(f"unknown source {source!r}; use 'db' or 'cache'")


def assemble_panel(*, source: str = "db", normalize: bool = True) -> pd.DataFrame:
    """Build the full (ticker, month) panel.

    ``source="db"`` (default) reads the Postgres warehouse — the production
    source of truth; ``source="cache"`` reads the legacy parquet artifacts in
    ``data/processed/`` for offline use.
    """
    fcfg = load_config("features")
    pcfg = fcfg["panel"]
    ncfg = fcfg["normalization"]

    returns, prices, fund, macro, sectors = _load_inputs(source)

    # 1-2. base + price features
    panel = compute_price_features(returns)
    panel = panel.merge(_melt(prices, "price"), on=["date", "ticker"], how="left")

    # 3. PIT fundamentals
    panel = pit_join_fundamentals(panel, fund)

    # 3b. sector tag from the universe (NOT the fundamentals join): every name
    #     with a known sector keeps its price/macro features under sector-neutral
    #     normalization, even if it has no fundamental filing yet.
    if sectors is not None:
        smap = sectors.dropna(subset=["gics_sector"]).drop_duplicates("ticker")
        panel = panel.merge(smap[["ticker", "gics_sector"]], on="ticker", how="left")
    else:
        panel["gics_sector"] = np.nan

    # 4. valuation ratios
    panel = compute_valuation_ratios(panel)

    # 5. macro broadcast (join on date)
    macro_long = macro.copy()
    macro_long.index.name = "date"
    panel = panel.merge(macro_long.reset_index(), on="date", how="left")

    # 6. target (within-sector cross-sectional rank of forward return)
    panel = panel.merge(build_target(returns), on=["date", "ticker"], how="left")

    features = _feature_list()
    panel = panel[
        ["date", "ticker", "gics_sector", "forward_return", "market_cap"] + features
    ]

    # 7. normalize features cross-sectionally (sector-neutral)
    if normalize:
        panel = normalize_features(
            panel,
            features,
            scheme=ncfg["scheme"],
            sector_neutral=ncfg["sector_neutral"],
            sector_col=pcfg["sector_field"],
            winsorize_pct=ncfg.get("winsorize_pct", 0.01),
        )
        # target as within-sector cross-sectional rank
        panel["target"] = panel.groupby(["date", pcfg["sector_field"]])[
            "forward_return"
        ].rank(pct=True)

    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
