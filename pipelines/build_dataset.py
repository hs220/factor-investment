"""Build the raw dataset: universe, prices, fundamentals, factors, macro.

Schedulable entry point. Orchestrates src/data functions and caches every
artifact to data/processed/ for the feature-panel stage to consume.

Usage:
    python -m pipelines.build_dataset                # full run
    python -m pipelines.build_dataset --quick 50     # first N tickers (smoke test)
    python -m pipelines.build_dataset --skip-fundamentals
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_config
from src.data import cache, factors, fundamentals, prices, universe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0, help="limit to first N tickers")
    ap.add_argument("--skip-fundamentals", action="store_true")
    args = ap.parse_args()

    data_cfg = load_config("data")
    start = data_cfg["prices"]["start_date"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # 1. Universe -------------------------------------------------------------
    print("[1/5] Fetching Russell 3000 universe...")
    uni = universe.fetch_russell3000()
    print(f"      {len(uni)} raw constituents")

    # 2. Prices & volume ------------------------------------------------------
    tickers = uni.index.tolist()
    if args.quick:
        tickers = tickers[: args.quick]
        uni = uni.loc[tickers]
        print(f"      QUICK mode: limited to {len(tickers)} tickers")

    print(f"[2/5] Downloading prices/volume for {len(tickers)} tickers...")
    close = prices.download_prices(tickers, start, end, field="Close")
    volume = prices.download_prices(tickers, start, end, field="Volume")

    # Liquidity filter using avg dollar volume, then realign price set.
    adv = prices.avg_dollar_volume(close, volume.reindex_like(close))
    uni = universe.apply_liquidity_filters(uni, dollar_volume=adv)
    keep = [t for t in uni.index if t in close.columns]
    close = close[keep]
    uni = uni.loc[keep]
    print(f"      {len(keep)} tickers pass liquidity filters")

    monthly_returns = prices.to_monthly_returns(close)
    cache.save(uni.reset_index(), "universe.parquet")
    cache.save(monthly_returns, "stock_returns_monthly.parquet")

    # 3. Fundamentals (slow path) --------------------------------------------
    fund_cfg = data_cfg["fundamentals"]
    fname = fund_cfg["cache_file"]
    if args.skip_fundamentals:
        print("[3/5] Skipping fundamentals (flag set)")
    elif cache.exists(fname) and not fund_cfg.get("refresh", False):
        print(f"[3/5] Fundamentals cache present ({fname}) - skipping fetch")
    else:
        print(f"[3/5] Fetching quarterly fundamentals for {len(keep)} tickers...")
        fund = fundamentals.fetch_fundamentals(keep)
        if not fund.empty:
            cache.save(fund, fname)

    # 4. Factors --------------------------------------------------------------
    print("[4/5] Fetching FF5 + Momentum...")
    cache.save(factors.load_factors(start, end), "ff5_monthly.parquet")

    # 5. Macro ----------------------------------------------------------------
    print("[5/5] Fetching macro regime series...")
    cache.save(factors.load_macro(start, end), "macro_monthly.parquet")

    print("Done. Artifacts in data/processed/")


if __name__ == "__main__":
    main()
