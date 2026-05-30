"""Build the raw dataset: universe, prices, fundamentals, factors, macro.

Schedulable entry point. Orchestrates src/data functions and caches every
artifact to data/processed/ for the feature-panel stage to consume.

Two modes:
  - **Backfill** (default): full history for all names. Slow, high query volume;
    run once or on a rare full rebuild.
  - **Incremental** (--incremental): extend prices from the last cached month
    and only re-pull fundamentals for tickers with a *new* SEC filing. Cheap;
    this is the scheduled weekly/monthly mode.

Usage:
    python -m pipelines.build_dataset                 # full backfill
    python -m pipelines.build_dataset --quick 50      # smoke test (first N)
    python -m pipelines.build_dataset --incremental   # scheduled update
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
    ap.add_argument("--incremental", action="store_true",
                    help="extend prices from last cached month; only re-pull "
                         "fundamentals for tickers with a new filing")
    ap.add_argument("--skip-fundamentals", action="store_true")
    args = ap.parse_args()

    data_cfg = load_config("data")
    backfill_start = data_cfg["prices"]["start_date"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # 1. Universe -------------------------------------------------------------
    print("[1/5] Fetching universe...")
    uni = universe.fetch_russell3000()
    print(f"      {len(uni)} raw constituents")

    tickers = uni.index.tolist()
    if args.quick:
        tickers = tickers[: args.quick]
        uni = uni.loc[tickers]
        print(f"      QUICK mode: limited to {len(tickers)} tickers")

    # 2. Prices & volume ------------------------------------------------------
    # Incremental: only fetch from the month after the last cached month-end.
    cached_close = (
        cache.load("stock_prices_monthly.parquet")
        if args.incremental and cache.exists("stock_prices_monthly.parquet")
        else None
    )
    last_month = prices.last_cached_month(cached_close)
    price_start = backfill_start
    if last_month is not None:
        price_start = (last_month - pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d")
        print(f"[2/5] Incremental prices since {price_start} "
              f"(last cached {last_month.date()})...")
    else:
        print(f"[2/5] Backfill prices for {len(tickers)} tickers...")

    close = prices.download_prices(tickers, price_start, end, field="Close")
    volume = prices.download_prices(tickers, price_start, end, field="Volume")

    # Liquidity filter on the freshly downloaded window.
    adv = prices.avg_dollar_volume(close, volume.reindex_like(close))
    uni = universe.apply_liquidity_filters(uni, dollar_volume=adv)
    keep = [t for t in uni.index if t in close.columns]
    close = close[keep]
    uni = uni.loc[keep]
    print(f"      {len(keep)} tickers pass liquidity filters")

    monthly_close = prices.merge_monthly(cached_close, prices.to_monthly_close(close))
    cache.save(uni.reset_index(), "universe.parquet")
    cache.save(monthly_close, "stock_prices_monthly.parquet")
    cache.save(monthly_close.pct_change().dropna(how="all"),
               "stock_returns_monthly.parquet")

    # 3. Fundamentals ---------------------------------------------------------
    fund_cfg = data_cfg["fundamentals"]
    fname = fund_cfg["cache_file"]
    if args.skip_fundamentals:
        print("[3/5] Skipping fundamentals (flag set)")
    elif args.incremental and cache.exists(fname):
        # Only re-pull tickers whose latest filing post-dates our cache.
        existing = cache.load(fname)
        latest = existing.groupby("ticker")["filed_date"].max().to_dict()
        print(f"[3/5] Incremental fundamentals: checking {len(keep)} tickers "
              f"for new filings...")
        fresh = fundamentals.fetch_fundamentals(keep, existing_latest=latest)
        if not fresh.empty:
            combined = pd.concat([existing, fresh], ignore_index=True)
            # raw facts are keyed (ticker, concept, period_end, filed_date)
            combined = combined.drop_duplicates(
                subset=["ticker", "concept", "period_end", "filed_date"], keep="last"
            )
            cache.save(combined, fname)
            print(f"      merged {fresh['ticker'].nunique()} updated tickers")
    elif cache.exists(fname) and not fund_cfg.get("refresh", False):
        print(f"[3/5] Fundamentals cache present ({fname}) - skipping fetch")
    else:
        print(f"[3/5] Backfill fundamentals for {len(keep)} tickers...")
        fund = fundamentals.fetch_fundamentals(keep)
        if not fund.empty:
            cache.save(fund, fname)

    # 4. Factors --------------------------------------------------------------
    print("[4/5] Fetching FF5 + Momentum...")
    cache.save(factors.load_factors(backfill_start, end), "ff5_monthly.parquet")

    # 5. Macro ----------------------------------------------------------------
    print("[5/5] Fetching macro regime series...")
    cache.save(factors.load_macro(backfill_start, end), "macro_monthly.parquet")

    print("Done. Artifacts in data/processed/")


if __name__ == "__main__":
    main()
