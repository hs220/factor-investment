"""Backfill the ``prices_daily`` table (prototype loader for the daily panel).

One consistent full download per ticker (no incremental seams). Scope: the
current investable set (``universe.is_active``) from ``panel_daily.start_date``
in config/features.yaml. Becomes a Dagster asset when the daily panel is
productionized (with ``prices.find_rebased``-style seam handling).

Run where yfinance + the warehouse are reachable (the NAS dagster container):
    python -m scripts.backfill_prices_daily
"""
from __future__ import annotations

import time

import pandas as pd

from src.config import load_config
from src.data import db, prices

BATCH = 250          # tickers per download+upsert round (bounds memory, shows progress)


def main() -> None:
    start = load_config("features")["panel_daily"]["start_date"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = db.read_sql("SELECT ticker FROM universe WHERE is_active ORDER BY ticker")["ticker"].tolist()
    db.ensure_prices_daily_table()
    print(f"backfilling {len(tickers)} tickers {start}..{end}", flush=True)

    t0, rows, got = time.time(), 0, 0
    for i in range(0, len(tickers), BATCH):
        bars = prices.download_daily_bars(tickers[i : i + BATCH], start, end)
        rows += db.load_prices_daily(bars)
        got += bars["ticker"].nunique() if not bars.empty else 0
        print(f"  {min(i + BATCH, len(tickers))}/{len(tickers)} tickers, {rows:,} rows, "
              f"{time.time() - t0:.0f}s", flush=True)
    print(f"done: {got} tickers with data, {rows:,} rows upserted")


if __name__ == "__main__":
    main()
