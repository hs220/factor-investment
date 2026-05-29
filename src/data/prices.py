"""Price and volume history via yfinance.

Chunked batch download (yfinance handles ~100 tickers/request well), resampled
to month-end. Also derives the average dollar-volume used for liquidity
filtering in :mod:`src.data.universe`.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.config import load_config


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    field: str = "Close",
    chunk_size: int | None = None,
) -> pd.DataFrame:
    """Download a single OHLCV field for many tickers, chunked.

    Returns a wide DataFrame (dates x tickers) of daily values. Tickers with
    no data at all are dropped.
    """
    if chunk_size is None:
        chunk_size = load_config("data")["prices"]["chunk_size"]

    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        raw = yf.download(
            chunk, start=start, end=end, auto_adjust=True, progress=False
        )
        # Single vs multi-ticker yields different column shapes.
        col = raw[field] if isinstance(raw.columns, pd.MultiIndex) else raw[[field]]
        if isinstance(col, pd.Series):
            col = col.to_frame(chunk[0])
        frames.append(col)

    prices = pd.concat(frames, axis=1)
    return prices.dropna(axis=1, how="all")


def to_monthly_close(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end close price levels (for market cap / valuation ratios)."""
    return daily_prices.resample("ME").last()


def to_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end resample -> simple monthly returns."""
    return to_monthly_close(daily_prices).pct_change().dropna(how="all")


def avg_dollar_volume(
    daily_close: pd.DataFrame, daily_volume: pd.DataFrame, window: int = 60
) -> pd.Series:
    """Trailing-``window`` average daily dollar volume, latest value per ticker."""
    dollar = (daily_close * daily_volume).rolling(window, min_periods=window // 2).mean()
    return dollar.iloc[-1]
