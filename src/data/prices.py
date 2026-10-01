"""Price and volume history via yfinance.

Chunked batch download with retry + exponential backoff, because yfinance
rate-limits aggressively on bulk (thousands of tickers) requests. Tickers that
fail a chunk are retried in later rounds with growing backoff, so a transient
rate-limit doesn't silently drop a liquid name from the universe.

Resampled to month-end; also derives average dollar-volume for liquidity
filtering in :mod:`src.data.universe`.
"""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from src.config import load_config


def _download_field(chunk: list[str], start: str, end: str, field: str) -> pd.DataFrame:
    """Single yfinance call -> wide (dates x tickers) for one field."""
    raw = yf.download(chunk, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        col = raw[field] if field in raw.columns.get_level_values(0) else pd.DataFrame()
    else:
        col = raw[[field]] if field in raw.columns else pd.DataFrame()
        if isinstance(col, pd.DataFrame) and len(col.columns) == 1:
            col.columns = [chunk[0]]
    return col


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    field: str = "Close",
    chunk_size: int | None = None,
) -> pd.DataFrame:
    """Download a single OHLCV field for many tickers, chunked with retries.

    Failed tickers (rate-limited or transient errors) are retried in successive
    rounds with exponential backoff. Genuinely delisted/empty names are dropped
    after retries are exhausted.
    """
    cfg = load_config("data")["prices"]
    chunk_size = chunk_size or cfg["chunk_size"]
    chunk_sleep = cfg.get("chunk_sleep", 1.0)
    max_retries = cfg.get("max_retries", 3)
    backoff_base = cfg.get("backoff_base", 5.0)

    collected: dict[str, pd.Series] = {}
    remaining = list(dict.fromkeys(tickers))

    for attempt in range(max_retries + 1):
        if not remaining:
            break
        if attempt > 0:
            wait = backoff_base * (2 ** (attempt - 1))
            print(f"  retry {attempt}/{max_retries}: {len(remaining)} tickers "
                  f"after {wait:.0f}s backoff", flush=True)
            time.sleep(wait)

        failed: list[str] = []
        for i in range(0, len(remaining), chunk_size):
            chunk = remaining[i : i + chunk_size]
            try:
                col = _download_field(chunk, start, end, field)
            except Exception:  # noqa: BLE001 - yfinance raises many shapes
                col = pd.DataFrame()
            for t in chunk:
                if t in col.columns and col[t].notna().any():
                    collected[t] = col[t]
                else:
                    failed.append(t)
            time.sleep(chunk_sleep)
        remaining = failed

    if not collected:
        return pd.DataFrame()
    return pd.DataFrame(collected).sort_index()


_BAR_FIELDS = ("Open", "High", "Low", "Close", "Volume")


def download_daily_bars(
    tickers: list[str],
    start: str,
    end: str,
    *,
    chunk_size: int | None = None,
) -> pd.DataFrame:
    """Daily adjusted OHLCV as a long frame (date, ticker, open..volume).

    One yfinance call per chunk returns every field (``download_prices`` would
    re-download per field). Same chunking + retry/backoff policy as
    ``download_prices``; tickers with no bars after the retries are dropped.
    """
    cfg = load_config("data")["prices"]
    chunk_size = chunk_size or cfg["chunk_size"]
    chunk_sleep = cfg.get("chunk_sleep", 1.0)
    max_retries = cfg.get("max_retries", 3)
    backoff_base = cfg.get("backoff_base", 5.0)

    frames: list[pd.DataFrame] = []
    remaining = list(dict.fromkeys(tickers))
    for attempt in range(max_retries + 1):
        if not remaining:
            break
        if attempt > 0:
            time.sleep(backoff_base * (2 ** (attempt - 1)))
        failed: list[str] = []
        for i in range(0, len(remaining), chunk_size):
            chunk = remaining[i : i + chunk_size]
            try:
                raw = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                  progress=False, group_by="column")
            except Exception:  # noqa: BLE001 - yfinance raises many shapes
                raw = pd.DataFrame()
            got: set[str] = set()
            if raw is not None and not raw.empty:
                if not isinstance(raw.columns, pd.MultiIndex):   # single ticker
                    raw.columns = pd.MultiIndex.from_product([raw.columns, chunk[:1]])
                fields = [f for f in _BAR_FIELDS if f in raw.columns.get_level_values(0)]
                long = raw[fields].stack(level=1, future_stack=True)
                long.index.names = ["date", "ticker"]
                long = long.dropna(subset=["Close"]).reset_index()
                long.columns = [c.lower() for c in long.columns]
                frames.append(long)
                got = set(long["ticker"])
            failed += [t for t in chunk if t not in got]
            time.sleep(chunk_sleep)
        remaining = failed
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def last_cached_month(monthly: pd.DataFrame | None) -> pd.Timestamp | None:
    """Most recent month-end already in a cached monthly frame, if any."""
    if monthly is None or monthly.empty:
        return None
    return pd.Timestamp(monthly.index.max())


def merge_monthly(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Merge new monthly rows into cached ones; new values win on overlap."""
    if old is None or old.empty:
        return new.sort_index()
    merged = new.combine_first(old)
    return merged.sort_index()


def to_monthly_close(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end close price levels (for market cap / valuation ratios)."""
    return daily_prices.resample("ME").last()


def find_rebased(
    new_monthly: pd.DataFrame,
    stored_monthly: pd.DataFrame,
    *,
    before: pd.Timestamp,
    tol: float = 0.02,
) -> list[str]:
    """Tickers whose freshly downloaded closes disagree with the stored ones.

    yfinance returns split/dividend-*adjusted* closes and re-adjusts a ticker's
    whole history after each corporate action. Appending a fresh incremental
    download to history stored on the old basis leaves a fake jump at the seam
    (a 1:10 reverse split reads as +900%), which polluted momentum, volatility
    and realized returns. Compare the overlap months completed ``before`` the
    newest stored month (that one is still a partial-month close) and flag any
    ticker whose median |new/stored - 1| exceeds ``tol`` — its history must be
    re-fetched in full, not appended to.
    """
    idx = new_monthly.index.intersection(stored_monthly.index)
    idx = idx[idx < before]
    cols = new_monthly.columns.intersection(stored_monthly.columns)
    if len(idx) == 0 or len(cols) == 0:
        return []
    ratio = new_monthly.loc[idx, cols] / stored_monthly.loc[idx, cols]
    dev = (ratio - 1).abs().median()
    return sorted(dev[dev > tol].index)


def to_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end resample -> simple monthly returns."""
    return to_monthly_close(daily_prices).pct_change().dropna(how="all")


def avg_dollar_volume(
    daily_close: pd.DataFrame, daily_volume: pd.DataFrame, window: int = 60
) -> pd.Series:
    """Trailing-``window`` average daily dollar volume, latest value per ticker."""
    dollar = (daily_close * daily_volume).rolling(window, min_periods=window // 2).mean()
    return dollar.iloc[-1]
