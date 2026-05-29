"""Investable universe: US-listed common stocks via NASDAQ Trader.

The official NASDAQ Trader symbol directory publishes two pipe-delimited files
covering all US-listed securities (NASDAQ + NYSE/AMEX/other). We keep common
stocks (drop ETFs, test issues, warrants/units/preferred), normalize tickers,
then trim to the liquid investable set via :func:`apply_liquidity_filters`.

Sector labels are not in this directory; they are attached downstream from the
per-ticker yfinance pass (see :mod:`src.data.fundamentals`).

Note: this is *current* listings (survivorship-biased). A point-in-time
universe with delisted names requires a paid source (see plans/modeling.md).
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from src.config import load_config

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; factor-analysis/1.0)"}
# Common-stock symbols: 1-5 letters, optional class suffix after '-'.
_TICKER_RE = r"^[A-Z]{1,5}(-[A-Z])?$"


def _normalize_ticker(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def _fetch_pipe_file(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    # Last line is a "File Creation Time" footer — drop it.
    text = "\n".join(
        ln for ln in resp.text.splitlines() if not ln.startswith("File Creation Time")
    )
    return pd.read_csv(io.StringIO(text), sep="|")


def fetch_us_listed() -> pd.DataFrame:
    """Current US-listed common stocks.

    Returns a DataFrame indexed by ticker with columns ``name``, ``exchange``.
    """
    cfg = load_config("data")["universe"]
    nasdaq = _fetch_pipe_file(cfg["nasdaq_listed_url"])
    other = _fetch_pipe_file(cfg["other_listed_url"])

    # NASDAQ file: Symbol|Security Name|Market Category|Test Issue|...|ETF|...
    nasdaq = nasdaq.rename(columns={"Symbol": "ticker", "Security Name": "name"})
    nasdaq["exchange"] = "NASDAQ"

    # Other file: ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|...|Test Issue|...
    other = other.rename(
        columns={"ACT Symbol": "ticker", "Security Name": "name", "Exchange": "exchange"}
    )

    frames = []
    for df in (nasdaq, other):
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] != "Y"]
        if "ETF" in df.columns:
            df = df[df["ETF"] != "Y"]
        frames.append(df[["ticker", "name", "exchange"]])

    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = out["ticker"].map(_normalize_ticker)
    # Drop preferred/warrant/unit/rights rows (names usually flag these).
    bad = out["name"].str.contains(
        r"(?:Warrants?|Units?|Rights?|Preferred|Depositary|Notes?|Debentures?)",
        case=False, na=False,
    )
    out = out[~bad]
    out = out[out["ticker"].str.match(_TICKER_RE, na=False)]
    out = out.dropna(subset=["ticker"]).drop_duplicates("ticker").set_index("ticker")
    return out.sort_index()


# Backwards-compatible alias used by the build pipeline.
def fetch_russell3000() -> pd.DataFrame:  # pragma: no cover - thin alias
    """Deprecated name; the universe is now US-listed common stocks."""
    return fetch_us_listed()


def apply_liquidity_filters(
    universe: pd.DataFrame, dollar_volume: pd.Series | None = None
) -> pd.DataFrame:
    """Filter to the liquid investable set using avg dollar volume.

    ``min_price`` is enforced in the pricing stage (we don't have a snapshot
    price here); ``min_dollar_volume`` is applied against ``dollar_volume`` when
    provided.
    """
    cfg = load_config("data")["universe"]["liquidity_filters"]
    filtered = universe
    if dollar_volume is not None and cfg.get("min_dollar_volume") is not None:
        keep = dollar_volume[dollar_volume >= cfg["min_dollar_volume"]].index
        filtered = filtered[filtered.index.isin(keep)]
    return filtered
