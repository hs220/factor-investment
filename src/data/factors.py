"""Fama-French factors and macro regime data.

- FF5 + Momentum: direct CSV-zip download from the Ken French data library.
- Macro: FRED public CSV API (yield curve, credit spread) + VIX via yfinance.

These are benchmark/attribution and regime inputs; both fetched at monthly
frequency aligned to month-end.
"""
from __future__ import annotations

import io
import time
import zipfile

import pandas as pd
import requests
import yfinance as yf

from src.config import load_config

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _get_with_retry(url: str, *, timeout: int = 30, retries: int = 4,
                    backoff: float = 3.0) -> requests.Response:
    """GET with retry on transient failures (5xx / timeout / connection).

    External sources (FRED, Ken French) intermittently return 502/504 or time
    out; retrying with backoff absorbs those without failing the whole asset.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp
            last_exc = requests.HTTPError(f"{resp.status_code} for {url}")
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


def _fetch_ff_csv(filename: str, skiprows: int) -> pd.DataFrame:
    """Download a Ken French CSV-zip and return the monthly data block."""
    resp = _get_with_retry(f"{FF_BASE}/{filename}")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")

    # FF CSVs: header section, then monthly rows (YYYYMM), then an annual block.
    data_lines: list[str] = []
    in_data = False
    for line in raw.splitlines()[skiprows:]:
        s = line.strip()
        if not s:
            if in_data:
                break
            continue
        if s[0].isdigit():
            in_data = True
            data_lines.append(s)
        elif in_data:
            break
    return pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)


def load_factors(start: str, end: str) -> pd.DataFrame:
    """FF5 + Momentum monthly factor returns (decimal), indexed at month-end."""
    ff5 = _fetch_ff_csv("F-F_Research_Data_5_Factors_2x3_CSV.zip", skiprows=3)
    ff5.columns = ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    ff5["date"] = pd.to_datetime(ff5["date"].astype(str), format="%Y%m")
    ff5 = ff5.set_index("date") / 100

    mom = _fetch_ff_csv("F-F_Momentum_Factor_CSV.zip", skiprows=13)
    mom.columns = ["date", "MOM"]
    mom["date"] = pd.to_datetime(mom["date"].astype(str), format="%Y%m")
    mom = mom.set_index("date") / 100

    factors = ff5.join(mom, how="left")
    factors = factors.loc[start:end]
    # Align index to month-end to match price/panel data
    factors.index = factors.index + pd.offsets.MonthEnd(0)
    return factors


def _fetch_fred(series_id: str, start: str) -> pd.Series:
    """Fetch a single FRED series as a date-indexed Series."""
    resp = _get_with_retry(f"{FRED_BASE}?id={series_id}")
    df = pd.read_csv(
        io.StringIO(resp.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    df.index.name = "date"
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(series_id).loc[start:]


def load_macro(start: str, end: str) -> pd.DataFrame:
    """Monthly macro regime features: yield curve, VIX, credit spread."""
    cfg = load_config("data")["factors"]
    yc = cfg["macro_series"]["yield_curve"]  # [long, short]
    long_s = _fetch_fred(yc[0], start)
    short_s = _fetch_fred(yc[1], start)
    credit = _fetch_fred(cfg["macro_series"]["credit_spread"], start)

    vix = yf.download(
        cfg["vix_ticker"], start=start, end=end, auto_adjust=True, progress=False
    )["Close"].squeeze()

    macro = pd.DataFrame(
        {
            "yield_curve": (long_s - short_s).resample("ME").last(),
            "vix": vix.resample("ME").last(),
            "credit_spread": credit.resample("ME").last(),
        }
    )
    return macro.loc[start:end]
