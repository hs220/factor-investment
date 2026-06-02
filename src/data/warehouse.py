"""Read warehouse tables into the shapes the feature panel expects.

The panel was originally built from parquet artifacts in ``data/processed/``.
These helpers return the same shapes straight from the Postgres warehouse (the
source of truth), so ``panel.assemble_panel(source="db")`` is a drop-in swap:

- wide (date x ticker) monthly close / returns,
- the long derived ``fundamental_features`` table,
- date-indexed macro,
- ticker -> sector.

Requires DB connectivity (``POSTGRES_PASSWORD`` env + LAN access); see
``src/data/db.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import db

# Stale labels (e.g. a literal "NaN"/"None" string from an earlier sector remap)
# that must become real NaN, so they never form a spurious sector group under
# sector-neutral normalization. Mirrors the coercion in
# src/factors/fundamentals_features.py.
_JUNK_SECTORS = {"NaN": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan, "": np.nan}


def load_prices_wide() -> pd.DataFrame:
    """Monthly close as wide (date index x ticker). Prices are stored monthly."""
    df = db.read_sql("SELECT ticker, date, close FROM prices")
    wide = df.pivot(index="date", columns="ticker", values="close")
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide.sort_index()


def load_returns_wide() -> pd.DataFrame:
    """Monthly simple returns (wide), matching the old cached returns artifact."""
    return load_prices_wide().pct_change().dropna(how="all")


def load_fundamental_features() -> pd.DataFrame:
    """The gold quarterly feature table (long), dates as datetime64[ns]."""
    df = db.read_sql("SELECT * FROM fundamental_features")
    for c in ("period_end", "availability_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def load_macro() -> pd.DataFrame:
    """Macro regime series, date-indexed (yield_curve, vix, credit_spread)."""
    df = db.read_sql("SELECT date, yield_curve, vix, credit_spread FROM macro")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_sectors() -> pd.DataFrame:
    """Investable ticker -> gics_sector (the universe table), junk coerced to NaN."""
    df = db.read_sql("SELECT ticker, gics_sector FROM universe WHERE is_active")
    df["gics_sector"] = df["gics_sector"].replace(_JUNK_SECTORS)
    return df
