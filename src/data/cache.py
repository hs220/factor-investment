"""Parquet cache helpers.

Thin wrappers so the rest of the library never hard-codes paths and every
artifact lands in ``data/processed/`` (or ``data/raw/``) consistently.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, ensure_dirs


def _resolve(name: str, raw: bool) -> Path:
    base = DATA_RAW if raw else DATA_PROCESSED
    return base / name


def save(df: pd.DataFrame, name: str, *, raw: bool = False) -> Path:
    """Write a DataFrame to ``data/{raw|processed}/<name>`` as parquet."""
    ensure_dirs()
    path = _resolve(name, raw)
    df.to_parquet(path)
    return path


def load(name: str, *, raw: bool = False) -> pd.DataFrame:
    """Read a parquet artifact from the cache."""
    return pd.read_parquet(_resolve(name, raw))


def exists(name: str, *, raw: bool = False) -> bool:
    """True if the named artifact is present in the cache."""
    return _resolve(name, raw).exists()
