"""Parquet cache helpers.

Thin wrappers so the rest of the library never hard-codes paths and every
artifact lands in ``data/processed/`` (or ``data/raw/``) consistently.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, ensure_dirs


def _resolve(name: str, raw: bool) -> Path:
    base = DATA_RAW if raw else DATA_PROCESSED
    return base / name


def save(df: pd.DataFrame, name: str, *, raw: bool = False) -> Path:
    """Write a DataFrame to ``data/{raw|processed}/<name>`` as parquet.

    Writes to a temp file then atomically renames, so a crashed or interrupted
    job (common over a network/NAS mount) never leaves a half-written parquet.
    """
    ensure_dirs()
    path = _resolve(name, raw)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)  # atomic within the same filesystem/export
    return path


def load(name: str, *, raw: bool = False) -> pd.DataFrame:
    """Read a parquet artifact from the cache."""
    return pd.read_parquet(_resolve(name, raw))


def exists(name: str, *, raw: bool = False) -> bool:
    """True if the named artifact is present in the cache."""
    return _resolve(name, raw).exists()
