"""Regression test for db.to_records NaN/NaT -> None conversion.

Guards a pandas-3.x gotcha: missing values in str/extension-dtype columns must
become real None (SQL NULL), not a float nan that psycopg2 writes as text 'NaN'.

Run: python -m pytest tests/test_db_records.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.db import to_records  # noqa: E402


def test_nan_in_str_column_becomes_none():
    # 'g' is a string column with a missing value (the gics_sector case).
    df = pd.DataFrame({"a": [1.0, 2.0], "g": ["X", np.nan]})
    recs = to_records(df)
    assert recs[0] == (1.0, "X")
    assert recs[1][1] is None  # NaN string-col -> None, not float('nan')


def test_nan_in_float_column_becomes_none():
    df = pd.DataFrame({"a": [1.0, np.nan]})
    recs = to_records(df)
    assert recs[1][0] is None


def test_nat_becomes_none():
    df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", None])})
    recs = to_records(df)
    assert recs[1][0] is None


def test_no_value_rendered_as_nan_text():
    df = pd.DataFrame({"g": [np.nan, "Energy", None]})
    flat = [r[0] for r in to_records(df)]
    assert flat[0] is None and flat[2] is None
    assert "NaN" not in flat  # never the literal text


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


def test_load_universe_writes_only_supplied_columns(monkeypatch):
    """The listing refresh carries no sector; it must not upsert NULL over it."""
    from src.data import db

    seen = {}
    monkeypatch.setattr(db, "upsert", lambda df, table, conflict: seen.update(cols=list(df.columns)) or len(df))
    db.load_universe(pd.DataFrame({"ticker": ["A"], "name": ["a"], "exchange": ["X"]}))
    assert seen["cols"] == ["ticker", "name", "exchange"]          # no gics_sector
    db.load_universe(pd.DataFrame({"ticker": ["A"], "gics_sector": ["Energy"]}))
    assert seen["cols"] == ["ticker", "gics_sector"]


def test_set_active_refuses_collapse(monkeypatch):
    import pytest

    from src.data import db

    monkeypatch.setattr(db, "read_sql", lambda q: pd.DataFrame({"c": [4000]}))
    with pytest.raises(ValueError, match="refusing to shrink"):
        db.set_active([f"T{i}" for i in range(50)])
