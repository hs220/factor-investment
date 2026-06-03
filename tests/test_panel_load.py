"""Tests for the panel_monthly loader contract (no DB)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import db


def _full_panel_row() -> pd.DataFrame:
    return pd.DataFrame([{c: 0.0 for c in db.PANEL_COLUMNS}]).assign(
        date=pd.Timestamp("2020-01-31"), ticker="AAA", gics_sector="Financials"
    )


def test_panel_columns_match_table_ddl():
    """PANEL_COLUMNS must equal the panel_monthly DDL column order (the table the
    loader writes to). Catches drift between the feature set and the schema."""
    ddl_cols = [
        line.strip().split()[0]
        for line in db._PANEL_DDL.splitlines()
        if line.strip() and line.strip().split()[0].islower()
        and not line.strip().startswith("PRIMARY")
    ]
    assert ddl_cols == db.PANEL_COLUMNS


def test_load_empty_is_noop():
    assert db.load_panel_monthly(pd.DataFrame()) == 0


def test_load_missing_columns_raises_before_db():
    """A panel missing feature columns fails fast (no DB call needed)."""
    bad = pd.DataFrame({"date": [pd.Timestamp("2020-01-31")], "ticker": ["AAA"]})
    with pytest.raises(ValueError, match="missing expected columns"):
        db.load_panel_monthly(bad)
