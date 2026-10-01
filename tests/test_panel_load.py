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


def test_clean_returns_masks_out_of_band():
    import numpy as np
    import pandas as pd

    from src.factors.panel import clean_returns

    r = pd.DataFrame({"A": [0.05, 160.0, -0.99, 0.1], "B": [0.02, np.nan, 3.0, -0.95]})
    clean, n = clean_returns(r, [-0.95, 3.0])
    assert n == 2                                   # 160.0 and -0.99; bounds inclusive
    assert clean["A"].isna().tolist() == [False, True, True, False]
    assert clean["B"].tolist()[2:] == [3.0, -0.95]


def test_find_rebased_flags_seam_not_partial_month():
    import pandas as pd

    from src.data.prices import find_rebased

    idx = pd.to_datetime(["2026-07-31", "2026-08-31", "2026-09-30"])
    stored = pd.DataFrame({"OK": [10.0, 11.0, 12.0], "SPLIT": [1.0, 1.1, 1.2]}, index=idx)
    new = pd.DataFrame({"OK": [10.0, 11.0, 13.5],            # only the partial month moved
                        "SPLIT": [10.0, 11.0, 12.0]}, index=idx)  # 1:10 re-adjusted history
    assert find_rebased(new, stored, before=idx[-1]) == ["SPLIT"]
