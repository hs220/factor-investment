"""Tests for the library-boundary Pandera schemas (src/data/schemas.py).

Run: python -m pytest tests/test_schemas.py
(or plain `python tests/test_schemas.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.schemas import (  # noqa: E402
    LOOKAHEAD_TOL_DAYS,
    validate_fundamental_facts,
)


def _row(**over) -> dict:
    base = dict(
        ticker="AAPL",
        cik="0000320193",
        concept="revenue",
        gaap_tag="Revenues",
        period_end=pd.Timestamp("2023-03-31"),
        fiscal_period="Q1",
        duration_days=91,
        filed_date=pd.Timestamp("2023-05-05"),
        form="10-Q",
        value=1.0e11,
        unit="USD",
    )
    base.update(over)
    return base


def test_valid_frame_passes():
    df = pd.DataFrame([_row(), _row(concept="assets", duration_days=0)])
    out = validate_fundamental_facts(df)
    assert len(out) == 2


def test_empty_frame_is_noop():
    empty = pd.DataFrame(columns=["ticker", "cik", "concept"])
    assert validate_fundamental_facts(empty).empty


def test_snap_forward_within_tolerance_passes():
    # filed_date a few weeks before period_end (calendar snap artifact) is allowed
    df = pd.DataFrame([_row(
        period_end=pd.Timestamp("2023-06-30"),
        filed_date=pd.Timestamp("2023-06-01"),  # 29d early — under tolerance
    )])
    assert len(validate_fundamental_facts(df)) == 1


def test_genuine_lookahead_rejected():
    # filed long before the period it reports → real lookahead, must fail
    df = pd.DataFrame([_row(
        period_end=pd.Timestamp("2023-03-31"),
        filed_date=pd.Timestamp("2022-06-01"),  # ~300d early
    )])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_fundamental_facts(df)


def test_lookahead_boundary_is_inclusive():
    # exactly LOOKAHEAD_TOL_DAYS early passes; one day more fails
    pe = pd.Timestamp("2023-06-30")
    ok = pd.DataFrame([_row(period_end=pe,
                            filed_date=pe - pd.Timedelta(days=LOOKAHEAD_TOL_DAYS))])
    assert len(validate_fundamental_facts(ok)) == 1
    bad = pd.DataFrame([_row(period_end=pe,
                             filed_date=pe - pd.Timedelta(days=LOOKAHEAD_TOL_DAYS + 1))])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_fundamental_facts(bad)


def test_unknown_concept_rejected():
    df = pd.DataFrame([_row(concept="ebitda_made_up")])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_fundamental_facts(df)


def test_unexpected_column_rejected():
    df = pd.DataFrame([_row()]).assign(surprise=1)
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_fundamental_facts(df)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
