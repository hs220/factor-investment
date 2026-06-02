"""Tests for the raw-facts -> quarterly feature derivation.

Run: python -m pytest tests/test_fundamental_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.factors.fundamentals_features import (  # noqa: E402
    OUTPUT_COLUMNS,
    derive_fundamental_features,
)

_QUARTERS = ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31"]


def _fact(ticker, concept, period_end, value):
    pe = pd.Timestamp(period_end)
    return {
        "ticker": ticker,
        "concept": concept,
        "period_end": pe,
        "filed_date": pe + pd.Timedelta(days=45),  # ~6 weeks after period close
        "value": float(value),
    }


def _aaa_facts() -> list[dict]:
    rows = []
    # flows (discrete quarterly), monotone for easy TTM/yoy checks
    flows = {
        "revenue": [100, 110, 120, 130, 140],
        "net_income": [10, 11, 12, 13, 14],
        "gross_profit": [50, 55, 60, 65, 70],
        "op_cashflow": [8, 9, 10, 11, 12],
        "operating_income": [20, 20, 20, 20, 20],
        "dna": [5, 5, 5, 5, 5],
    }
    stocks = {"equity": 200, "assets": 500, "cash": 50,
              "debt_lt": 100, "debt_cur": 20, "shares": 1000}
    for q_i, pe in enumerate(_QUARTERS):
        for c, vals in flows.items():
            rows.append(_fact("AAA", c, pe, vals[q_i]))
        for c, v in stocks.items():
            rows.append(_fact("AAA", c, pe, v))
    return rows


def _build():
    facts = pd.DataFrame(_aaa_facts())
    sectors = pd.DataFrame({"ticker": ["AAA"], "gics_sector": ["Information Technology"]})
    return derive_fundamental_features(facts, sectors)


def test_output_contract():
    out = _build()
    assert list(out.columns) == OUTPUT_COLUMNS
    assert (out["ticker"] == "AAA").all()


def test_ttm_and_ratios_latest_quarter():
    out = _build()
    row = out[out.period_end == pd.Timestamp("2023-03-31")].iloc[0]
    # TTM = trailing 4 quarters (Q2'22..Q1'23)
    assert row.revenue_ttm == 110 + 120 + 130 + 140          # 500
    assert row.net_income_ttm == 11 + 12 + 13 + 14           # 50
    assert row.op_cashflow_ttm == 9 + 10 + 11 + 12           # 42
    # ebitda = operating_income_ttm (80) + dna_ttm (20)
    assert row.ebitda_ttm == 80 + 20                          # 100
    assert np.isclose(row.roe, 50 / 200)                      # 0.25
    assert np.isclose(row.profit_margin, 50 / 500)            # 0.10
    assert np.isclose(row.gross_margin, (55 + 60 + 65 + 70) / 500)
    assert np.isclose(row.accruals, (50 - 42) / 500)
    assert np.isclose(row.revenue_growth_yoy, 140 / 100 - 1)  # 0.40
    assert row.debt == 120                                     # 100 + 20
    assert row.gics_sector == "Information Technology"


def test_availability_is_latest_filing():
    out = _build()
    row = out[out.period_end == pd.Timestamp("2023-03-31")].iloc[0]
    assert row.availability_date == pd.Timestamp("2023-03-31") + pd.Timedelta(days=45)


def test_ttm_requires_four_quarters():
    out = _build()
    # First quarter has no 4-quarter history -> TTM NaN
    first = out[out.period_end == pd.Timestamp("2022-03-31")].iloc[0]
    assert pd.isna(first.revenue_ttm)


def test_first_reported_wins_on_restatement():
    facts = pd.DataFrame(_aaa_facts())
    # A later restatement of Q1'23 revenue must NOT override the first report.
    restated = _fact("AAA", "revenue", "2023-03-31", 999)
    restated["filed_date"] = pd.Timestamp("2023-03-31") + pd.Timedelta(days=400)
    facts = pd.concat([facts, pd.DataFrame([restated])], ignore_index=True)
    out = derive_fundamental_features(facts, None)
    row = out[out.period_end == pd.Timestamp("2023-03-31")].iloc[0]
    assert row.revenue_ttm == 110 + 120 + 130 + 140  # original 140, not 999


def test_gate_drops_ticker_without_revenue_or_equity():
    facts = pd.DataFrame(_aaa_facts())
    # BBB reports only net_income + assets -> no revenue, no equity -> dropped
    bbb = [_fact("BBB", "net_income", pe, 5) for pe in _QUARTERS]
    bbb += [_fact("BBB", "assets", pe, 100) for pe in _QUARTERS]
    facts = pd.concat([facts, pd.DataFrame(bbb)], ignore_index=True)
    out = derive_fundamental_features(facts, None)
    assert set(out.ticker.unique()) == {"AAA"}


def test_junk_sector_labels_coerced_to_nan():
    facts = pd.DataFrame(_aaa_facts())
    sectors = pd.DataFrame({"ticker": ["AAA"], "gics_sector": ["NaN"]})  # stale junk
    out = derive_fundamental_features(facts, sectors)
    assert out["gics_sector"].isna().all()  # "NaN" string -> real NaN, not a group


def test_empty_input():
    assert derive_fundamental_features(pd.DataFrame(), None).empty


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
