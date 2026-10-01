"""Daily multi-horizon panel: alignment, masking, no-lookahead (synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors import panel_daily as PD


def _bars(n_days=400, n_names=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    names = [f"T{i}" for i in range(n_names)]
    r = rng.normal(0.0005, 0.02, size=(n_days, n_names))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=names)
    vol = pd.DataFrame(rng.uniform(1e5, 1e6, size=close.shape), index=dates, columns=names)
    return {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": vol}


def _fund(names):
    return pd.DataFrame({
        "ticker": names, "availability_date": pd.Timestamp("2019-12-01"),
        "net_income_ttm": 1e6, "ebitda_ttm": 2e6, "equity": 5e6, "assets": 1e7, "debt": 1e6,
        "cash": 5e5, "shares": 1e5, "roe": np.linspace(0.01, 0.2, len(names)),
        "gross_margin": 0.4, "profit_margin": 0.1, "accruals": 0.0,
        "revenue_growth_yoy": 0.05, "asset_growth_yoy": 0.05})


def _sectors(names):
    return pd.DataFrame({"ticker": names, "gics_sector": ["A", "B"] * (len(names) // 2)})


def test_forward_returns_cover_t_plus_1_to_t_plus_h():
    r = pd.DataFrame({"X": [0.0, 0.10, 0.20, -0.05, 0.0]})
    fwd = PD.forward_returns(r, 2)
    assert fwd["X"].iloc[0] == pytest.approx(1.10 * 1.20 - 1)     # days 1..2, not day 0
    assert fwd["X"].iloc[3:].isna().all()                          # not enough future


def test_features_use_no_future_bars():
    bars = _bars()
    r = bars["close"].pct_change(fill_method=None)
    base = PD.compute_daily_features(bars, r)
    cut = 300                                                       # perturb the future only
    fut = {k: v.copy() for k, v in bars.items()}
    fut["close"].iloc[cut:] *= 3.0
    fut["volume"].iloc[cut:] *= 7.0
    pert = PD.compute_daily_features(fut, fut["close"].pct_change(fill_method=None))
    for name, wide in base.items():
        pd.testing.assert_frame_equal(wide.iloc[:cut], pert[name].iloc[:cut], check_names=False)


def test_assemble_masks_bad_print_and_ranks_targets():
    bars = _bars()
    names = list(bars["close"].columns)
    bars["close"].iloc[200:, 0] *= 50              # +4,900% seam on T0
    p = PD.assemble_panel_daily(bars=bars, fund=_fund(names), sectors=_sectors(names),
                                horizons=[5, 10], since="2020-06-01")
    assert p.attrs["returns_masked"] == 1
    assert {"target_5d", "target_10d", "fwd_5d", "fwd_10d"} <= set(p.columns)
    assert p["fwd_5d"].max() < 1.0                 # the seam never becomes a label
    t = p["target_10d"].dropna()
    assert t.between(0, 1).all()
    assert p["roe"].dropna().between(0, 1).all()   # normalized features are ranks
    assert p["date"].min() >= pd.Timestamp("2020-06-01")


def test_ic_by_horizon_samples_non_overlapping_dates():
    bars = _bars(n_days=300)
    names = list(bars["close"].columns)
    p = PD.assemble_panel_daily(bars=bars, fund=_fund(names), sectors=_sectors(names),
                                horizons=[5, 10])
    ic = PD.ic_by_horizon(p, ["rev_5d"], [5, 10])
    n5, n10 = ic.set_index("horizon").loc[["5d", "10d"], "n_months"]
    assert n5 > n10 > 0 and n5 <= p["date"].nunique() / 5 + 1
