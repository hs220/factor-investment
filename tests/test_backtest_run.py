"""run_strategy_backtest: synthetic OOS predictions, no warehouse needed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.run import run_strategy_backtest


def _preds(n_dates=24, n_names=60, live_month=True, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    rows = []
    for d in dates:
        pred = rng.normal(size=n_names)
        fr = 0.01 * pred + rng.normal(scale=0.05, size=n_names)   # signal + noise
        rows.append(pd.DataFrame({"date": d, "ticker": [f"T{i}" for i in range(n_names)],
                                  "pred": pred, "forward_return": fr}))
    df = pd.concat(rows, ignore_index=True)
    if live_month:   # newest month: scores but no realized return yet
        last = df[df["date"] == dates[-1]].copy()
        last["date"] = dates[-1] + pd.offsets.MonthEnd(1)
        last["forward_return"] = np.nan
        df = pd.concat([df, last], ignore_index=True)
    return df, dates


def test_backtest_excludes_unrealized_month_and_beats_benchmark():
    preds, dates = _preds()
    res = run_strategy_backtest(preds, n_holdings=10)
    # live month dropped; rows keyed by realization month (formation + 1)
    assert list(res.returns.index) == list(dates + pd.offsets.MonthEnd(1))
    assert {"gross", "net", "cost", "turnover", "benchmark"} <= set(res.returns.columns)
    assert (res.returns["net"] <= res.returns["gross"]).all()  # costs only subtract
    assert res.strategy["ann_return"] > res.benchmark["ann_return"]   # signal is real
    assert res.holdings.groupby("date")["weight"].sum().round(6).eq(1).all()
    assert "error" in res.attribution                          # no factors passed


def test_backtest_attribution_with_factors():
    preds, dates = _preds()
    rng = np.random.default_rng(1)
    factors = pd.DataFrame(rng.normal(scale=0.03, size=(len(dates), 7)),
                           index=dates + pd.offsets.MonthEnd(1),
                           columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"])
    factors["RF"] = 0.001
    res = run_strategy_backtest(preds, factors, n_holdings=10)
    assert res.attribution["n_months"] == len(dates)
    assert set(res.attribution["betas"]) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"}


def test_attribution_aligns_realization_month():
    """A strategy that IS the market (realized next month) must load ~1 on Mkt-RF."""
    rng = np.random.default_rng(2)
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    mkt = pd.Series(rng.normal(0.01, 0.04, size=61), index=dates.append(dates[-1:] + pd.offsets.MonthEnd(1)))
    rows = [pd.DataFrame({"date": d, "ticker": [f"T{i}" for i in range(20)], "pred": rng.normal(size=20),
                          "forward_return": mkt.loc[d + pd.offsets.MonthEnd(1)]}) for d in dates]
    factors = pd.DataFrame(0.0, index=mkt.index, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"])
    factors["Mkt-RF"] = mkt
    factors[["SMB", "HML", "RMW", "CMA", "MOM"]] = rng.normal(scale=0.02, size=(61, 5))
    res = run_strategy_backtest(pd.concat(rows, ignore_index=True), factors, n_holdings=20)
    assert res.attribution["betas"]["Mkt-RF"] > 0.9 and res.attribution["r_squared"] > 0.9
