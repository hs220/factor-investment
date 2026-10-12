"""The serving assets store exactly what the dashboard shows: same src/ calls,
same config defaults. The warehouse and the model store are faked; the scoring,
ranking and backtest are real."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dagster import build_asset_context
from sklearn.linear_model import LinearRegression

from orchestration import assets
from src.backtest.run import run_strategy_backtest
from src.config import load_config
from src.data import db, warehouse
from src.models import artifact
from src.serving.recommend import latest_recommendations

DASHBOARD = (Path(__file__).resolve().parents[1] / "app" / "lib" / "data.py").read_text()


def _panel(n_dates=4, n_names=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-07-31", periods=n_dates, freq="ME")
    return pd.concat([
        pd.DataFrame({"date": d, "ticker": [f"T{i}" for i in range(n_names)],
                      "gics_sector": rng.choice(["Tech", "Energy", None], n_names),
                      "f0": rng.normal(size=n_names), "f1": rng.normal(size=n_names)})
        for d in dates], ignore_index=True)


@pytest.fixture
def model(monkeypatch):
    """A real fitted estimator behind the artifact store, as the registry would serve it."""
    x = pd.DataFrame({"f0": np.arange(20.0), "f1": np.arange(20.0)[::-1] ** 2})
    est = LinearRegression().fit(x, x["f0"] * 2 - x["f1"] * 0.1)
    man = artifact.Manifest(
        model_version="lightgbm-1m-20261006", model_name="lightgbm", horizon="1m",
        feature_list=["f0", "f1"], hyperparams={}, normalization="rank", target="target",
        train_start="2016-01-31", train_end="2026-09-30", n_train_rows=20)
    monkeypatch.setattr(artifact, "load_artifact", lambda *a, **k: (est, man))
    return man


def test_stored_ranking_is_the_dashboards(monkeypatch, model):
    full = _panel()
    latest = full["date"].max()
    seen = {}

    def read_sql(sql, **params):
        # The asset reads only the newest month -- all that is scored.
        assert "max(date) FROM panel_monthly" in sql
        return full[full["date"] == latest].assign(date=lambda d: d["date"].dt.date)

    monkeypatch.setattr(db, "read_sql", read_sql)
    monkeypatch.setattr(db, "load_recommendations",
                        lambda ranked, v, asof: seen.update(ranked=ranked, v=v, asof=asof) or len(ranked))
    assets.recommendations(build_asset_context())

    # The dashboard's path (app/lib/data.py recommendations): the whole panel in.
    assert "latest_recommendations(horizon=horizon, panel=panel())" in DASHBOARD
    ranked, man, asof = latest_recommendations(horizon="1m", panel=full)
    n = load_config("model")["portfolio"]["n_holdings"]
    assert seen["ranked"]["ticker"].tolist() == ranked["ticker"].tolist()
    assert seen["ranked"]["ticker"].head(n).tolist() == ranked["ticker"].head(n).tolist()
    assert np.allclose(seen["ranked"]["pred"], ranked["pred"])
    assert seen["v"] == man.model_version and seen["asof"] == asof == latest


def _preds(n_dates=30, n_names=60):
    rng = np.random.default_rng(1)
    rows = []
    for d in pd.date_range("2024-01-31", periods=n_dates, freq="ME"):
        pred = rng.normal(size=n_names)
        rows.append(pd.DataFrame({"date": d, "ticker": [f"T{i}" for i in range(n_names)],
                                  "pred": pred, "gics_sector": "Tech",
                                  "forward_return": 0.01 * pred + rng.normal(0, 0.05, n_names)}))
    return pd.concat(rows, ignore_index=True)


def test_stored_backtest_is_the_dashboards_at_its_defaults(monkeypatch):
    preds = _preds()
    idx = pd.DatetimeIndex(sorted(preds["date"].unique())) + pd.offsets.MonthEnd(1)
    ff = pd.DataFrame(np.random.default_rng(2).normal(0, 0.03, (len(idx), 7)), index=idx,
                      columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"])
    seen = {}
    monkeypatch.setattr(warehouse, "latest_model_version", lambda h="1m": "v1")
    monkeypatch.setattr(warehouse, "load_oos_predictions", lambda v, h="1m": preds)
    monkeypatch.setattr(warehouse, "load_ff_factors", lambda: ff)
    monkeypatch.setattr(db, "load_strategy_backtest",
                        lambda v, res, params: seen.update(v=v, res=res, params=params) or 0)
    assets.strategy_backtest(build_asset_context())

    # The Performance page at its default sliders: config n_holdings and costs.
    assert "run_strategy_backtest(preds, warehouse.load_ff_factors()" in DASHBOARD
    cfg = load_config("model")
    n = cfg["portfolio"]["n_holdings"]
    cost = cfg["costs"]["commission_bps"] + cfg["costs"]["spread_bps"]
    dash = run_strategy_backtest(preds, ff, n_holdings=n, cost_bps=float(cost))
    pd.testing.assert_frame_equal(seen["res"].returns, dash.returns)
    assert seen["res"].strategy == dash.strategy
    assert seen["res"].attribution["alpha_annual"] == pytest.approx(dash.attribution["alpha_annual"])
    assert seen["params"]["n_holdings"] == n and seen["params"]["cost_bps"] == cost
    assert seen["params"]["buffer_rank"] == max(cfg["portfolio"]["buffer_rank"], n)
    assert seen["params"]["min_price"] == load_config("data")["universe"]["liquidity_filters"]["min_price"]
