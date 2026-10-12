"""Serving snapshots (recommendations, strategy_backtest): the row shapes written,
with the warehouse faked at the upsert/engine seam."""
from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from src.backtest.run import run_strategy_backtest
from src.data import db


@pytest.fixture
def sink(monkeypatch):
    """Captures what would be upserted and executed, instead of a database."""
    seen: dict = {"upserts": {}, "sql": []}

    def upsert(df, table, conflict, **_):
        seen["upserts"][table] = (df.copy(), conflict)
        return len(df)

    class Conn:
        def execute(self, stmt, params=None):
            seen["sql"].append((str(stmt), params))

    class Engine:
        @contextmanager
        def begin(self):
            yield Conn()

    monkeypatch.setattr(db, "upsert", upsert)
    monkeypatch.setattr(db, "ensure_serving_tables", lambda: None)
    monkeypatch.setattr(db, "get_engine", lambda: Engine())
    return seen


def test_recommendations_are_ranked_in_the_order_given(sink):
    ranked = pd.DataFrame({"ticker": ["B", "A", "C"], "gics_sector": ["X", "Y", None],
                           "pred": [0.9, 0.5, 0.1]})
    n = db.load_recommendations(ranked, "lightgbm-1m-1", pd.Timestamp("2026-10-31"))
    df, key = sink["upserts"]["recommendations"]
    assert n == 3 and key == ["as_of", "model_version", "ticker"]
    assert df["ticker"].tolist() == ["B", "A", "C"] and df["rank"].tolist() == [1, 2, 3]
    assert (df["as_of"] == pd.Timestamp("2026-10-31").date()).all()
    assert df["scored_at"].iloc[0].tzinfo is not None


def _preds(n_dates=24, n_names=60):
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2020-01-31", periods=n_dates, freq="ME"):
        pred = rng.normal(size=n_names)
        rows.append(pd.DataFrame({"date": d, "ticker": [f"T{i}" for i in range(n_names)],
                                  "pred": pred,
                                  "forward_return": 0.01 * pred + rng.normal(0, 0.05, n_names)}))
    return pd.concat(rows, ignore_index=True)


def test_backtest_summary_and_months_are_stored_with_their_parameters(sink):
    res = run_strategy_backtest(_preds(), n_holdings=10, buffer_rank=10)
    params = {"n_holdings": 10, "buffer_rank": 10, "cost_bps": 6.0, "min_price": 5.0}
    n = db.load_strategy_backtest("v1", res, params)

    (insert, row), (delete, _) = sink["sql"]
    assert "INSERT INTO strategy_backtest" in insert and "DELETE FROM strategy_returns" in delete
    assert row["n"] == 10 and row["cost"] == 6.0 and row["months"] == len(res.returns)
    strategy = json.loads(row["s"])
    assert strategy["ann_return"] == pytest.approx(res.strategy["ann_return"])
    # No factors passed: the attribution is an error message, kept as text.
    assert json.loads(row["a"]) == {"error": "no factors supplied"}

    df, key = sink["upserts"]["strategy_returns"]
    assert n == len(res.returns) and key == ["model_version", "month"]
    assert df["month"].tolist() == list(res.returns.index.date)


def test_jsonable_drops_frames_and_nan():
    out = db._jsonable({"a": np.float64(1.5), "b": float("nan"), "n": 3,
                        "betas": {"MOM": np.float64(0.2)}, "coef_table": pd.DataFrame()})
    assert out == {"a": 1.5, "b": None, "n": 3.0, "betas": {"MOM": 0.2}}
    json.dumps(out)
