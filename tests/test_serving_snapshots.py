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
    """Records each transaction's statements and row writes, in order, instead
    of a database. ``seen["tx"]`` is one list per ``engine.begin()`` block."""
    seen: dict = {"tx": [], "commits": 0}

    def upsert_rows(cur, df, table, conflict, **_):
        cur.tx.append(("rows", table, df.copy(), conflict))
        return len(df)

    class Cursor:
        def __init__(self, tx):
            self.tx = tx

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Conn:
        def __init__(self, tx):
            self.tx = tx
            self.connection = self

        def execute(self, stmt, params=None):
            self.tx.append(("sql", str(stmt), params))

        def cursor(self):
            return Cursor(self.tx)

    class Engine:
        @contextmanager
        def begin(self):
            tx: list = []
            seen["tx"].append(tx)
            yield Conn(tx)
            seen["commits"] += 1

    monkeypatch.setattr(db, "_upsert_rows", upsert_rows)
    monkeypatch.setattr(db, "ensure_serving_tables", lambda: None)
    monkeypatch.setattr(db, "get_engine", lambda: Engine())
    return seen


def test_recommendations_replace_their_slice_in_one_transaction(sink):
    ranked = pd.DataFrame({"ticker": ["B", "A", "C"], "gics_sector": ["X", "Y", None],
                           "pred": [0.9, 0.5, 0.1]})
    n = db.load_recommendations(ranked, "lightgbm-1m-1", pd.Timestamp("2026-10-31"))
    (tx,) = sink["tx"]
    (kind, delete, params), (_, table, df, key) = tx
    # Only this (as_of, model_version) is cleared, then rewritten, in one commit.
    assert "DELETE FROM recommendations WHERE as_of = :a AND model_version = :v" in delete
    assert params == {"a": pd.Timestamp("2026-10-31").date(), "v": "lightgbm-1m-1"}
    assert table == "recommendations" and key == ["as_of", "model_version", "ticker"]
    assert n == 3 and sink["commits"] == 1
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


def test_backtest_summary_and_months_are_one_transaction(sink):
    res = run_strategy_backtest(_preds(), n_holdings=10, buffer_rank=10)
    params = {"n_holdings": 10, "buffer_rank": 10, "cost_bps": 6.0, "min_price": 5.0,
              "max_position_weight": 0.08}
    n = db.load_strategy_backtest("v1", res, params)

    (tx,) = sink["tx"]
    (_, insert, row), (_, delete, _), (_, table, df, key) = tx
    assert sink["commits"] == 1
    assert "INSERT INTO strategy_backtest" in insert and "DELETE FROM strategy_returns" in delete
    assert row["n"] == 10 and row["cost"] == 6.0 and row["maxw"] == 0.08
    assert row["months"] == len(res.returns)
    strategy = json.loads(row["s"])
    assert strategy["ann_return"] == pytest.approx(res.strategy["ann_return"])
    # No factors passed: the attribution is an error message, kept as text.
    assert json.loads(row["a"]) == {"error": "no factors supplied"}
    assert table == "strategy_returns" and key == ["model_version", "month"]
    assert n == len(res.returns) and df["month"].tolist() == list(res.returns.index.date)


def test_jsonable_drops_frames_and_nan():
    out = db._jsonable({"a": np.float64(1.5), "b": float("nan"), "n": 3,
                        "betas": {"MOM": np.float64(0.2)}, "coef_table": pd.DataFrame()})
    assert out == {"a": 1.5, "b": None, "n": 3.0, "betas": {"MOM": 0.2}}
    json.dumps(out)
