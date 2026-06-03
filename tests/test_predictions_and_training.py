"""Tests for the predictions DB sink and the shared training function."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import db
from src.models.training import train_and_deploy


# --------------------------------------------------------------------------- #
# db.load_predictions — column mapping (no DB)
# --------------------------------------------------------------------------- #
def test_load_predictions_empty():
    assert db.load_predictions(pd.DataFrame(), "1m", "v1") == 0


def test_load_predictions_maps_columns(monkeypatch):
    captured = {}
    monkeypatch.setattr(db, "ensure_predictions_table", lambda: None)
    monkeypatch.setattr(
        db, "upsert",
        lambda df, table, conflict, **k: captured.update(
            df=df, table=table, conflict=conflict) or len(df),
    )
    preds = pd.DataFrame({"date": [pd.Timestamp("2020-01-31")], "ticker": ["AAA"],
                          "pred": [0.7], "extra": [9]})
    n = db.load_predictions(preds, horizon="1m", model_version="lgbm-1m-x")
    assert n == 1
    df = captured["df"]
    assert list(df.columns) == ["date", "ticker", "horizon", "model_version", "score"]
    assert df.iloc[0]["score"] == 0.7
    assert df.iloc[0]["horizon"] == "1m"
    assert df.iloc[0]["model_version"] == "lgbm-1m-x"
    assert captured["table"] == "predictions"
    assert captured["conflict"] == ["date", "ticker", "horizon", "model_version"]


# --------------------------------------------------------------------------- #
# train_and_deploy — walk-forward path (no artifact, no DB)
# --------------------------------------------------------------------------- #
def _toy_panel(n_months=44, n_names=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2015-01-31", periods=n_months, freq="ME"):
        f0 = rng.uniform(0, 1, n_names)
        fwd = 0.05 * f0 + rng.normal(0, 0.02, n_names)
        tgt = pd.Series(fwd).rank(pct=True).values
        for i in range(n_names):
            rows.append({"date": d, "ticker": f"T{i}", "f0": f0[i],
                         "f1": rng.uniform(), "forward_return": fwd[i], "target": tgt[i]})
    return pd.DataFrame(rows)


def test_train_and_deploy_oos_only():
    cfg = {
        "walk_forward": {"min_train_months": 24, "test_months": 1, "embargo_months": 1},
        "tuning": {"enabled": False},
        "models": {"elasticnet": {"alpha": 0.001, "l1_ratio": 0.5}},
    }
    res = train_and_deploy(_toy_panel(), ["f0", "f1"], model_name="elasticnet",
                           tune=False, save_model=False, cfg=cfg)
    assert not res.oos_preds.empty
    assert "pred" in res.oos_preds.columns
    assert set(("ic_mean", "ic_ir", "t_stat", "n_months")) <= set(res.summary)
    assert res.manifest is None        # save_model=False -> no artifact
