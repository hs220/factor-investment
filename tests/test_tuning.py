"""Tests for hyperparameter grid expansion and nested-CV tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import rankers, tuning


def test_param_grid_expands_lists_holds_scalars():
    params = {"n_estimators": [100, 200], "learning_rate": [0.02, 0.05], "max_depth": -1}
    grid = rankers.param_grid(params)
    assert len(grid) == 4                       # 2 x 2
    assert all(g["max_depth"] == -1 for g in grid)   # scalar fixed everywhere
    assert {(g["n_estimators"], g["learning_rate"]) for g in grid} == {
        (100, 0.02), (100, 0.05), (200, 0.02), (200, 0.05)
    }


def test_param_grid_all_scalar_is_single_point():
    assert rankers.param_grid({"alpha": 0.1, "l1_ratio": 0.5}) == [{"alpha": 0.1, "l1_ratio": 0.5}]


def test_default_params_takes_first_of_each_list():
    d = rankers.default_params({"alpha": [0.1, 0.2, 0.3], "fixed": 7})
    assert d == {"alpha": 0.1, "fixed": 7}


def _toy_panel(n_months=40, n_names=60, seed=0):
    """Panel where forward_return depends linearly on feature f0 (signal)."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    for d in dates:
        f0 = rng.uniform(0, 1, n_names)
        f1 = rng.uniform(0, 1, n_names)
        fwd = 0.05 * f0 + rng.normal(0, 0.02, n_names)   # real signal in f0
        tgt = pd.Series(fwd).rank(pct=True).values
        for i in range(n_names):
            rows.append({"date": d, "ticker": f"T{i}", "f0": f0[i], "f1": f1[i],
                         "forward_return": fwd[i], "target": tgt[i]})
    return pd.DataFrame(rows)


def test_tune_returns_a_grid_point_and_score():
    panel = _toy_panel()
    cfg = {
        "models": {"elasticnet": {"alpha": [0.0001, 0.01, 1.0], "l1_ratio": [0.2, 0.8]}},
        "tuning": {"inner_splits": 3, "inner_embargo_months": 1, "inner_min_train_months": 12},
    }
    best, score = tuning.tune(panel, "elasticnet", ["f0", "f1"], cfg=cfg)
    assert set(best) == {"alpha", "l1_ratio"}
    assert best["alpha"] in (0.0001, 0.01, 1.0)
    assert np.isfinite(score)
    # The strongest-regularized alpha=1.0 zeros coefficients -> it should not win
    # outright over the well-fit small alpha on a panel with real signal.
    assert best["alpha"] < 1.0


def test_tune_short_window_falls_back_to_defaults():
    panel = _toy_panel(n_months=8)
    cfg = {
        "models": {"elasticnet": {"alpha": [0.001, 0.01], "l1_ratio": [0.5]}},
        "tuning": {"inner_splits": 3, "inner_embargo_months": 1, "inner_min_train_months": 24},
    }
    best, score = tuning.tune(panel, "elasticnet", ["f0", "f1"], cfg=cfg)
    assert best == {"alpha": 0.001, "l1_ratio": 0.5}    # first-of-each-list default
    assert np.isnan(score)
