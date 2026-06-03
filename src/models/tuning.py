"""Nested (inner) hyperparameter tuning for the walk-forward harness.

Each *outer* walk-forward fold trains on an expanding history. To pick
hyperparameters without leaking the future, we tune **inside** that training
window: carve the most recent months into inner time-ordered validation folds,
grid-search the candidate params, and keep the combo with the best mean inner
validation IC. The chosen params are then used to fit the outer model.

This keeps tuning strictly causal — the outer test block is never seen during
the search. Tuning is re-run periodically (``tuning.retune_every``) rather than
every fold, which is the standard cost/rigor trade-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors import evaluate
from src.models import rankers
from src.models.walkforward import expanding_splits


def _inner_ic(
    model_name: str,
    params: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    target: str,
) -> float:
    """Mean per-date IC of params, fit on ``train`` and scored on ``val``.

    The model is a pipeline that imputes internally, so raw features are passed.
    """
    if len(train) < 20 or val.empty:
        return np.nan
    model = rankers.build_model(model_name, params)
    model.fit(train[features], train[target])
    scored = val.copy()
    scored["pred"] = model.predict(val[features])
    ic = evaluate.information_coefficient(scored, "pred", target="forward_return")
    return ic.mean()


def tune(
    train_panel: pd.DataFrame,
    model_name: str,
    features: list[str],
    *,
    cfg: dict,
    target: str = "target",
    date_col: str = "date",
) -> tuple[dict, float]:
    """Grid-search hyperparameters on inner splits of ``train_panel``.

    Returns ``(best_params, best_score)``. Falls back to the config default
    params (score ``nan``) when the grid is trivial or the window is too short
    to carve the requested inner validation folds.
    """
    tcfg = cfg["tuning"]
    model_params = cfg["models"][model_name]
    grid = rankers.param_grid(model_params)
    if len(grid) == 1:
        return grid[0], float("nan")

    dates = sorted(pd.unique(train_panel[date_col]))
    k = int(tcfg.get("inner_splits", 3))
    emb = int(tcfg.get("inner_embargo_months", 1))
    inner_min = len(dates) - k - emb
    if inner_min < int(tcfg.get("inner_min_train_months", 24)):
        return rankers.default_params(model_params), float("nan")

    splits = list(
        expanding_splits(dates, min_train_months=inner_min, test_months=1, embargo_months=emb)
    )
    if not splits:
        return rankers.default_params(model_params), float("nan")

    best_params, best_score = None, -np.inf
    for params in grid:
        fold_ics = []
        for tr_dates, va_dates in splits:
            tr = train_panel[train_panel[date_col].isin(tr_dates)].dropna(subset=[target])
            va = train_panel[train_panel[date_col].isin(va_dates)]
            fold_ics.append(
                _inner_ic(model_name, params, tr, va, features, target)
            )
        score = np.nanmean(fold_ics) if any(~np.isnan(fold_ics)) else -np.inf
        if score > best_score:
            best_params, best_score = params, score

    if best_params is None:  # all folds degenerate
        return rankers.default_params(model_params), float("nan")
    return best_params, float(best_score)
