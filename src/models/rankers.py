"""Model factories for cross-sectional ranking.

We start with a regularized linear baseline (ElasticNet) and a gradient-boosted
tree model (LightGBM), both fit as regressions on the within-sector forward-
return rank target. Tree models capture the factor *interactions* that linear
Fama-French style models cannot (e.g. value only pays off in low-volatility
regimes). A learning-to-rank objective (``lambdarank``) is available as a later
refinement; regression on the rank target is the simple, strong baseline.

Each factory returns a fresh sklearn-style estimator (``fit``/``predict``) so
the walk-forward loop can instantiate one per fold.
"""
from __future__ import annotations

from collections.abc import Callable

from src.config import load_config


def make_elasticnet(params: dict | None = None) -> Callable[[], object]:
    from sklearn.linear_model import ElasticNet

    p = params or load_config("model")["models"]["elasticnet"]
    return lambda: ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=5000)


def make_lightgbm(params: dict | None = None) -> Callable[[], object]:
    from lightgbm import LGBMRegressor

    p = dict(params or load_config("model")["models"]["lightgbm"])
    p.pop("enabled", None)
    p.pop("objective", None)  # regression objective; lambdarank handled separately
    return lambda: LGBMRegressor(
        n_estimators=p.get("n_estimators", 500),
        learning_rate=p.get("learning_rate", 0.02),
        num_leaves=p.get("num_leaves", 31),
        max_depth=p.get("max_depth", -1),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
        min_child_samples=p.get("min_child_samples", 100),
        reg_lambda=p.get("reg_lambda", 1.0),
        verbose=-1,
    )


def make_xgboost(params: dict | None = None) -> Callable[[], object]:
    from xgboost import XGBRegressor

    p = dict(params or load_config("model")["models"]["xgboost"])
    p.pop("enabled", None)
    return lambda: XGBRegressor(
        objective=p.get("objective", "reg:squarederror"),
        n_estimators=p.get("n_estimators", 500),
        learning_rate=p.get("learning_rate", 0.02),
        max_depth=p.get("max_depth", 5),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
    )


_FACTORIES = {
    "elasticnet": make_elasticnet,
    "lightgbm": make_lightgbm,
    "xgboost": make_xgboost,
}


def get_model_factory(name: str) -> Callable[[], object]:
    """Return a zero-arg factory that builds a fresh estimator for ``name``."""
    if name not in _FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Options: {list(_FACTORIES)}")
    return _FACTORIES[name]()
