"""Model factories for cross-sectional ranking.

We start with a regularized linear baseline (ElasticNet) and a gradient-boosted
tree model (LightGBM), both fit as regressions on the within-sector forward-
return rank target. Tree models capture the factor *interactions* that linear
Fama-French style models cannot (e.g. value only pays off in low-volatility
regimes). A learning-to-rank objective (``lambdarank``) is available as a later
refinement; regression on the rank target is the simple, strong baseline.

Each factory returns a fresh sklearn-style estimator (``fit``/``predict``) so
the walk-forward loop can instantiate one per fold.

Model params in ``config/model.yaml`` may be given as a LIST (a tuning search
candidate) or a SCALAR (fixed). ``default_params`` collapses each list to its
first element; ``param_grid`` expands the lists into the cartesian search grid.
The factories accept an explicit resolved (scalar) param dict; called without
one they fall back to the config defaults.

**Preprocessing is bundled in the estimator.** Every factory returns an sklearn
``Pipeline`` whose first step imputes missing features, so the *same* imputation
is applied at train and inference time and travels inside the saved artifact —
never re-implemented in calling code (see CLAUDE.md "sklearn components for
standard ops; inference-time parity"). Our model features are all numeric
cross-sectional ranks, imputed to the neutral 0.5; if categorical features are
ever added, encode them with ``OneHotEncoder`` inside a ``ColumnTransformer`` in
this same pipeline so encoding is likewise fitted once and reused.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import product

from src.config import load_config

# Missing numeric rank features impute to the neutral cross-sectional rank.
_IMPUTE_FILL = 0.5


def _pipeline(estimator: object) -> object:
    """Wrap an estimator with the shared preprocessing (a neutral-fill imputer).

    Returns an sklearn Pipeline so the imputer is fitted/serialized with the
    model and applied identically at inference. All current features are numeric,
    so a single SimpleImputer suffices; add a ColumnTransformer + OneHotEncoder
    step here if categorical features are introduced.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    # keep_empty_features: an all-NaN column (e.g. fundamentals in the earliest
    # walk-forward folds before any filings exist) is filled with the neutral
    # value and KEPT, not dropped — so the feature set/order is stable across
    # every fold and matches the manifest's feature_list at inference.
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=_IMPUTE_FILL,
                                 keep_empty_features=True)),
        ("model", estimator),
    ])


def default_params(model_params: dict) -> dict:
    """Collapse a config block to scalars: first element of each list param."""
    return {
        k: (v[0] if isinstance(v, list) else v)
        for k, v in model_params.items()
    }


def param_grid(model_params: dict) -> list[dict]:
    """Cartesian grid of scalar param dicts over the list-valued params.

    Scalar params are held constant across the grid; if no param is a list the
    grid is a single (default) point.
    """
    list_keys = [k for k, v in model_params.items() if isinstance(v, list)]
    fixed = {k: v for k, v in model_params.items() if not isinstance(v, list)}
    if not list_keys:
        return [dict(fixed)]
    grid = []
    for combo in product(*[model_params[k] for k in list_keys]):
        grid.append({**fixed, **dict(zip(list_keys, combo))})
    return grid


def _cfg_params(name: str) -> dict:
    return load_config("model")["models"][name]


def make_elasticnet(params: dict | None = None) -> Callable[[], object]:
    from sklearn.linear_model import ElasticNet

    p = params if params is not None else default_params(_cfg_params("elasticnet"))
    return lambda: _pipeline(ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=5000))


def make_lightgbm(params: dict | None = None) -> Callable[[], object]:
    from lightgbm import LGBMRegressor

    src = params if params is not None else default_params(_cfg_params("lightgbm"))
    p = dict(src)
    p.pop("enabled", None)
    p.pop("objective", None)  # regression objective; lambdarank handled separately
    return lambda: _pipeline(LGBMRegressor(
        n_estimators=p.get("n_estimators", 500),
        learning_rate=p.get("learning_rate", 0.02),
        num_leaves=p.get("num_leaves", 31),
        max_depth=p.get("max_depth", -1),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
        min_child_samples=p.get("min_child_samples", 100),
        reg_lambda=p.get("reg_lambda", 1.0),
        verbose=-1,
    ))


def make_xgboost(params: dict | None = None) -> Callable[[], object]:
    from xgboost import XGBRegressor

    src = params if params is not None else default_params(_cfg_params("xgboost"))
    p = dict(src)
    p.pop("enabled", None)
    return lambda: _pipeline(XGBRegressor(
        objective=p.get("objective", "reg:squarederror"),
        n_estimators=p.get("n_estimators", 500),
        learning_rate=p.get("learning_rate", 0.02),
        max_depth=p.get("max_depth", 5),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
    ))


_FACTORIES = {
    "elasticnet": make_elasticnet,
    "lightgbm": make_lightgbm,
    "xgboost": make_xgboost,
}


def get_model_factory(name: str) -> Callable[[], object]:
    """Return a zero-arg factory that builds a fresh estimator for ``name``
    using the config default params."""
    if name not in _FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Options: {list(_FACTORIES)}")
    return _FACTORIES[name]()


def build_model(name: str, params: dict) -> object:
    """Build a fresh, unfitted estimator for ``name`` from explicit scalar params."""
    if name not in _FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Options: {list(_FACTORIES)}")
    return _FACTORIES[name](params)()
