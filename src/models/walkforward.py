"""Walk-forward (expanding-window) cross-validation with embargo.

The cardinal rule for time-series ML: never let future data inform the past.
We train on an expanding history, leave an embargo gap, then predict the next
block — stepping forward through time. The embargo prevents the forward-return
target of the last training month from overlapping the test window.

This is the only valid way to estimate out-of-sample IC; random K-fold would
leak future information and produce fantasy backtests.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator

import pandas as pd

from src.config import load_config


def expanding_splits(
    dates: pd.Series | list,
    *,
    min_train_months: int,
    test_months: int,
    embargo_months: int,
) -> Iterator[tuple[list, list]]:
    """Yield (train_dates, test_dates) for an expanding-window walk-forward."""
    uniq = sorted(pd.unique(pd.Series(dates)))
    n = len(uniq)
    start = min_train_months
    while True:
        test_lo = start + embargo_months
        if test_lo >= n:
            break
        test_hi = min(test_lo + test_months, n)
        train_dates = uniq[:start]
        test_dates = uniq[test_lo:test_hi]
        if not test_dates:
            break
        yield train_dates, test_dates
        start += test_months


def walk_forward_predict(
    panel: pd.DataFrame,
    make_model: Callable[[], object] | None = None,
    features: list[str] | None = None,
    *,
    target: str = "target",
    date_col: str = "date",
    min_names_per_fold: int = 20,
    cfg: dict | None = None,
    model_name: str | None = None,
    tune: bool = False,
) -> pd.DataFrame:
    """Run the walk-forward loop and return test rows with a ``pred`` column.

    Missing features are imputed inside the model pipeline (``src.models.rankers``
    bundles a SimpleImputer with each estimator), so the same imputation is used
    at train and inference — calling code passes raw feature columns.

    Two modes:
    - **fixed** (default): pass ``make_model`` (a zero-arg factory); the same
      estimator spec is fit on each expanding window.
    - **tuned** (``tune=True`` + ``model_name``): each fold's hyperparameters are
      chosen by nested inner-CV on the training window (``src.models.tuning``),
      re-tuned every ``cfg["tuning"]["retune_every"]`` folds. The per-fold chosen
      params are recorded on the result's ``.attrs["tuning_history"]``.
    """
    cfg = cfg or load_config("model")
    wf = cfg["walk_forward"]
    splits = expanding_splits(
        panel[date_col],
        min_train_months=wf["min_train_months"],
        test_months=wf["test_months"],
        embargo_months=wf["embargo_months"],
    )

    if tune:
        if model_name is None:
            raise ValueError("tune=True requires model_name")
        from src.models import rankers, tuning  # lazy: avoid import cycle
        retune_every = int(cfg["tuning"].get("retune_every", 1))
    elif make_model is None:
        raise ValueError("provide make_model, or tune=True with model_name")

    out_frames: list[pd.DataFrame] = []
    tuning_history: list[dict] = []
    cur_params: dict | None = None

    for i, (train_dates, test_dates) in enumerate(splits):
        tr = panel[panel[date_col].isin(train_dates)].dropna(subset=[target])
        te = panel[panel[date_col].isin(test_dates)]
        if len(tr) < min_names_per_fold or te.empty:
            continue

        if tune:
            if cur_params is None or i % retune_every == 0:
                cur_params, score = tuning.tune(
                    tr, model_name, features, cfg=cfg,
                    target=target, date_col=date_col,
                )
                tuning_history.append(
                    {"date": str(pd.Timestamp(test_dates[0]).date()),
                     "params": cur_params, "inner_ic": score}
                )
            model = rankers.build_model(model_name, cur_params)
        else:
            model = make_model()

        model.fit(tr[features], tr[target])   # pipeline imputes
        te = te.copy()
        te["pred"] = model.predict(te[features])
        out_frames.append(te)

    if not out_frames:
        return pd.DataFrame()
    result = pd.concat(out_frames, ignore_index=True)
    result.attrs["tuning_history"] = tuning_history
    return result
