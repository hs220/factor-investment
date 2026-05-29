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
    make_model: Callable[[], object],
    features: list[str],
    *,
    target: str = "target",
    date_col: str = "date",
    min_names_per_fold: int = 20,
    impute: float = 0.5,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Run the walk-forward loop and return test rows with a ``pred`` column.

    Features are rank-normalized in [0,1]; missing values are imputed to a
    neutral ``impute`` (0.5) so linear models work and tree models stay robust.
    """
    wf = (cfg or load_config("model"))["walk_forward"]
    splits = expanding_splits(
        panel[date_col],
        min_train_months=wf["min_train_months"],
        test_months=wf["test_months"],
        embargo_months=wf["embargo_months"],
    )

    out_frames = []
    for train_dates, test_dates in splits:
        tr = panel[panel[date_col].isin(train_dates)].dropna(subset=[target])
        te = panel[panel[date_col].isin(test_dates)]
        if len(tr) < min_names_per_fold or te.empty:
            continue

        x_tr = tr[features].fillna(impute)
        x_te = te[features].fillna(impute)
        model = make_model()
        model.fit(x_tr, tr[target])

        te = te.copy()
        te["pred"] = model.predict(x_te)
        out_frames.append(te)

    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)
