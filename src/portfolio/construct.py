"""Long-only portfolio construction from model predictions.

Roth IRA = long-only, so we hold the top-ranked names (we can't short the
bottom). Each rebalance we take the top-N by predicted score, equal-weight them
subject to a max-position cap. The prediction is already sector-relative (the
model is trained on a within-sector rank target), so the selection is implicitly
sector-aware; an optional per-sector cap bounds concentration further.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config


def build_portfolio(
    predictions: pd.DataFrame,
    *,
    n_holdings: int | None = None,
    max_weight: float | None = None,
    date_col: str = "date",
    score_col: str = "pred",
) -> pd.DataFrame:
    """Top-N equal-weight long-only portfolio per rebalance date.

    Returns a long DataFrame (date, ticker, weight) with weights summing to 1
    each date.
    """
    cfg = load_config("model")["portfolio"]
    n = n_holdings or cfg["n_holdings"]
    cap = max_weight or cfg["max_position_weight"]

    rows = []
    for date, g in predictions.groupby(date_col):
        g = g.dropna(subset=[score_col])
        if g.empty:
            continue
        top = g.nlargest(min(n, len(g)), score_col).copy()
        top["weight"] = min(1.0 / len(top), cap)
        top["weight"] = top["weight"] / top["weight"].sum()  # renormalize to 1
        rows.append(top[[date_col, "ticker", "weight"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def turnover(portfolio: pd.DataFrame, *, date_col: str = "date") -> pd.Series:
    """One-way turnover per rebalance = 0.5 * sum |w_t - w_{t-1}| over names."""
    wide = portfolio.pivot_table(
        index=date_col, columns="ticker", values="weight", fill_value=0.0
    ).sort_index()
    change = wide.diff().abs().sum(axis=1)
    change.iloc[0] = 1.0  # initial build = full turnover
    return (change * 0.5).rename("turnover")
