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


def select_with_buffer(
    ranked: list[str], held: set[str] | list[str], n: int, buffer_rank: int
) -> tuple[list[str], list[str], list[str]]:
    """The hold-buffer selection rule shared by the backtest and live rebalancing.

    ``ranked`` is best-first. Holdings ranked within ``buffer_rank`` are kept
    (best first, at most ``n``); remaining slots are filled with the best-ranked
    names not held. Returns ``(keep, buy, sell)``; held names absent from
    ``ranked`` are not the strategy's to judge and appear in none of the lists.
    """
    rank = {t: i for i, t in enumerate(ranked, start=1)}
    held = [t for t in held if t in rank]
    keep = sorted((t for t in held if rank[t] <= buffer_rank), key=rank.get)[:n]
    keep_set = set(keep)
    buy = [t for t in ranked if t not in keep_set and t not in held][: n - len(keep)]
    sell = sorted((t for t in held if t not in keep_set), key=rank.get)
    return keep, buy, sell


def build_portfolio(
    predictions: pd.DataFrame,
    *,
    n_holdings: int | None = None,
    buffer_rank: int | None = None,
    max_weight: float | None = None,
    date_col: str = "date",
    score_col: str = "pred",
) -> pd.DataFrame:
    """Equal-weight long-only portfolio per rebalance date, with a hold-buffer.

    Path-dependent: each date keeps the prior holdings still ranked within
    ``buffer_rank`` and fills the rest from the top (``select_with_buffer`` — the
    same rule live rebalancing uses). Returns a long DataFrame
    (date, ticker, weight) with weights summing to 1 each date.
    """
    cfg = load_config("model")["portfolio"]
    n = n_holdings or cfg["n_holdings"]
    buf = max(buffer_rank or cfg.get("buffer_rank", n), n)
    cap = max_weight or cfg["max_position_weight"]

    rows, held = [], []
    for date, g in predictions.groupby(date_col):
        g = g.dropna(subset=[score_col])
        if g.empty:
            continue
        ranked = g.sort_values(score_col, ascending=False)["ticker"].tolist()
        keep, buy, _ = select_with_buffer(ranked, held, n, buf)
        held = keep + buy
        w = min(1.0 / len(held), cap)
        port = pd.DataFrame({date_col: date, "ticker": held, "weight": w})
        port["weight"] = port["weight"] / port["weight"].sum()   # renormalize to 1
        rows.append(port)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def turnover(portfolio: pd.DataFrame, *, date_col: str = "date") -> pd.Series:
    """One-way turnover per rebalance = 0.5 * sum |w_t - w_{t-1}| over names."""
    wide = portfolio.pivot_table(
        index=date_col, columns="ticker", values="weight", fill_value=0.0
    ).sort_index()
    change = wide.diff().abs().sum(axis=1)
    change.iloc[0] = 1.0  # initial build = full turnover
    return (change * 0.5).rename("turnover")
