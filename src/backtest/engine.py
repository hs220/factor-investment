"""Cost-aware backtest engine.

Given a long-only portfolio (date, ticker, weight) and the realized forward
returns from the prediction set, compute gross and net (post-cost) monthly
returns. Costs are modeled as turnover x (commission + half-spread) in bps.
Roth removes taxes, but spread and commissions still erode high-turnover alpha.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.portfolio.construct import turnover as _turnover


def run_backtest(
    portfolio: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    date_col: str = "date",
    fwd_col: str = "forward_return",
) -> pd.DataFrame:
    """Return a DataFrame indexed by date with gross/net returns, turnover, cost."""
    costs = load_config("model")["costs"]
    bps = (costs["commission_bps"] + costs["spread_bps"]) / 10_000.0

    merged = portfolio.merge(
        predictions[[date_col, "ticker", fwd_col]], on=[date_col, "ticker"], how="left"
    )
    gross = (
        merged.assign(contrib=merged["weight"] * merged[fwd_col].fillna(0))
        .groupby(date_col)["contrib"]
        .sum()
        .rename("gross")
    )
    tov = _turnover(portfolio, date_col=date_col)
    cost = (tov * 2 * bps).rename("cost")  # round-trip: enter + exit
    out = pd.concat([gross, tov, cost], axis=1).sort_index()
    out["net"] = out["gross"] - out["cost"]
    return out


def benchmark_return(
    predictions: pd.DataFrame, *, date_col: str = "date", fwd_col: str = "forward_return"
) -> pd.Series:
    """Equal-weight universe return per date — the long-only benchmark to beat."""
    return predictions.groupby(date_col)[fwd_col].mean().rename("benchmark")
