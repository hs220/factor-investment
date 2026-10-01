"""One backtest entry point for the pipeline, notebook 05, and the dashboard.

OOS predictions -> long-only top-N portfolio -> cost-aware monthly returns vs the
equal-weight universe -> performance summary + FF5/MOM attribution. Callers pass
the predictions + factors (warehouse or cache) and only render the result.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest import attribution, engine, metrics
from src.portfolio.construct import build_portfolio


@dataclass
class BacktestResult:
    returns: pd.DataFrame        # date-indexed: gross, turnover, cost, net, benchmark
    holdings: pd.DataFrame       # long (date, ticker, weight)
    strategy: dict               # performance_summary of net returns
    benchmark: dict              # performance_summary of the benchmark
    attribution: dict            # factor_attribution of net returns


def run_strategy_backtest(
    predictions: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    *,
    n_holdings: int | None = None,
) -> BacktestResult:
    """Backtest the long-only top-N strategy on OOS ``predictions``.

    Only months with a realized forward return are simulated: the newest month's
    scores are live picks, not a backtest period (counting them would add a
    spurious 0% month).
    """
    realized = predictions.groupby("date")["forward_return"].transform("count") > 0
    preds = predictions[realized]

    holdings = build_portfolio(preds, n_holdings=n_holdings)
    bt = engine.run_backtest(holdings, preds)
    bt["benchmark"] = engine.benchmark_return(preds)

    attr = (attribution.factor_attribution(bt["net"], factors)
            if factors is not None else {"error": "no factors supplied"})
    return BacktestResult(
        returns=bt,
        holdings=holdings,
        strategy=metrics.performance_summary(bt["net"]),
        benchmark=metrics.performance_summary(bt["benchmark"]),
        attribution=attr,
    )
