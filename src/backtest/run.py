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
    # Indexed by REALIZATION month-end: the row for month m holds the return over
    # month m of the portfolio formed at the end of month m-1.
    returns: pd.DataFrame        # gross, turnover, cost, net, benchmark
    holdings: pd.DataFrame       # long (date, ticker, weight)
    strategy: dict               # performance_summary of net returns
    benchmark: dict              # performance_summary of the benchmark
    attribution: dict            # factor_attribution of net returns


def run_strategy_backtest(
    predictions: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    *,
    n_holdings: int | None = None,
    buffer_rank: int | None = None,
    cost_bps: float | None = None,
) -> BacktestResult:
    """Backtest the long-only top-N strategy on OOS ``predictions``.

    Only months with a realized forward return are simulated: the newest month's
    scores are live picks, not a backtest period (counting them would add a
    spurious 0% month).
    """
    realized = predictions.groupby("date")["forward_return"].transform("count") > 0
    preds = predictions[realized]

    holdings = build_portfolio(preds, n_holdings=n_holdings, buffer_rank=buffer_rank)
    bt = engine.run_backtest(holdings, preds)
    bt["benchmark"] = engine.benchmark_return(preds)
    if cost_bps is not None:   # override config costs: total bps per side (commission + spread)
        bt["cost"] = bt["turnover"] * 2 * cost_bps / 10_000.0
        bt["net"] = bt["gross"] - bt["cost"]
    # Predictions are keyed by formation date t, but forward_return is realized
    # over (t, t+1]. Re-key to the realization month so the series lines up with
    # calendar factor returns (attribution) and dates the equity curve correctly;
    # left at t, the regression pairs each return with the prior month's factors.
    bt.index = bt.index + pd.offsets.MonthEnd(1)

    attr = (attribution.factor_attribution(bt["net"], factors)
            if factors is not None else {"error": "no factors supplied"})
    return BacktestResult(
        returns=bt,
        holdings=holdings,
        strategy=metrics.performance_summary(bt["net"]),
        benchmark=metrics.performance_summary(bt["benchmark"]),
        attribution=attr,
    )
