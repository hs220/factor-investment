"""Backtest the long-only strategy and attribute its returns to FF factors.

Schedulable entry point: loads OOS predictions (from pipelines.train), builds
the top-N long-only portfolio, runs the cost-aware backtest, prints performance
vs. the equal-weight universe benchmark, and reports FF5+MOM attribution.

Usage:
    python -m pipelines.train --model elasticnet   # produce predictions first
    python -m pipelines.backtest
"""
from __future__ import annotations

from src.backtest import attribution, engine, metrics
from src.data import cache
from src.portfolio.construct import build_portfolio


def main() -> None:
    preds = cache.load("predictions.parquet")
    factors = cache.load("ff5_monthly.parquet")

    portfolio = build_portfolio(preds)
    bt = engine.run_backtest(portfolio, preds)
    bench = engine.benchmark_return(preds)

    strat = metrics.performance_summary(bt["net"])
    benchk = metrics.performance_summary(bench)

    print("=== Strategy (net of costs) vs equal-weight universe ===")
    print(f"{'metric':<16}{'strategy':>12}{'benchmark':>12}")
    for k in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "hit_rate"]:
        print(f"{k:<16}{strat.get(k, float('nan')):>12.3f}{benchk.get(k, float('nan')):>12.3f}")
    print(f"{'avg turnover':<16}{bt['turnover'].mean():>12.2%}")
    print(f"{'avg cost/mo':<16}{bt['cost'].mean():>12.4%}")

    print("\n=== FF5 + Momentum attribution (strategy net excess) ===")
    attr = attribution.factor_attribution(bt["net"], factors)
    if "error" in attr:
        print("  ", attr["error"])
    else:
        print(f"  Annualized alpha: {attr['alpha_annual']:.2%} "
              f"(t = {attr['alpha_tstat']:.2f})")
        print(f"  R^2: {attr['r_squared']:.2f} | months: {attr['n_months']}")
        print("  Factor betas:")
        for f, b in attr["betas"].items():
            print(f"    {f:<8}{b:>7.3f}  (t={attr['beta_tstats'][f]:.2f})")

    cache.save(bt.reset_index(), "backtest_returns.parquet")
    print("\nSaved -> data/processed/backtest_returns.parquet")


if __name__ == "__main__":
    main()
