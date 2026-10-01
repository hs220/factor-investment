"""Backtest the long-only strategy and attribute its returns to FF factors.

Schedulable entry point: loads the newest model's OOS predictions from the
warehouse (``predictions`` joined to realized forward returns), builds the top-N
long-only portfolio, runs the cost-aware backtest, prints performance vs. the
equal-weight universe benchmark, and reports FF5+MOM attribution. Same
``src.backtest.run`` the dashboard's Performance page calls.

Requires ``POSTGRES_PASSWORD`` (and ``FACTOR_DB_HOST`` if off-LAN).

Usage:
    python -m pipelines.backtest                         # newest 1m model
    python -m pipelines.backtest --model-version lightgbm-1m-20261001171848
"""
from __future__ import annotations

import argparse

from src.backtest.run import run_strategy_backtest
from src.data import warehouse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="1m")
    ap.add_argument("--model-version", default=None, help="default: newest registered")
    ap.add_argument("--top", type=int, default=None, help="holdings (default: config)")
    args = ap.parse_args()

    version = args.model_version or warehouse.latest_model_version(args.horizon)
    if version is None:
        raise SystemExit("no model in model_registry — run model_train first")
    preds = warehouse.load_oos_predictions(version, args.horizon)
    res = run_strategy_backtest(preds, warehouse.load_ff_factors(), n_holdings=args.top)
    bt = res.returns

    print(f"Model {version} | {bt.index.min().date()} .. {bt.index.max().date()} "
          f"({len(bt)} months)")
    print("=== Strategy (net of costs) vs equal-weight universe ===")
    print(f"{'metric':<16}{'strategy':>12}{'benchmark':>12}")
    for k in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "hit_rate"]:
        print(f"{k:<16}{res.strategy.get(k, float('nan')):>12.3f}"
              f"{res.benchmark.get(k, float('nan')):>12.3f}")
    print(f"{'avg turnover':<16}{bt['turnover'].mean():>12.2%}")
    print(f"{'avg cost/mo':<16}{bt['cost'].mean():>12.4%}")

    print("\n=== FF5 + Momentum attribution (strategy net excess) ===")
    attr = res.attribution
    if "error" in attr:
        print("  ", attr["error"])
    else:
        print(f"  Annualized alpha: {attr['alpha_annual']:.2%} "
              f"(t = {attr['alpha_tstat']:.2f})")
        print(f"  R^2: {attr['r_squared']:.2f} | months: {attr['n_months']}")
        print("  Factor betas:")
        for f, b in attr["betas"].items():
            print(f"    {f:<8}{b:>7.3f}  (t={attr['beta_tstats'][f]:.2f})")


if __name__ == "__main__":
    main()
