"""Train cross-sectional ranking models with walk-forward validation.

Schedulable entry point: assembles the panel, runs the walk-forward loop for the
configured model, reports out-of-sample IC, and caches the OOS predictions for
the backtest stage.

Usage:
    python -m pipelines.train                 # model from config (default lightgbm)
    python -m pipelines.train --model elasticnet
    python -m pipelines.train --shuffle       # leakage test: IC should be ~0
"""
from __future__ import annotations

import argparse

import numpy as np

from src.config import load_config
from src.data import cache
from src.factors import evaluate
from src.factors.panel import _feature_list
from src.models import rankers, walkforward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm", help="elasticnet | lightgbm | xgboost")
    ap.add_argument("--shuffle", action="store_true", help="shuffle target (leakage test)")
    args = ap.parse_args()

    panel = cache.load("panel.parquet")
    features = _feature_list()

    if args.shuffle:
        # Destroy any real signal; OOS IC must collapse toward zero.
        panel = panel.copy()
        panel["target"] = (
            panel.groupby("date")["target"].transform(lambda s: s.sample(frac=1).values)
        )

    factory = rankers.get_model_factory(args.model)
    preds = walkforward.walk_forward_predict(panel, factory, features)
    if preds.empty:
        print("No predictions produced (insufficient data for the walk-forward).")
        return

    ic = evaluate.information_coefficient(preds, "pred", target="forward_return")
    summary = evaluate.ic_summary(ic)
    print(f"Model: {args.model}{' [SHUFFLED]' if args.shuffle else ''}")
    print(f"  OOS months:   {summary['n_months']}")
    print(f"  Mean IC:      {summary['ic_mean']:.4f}")
    print(f"  IC IR:        {summary['ic_ir']:.3f}")
    print(f"  t-stat:       {summary['t_stat']:.2f}")
    print(f"  Hit rate:     {summary['hit_rate']:.2%}")

    if not args.shuffle:
        cache.save(preds[["date", "ticker", "gics_sector", "pred", "forward_return",
                          "market_cap"]], "predictions.parquet")
        print("Saved OOS predictions -> data/processed/predictions.parquet")


if __name__ == "__main__":
    main()
