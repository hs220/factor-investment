"""Train cross-sectional ranking models with walk-forward validation.

Schedulable entry point. Reads the gold ``panel_monthly`` table, runs the
walk-forward loop (with nested hyperparameter tuning by default) to estimate
out-of-sample IC, caches the OOS predictions for the backtest stage, then fits a
**deployment model** on all data through today and saves a versioned artifact
(``models/<horizon>/<version>/``) that inference reuses without re-training. The
heavy lifting lives in ``src.models.training.train_and_deploy`` — the same code
the ``model_predictions`` Dagster asset runs.

Requires ``POSTGRES_PASSWORD`` (and ``FACTOR_DB_HOST`` if off-LAN).

Usage:
    python -m pipelines.train                 # model from config (default lightgbm)
    python -m pipelines.train --model elasticnet
    python -m pipelines.train --no-tune       # fixed config params (no inner-CV)
    python -m pipelines.train --no-save-model # skip the deployment artifact
    python -m pipelines.train --shuffle       # leakage test: IC should be ~0
"""
from __future__ import annotations

import argparse

from src.config import load_config
from src.data import cache, warehouse
from src.factors.panel import _feature_list
from src.models.training import train_and_deploy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm", help="elasticnet | lightgbm | xgboost")
    ap.add_argument("--horizon", default="1m", help="prediction horizon label (artifact key)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle target (leakage test)")
    ap.add_argument("--no-tune", action="store_true", help="disable nested-CV tuning")
    ap.add_argument("--no-save-model", action="store_true", help="skip the deployment artifact")
    args = ap.parse_args()

    mcfg = load_config("model")
    tune = mcfg["tuning"].get("enabled", False) and not args.no_tune

    panel = warehouse.load_panel_monthly()
    features = _feature_list()

    if args.shuffle:
        # Destroy any real signal; OOS IC must collapse toward zero.
        panel = panel.copy()
        panel["target"] = (
            panel.groupby("date")["target"].transform(lambda s: s.sample(frac=1).values)
        )

    res = train_and_deploy(
        panel, features, model_name=args.model, horizon=args.horizon,
        tune=tune, save_model=not args.shuffle and not args.no_save_model, cfg=mcfg,
    )
    if res.oos_preds.empty:
        print("No predictions produced (insufficient data for the walk-forward).")
        return

    s = res.summary
    print(f"Model: {args.model}{' [SHUFFLED]' if args.shuffle else ''} | "
          f"tuning: {'on' if tune else 'off'}")
    print(f"  OOS months:   {s['n_months']}")
    print(f"  Mean IC:      {s['ic_mean']:.4f}")
    print(f"  IC IR:        {s['ic_ir']:.3f}")
    print(f"  t-stat:       {s['t_stat']:.2f}")
    print(f"  Hit rate:     {s['hit_rate']:.2%}")
    hist = res.oos_preds.attrs.get("tuning_history")
    if tune and hist:
        print(f"  Re-tuned {len(hist)}x; latest params: {hist[-1]['params']}")

    if args.shuffle:
        return

    cache.save(res.oos_preds[["date", "ticker", "gics_sector", "pred", "forward_return",
                              "market_cap"]], "predictions.parquet")
    print("Saved OOS predictions -> data/processed/predictions.parquet")
    if res.manifest:
        print(f"Saved deployment artifact (version {res.manifest.model_version})")


if __name__ == "__main__":
    main()
