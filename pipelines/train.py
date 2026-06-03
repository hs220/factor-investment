"""Train cross-sectional ranking models with walk-forward validation.

Schedulable entry point. Reads the gold ``panel_monthly`` table, runs the
walk-forward loop (with nested hyperparameter tuning by default) to estimate
out-of-sample IC, caches the OOS predictions for the backtest stage, then fits a
**deployment model** on all data through today and saves a versioned artifact
(``models/<horizon>/<version>/``) that inference reuses without re-training.

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
from datetime import datetime, timezone

from src.config import load_config
from src.data import cache, warehouse
from src.factors import evaluate
from src.factors.panel import _feature_list
from src.models import artifact, rankers, tuning, walkforward


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

    # --- Walk-forward OOS estimate ------------------------------------------
    if tune:
        preds = walkforward.walk_forward_predict(
            panel, features=features, model_name=args.model, tune=True, cfg=mcfg
        )
    else:
        factory = rankers.get_model_factory(args.model)
        preds = walkforward.walk_forward_predict(panel, factory, features, cfg=mcfg)

    if preds.empty:
        print("No predictions produced (insufficient data for the walk-forward).")
        return

    ic = evaluate.information_coefficient(preds, "pred", target="forward_return")
    summary = evaluate.ic_summary(ic)
    print(f"Model: {args.model}{' [SHUFFLED]' if args.shuffle else ''} | "
          f"tuning: {'on' if tune else 'off'}")
    print(f"  OOS months:   {summary['n_months']}")
    print(f"  Mean IC:      {summary['ic_mean']:.4f}")
    print(f"  IC IR:        {summary['ic_ir']:.3f}")
    print(f"  t-stat:       {summary['t_stat']:.2f}")
    print(f"  Hit rate:     {summary['hit_rate']:.2%}")
    if tune and preds.attrs.get("tuning_history"):
        last = preds.attrs["tuning_history"][-1]
        print(f"  Re-tuned {len(preds.attrs['tuning_history'])}x; "
              f"latest params: {last['params']}")

    if args.shuffle:
        return

    cache.save(preds[["date", "ticker", "gics_sector", "pred", "forward_return",
                      "market_cap"]], "predictions.parquet")
    print("Saved OOS predictions -> data/processed/predictions.parquet")

    if args.no_save_model:
        return

    # --- Deployment model: fit once on all data, persist for inference -------
    fit_df = panel.dropna(subset=["target"])
    if tune:
        dep_params, inner_ic = tuning.tune(fit_df, args.model, features, cfg=mcfg)
        print(f"Deployment tuning: inner IC {inner_ic:.4f} | params {dep_params}")
    else:
        dep_params = rankers.default_params(mcfg["models"][args.model])

    model = rankers.build_model(args.model, dep_params)
    model.fit(fit_df[features], fit_df["target"])   # pipeline imputes

    manifest = artifact.Manifest(
        model_version=artifact.make_version(args.model, args.horizon),
        model_name=args.model,
        horizon=args.horizon,
        feature_list=features,
        hyperparams=dep_params,
        normalization=load_config("features")["normalization"]["scheme"],
        target="target",
        train_start=str(fit_df["date"].min().date()),
        train_end=str(fit_df["date"].max().date()),
        n_train_rows=int(len(fit_df)),
        oos_metrics={k: float(summary[k]) for k in ("ic_mean", "ic_ir", "t_stat", "hit_rate")},
        code_sha=artifact._git_sha(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    dest = artifact.save_artifact(model, manifest)
    print(f"Saved deployment artifact -> {dest} (version {manifest.model_version})")


if __name__ == "__main__":
    main()
