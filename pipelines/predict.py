"""Score the latest cross-section with a deployed model (no re-training).

The inference counterpart to ``pipelines.train``: load the most recent month of
the ``panel_monthly`` feature matrix, load the deployment artifact for the given
horizon, assert the live features match the model's manifest (train/serve skew
guard), predict, and emit the ranked recommendations. Reuses the *exact* feature
matrix and ``predict`` path used in training — the model is loaded, never refit.

Requires ``POSTGRES_PASSWORD`` (and ``FACTOR_DB_HOST`` if off-LAN).

Usage:
    python -m pipelines.predict                  # latest month, latest 1m model
    python -m pipelines.predict --horizon 1m --top 30
    python -m pipelines.predict --date 2026-05-31
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_config
from src.data import warehouse
from src.models import artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="1m", help="model horizon to load")
    ap.add_argument("--version", default="latest", help="artifact version (default latest)")
    ap.add_argument("--date", default=None, help="cross-section month-end (default: latest)")
    ap.add_argument("--top", type=int, default=0, help="top-N to show (default: config n_holdings)")
    args = ap.parse_args()

    panel = warehouse.load_panel_monthly()
    asof = pd.Timestamp(args.date) if args.date else panel["date"].max()
    cross = panel[panel["date"] == asof].copy()
    if cross.empty:
        raise SystemExit(f"no panel rows for {asof.date()}")

    model, manifest = artifact.load_artifact(args.horizon, args.version)
    cross["pred"] = artifact.predict_with_artifact(cross, args.horizon, version=args.version)

    top = args.top or load_config("model")["portfolio"]["n_holdings"]
    ranked = cross.sort_values("pred", ascending=False)

    print(f"As-of {asof.date()} | model {manifest.model_version} "
          f"(trained {manifest.train_start}..{manifest.train_end}) | {len(cross)} names")
    print(f"\nTop {top} by predicted rank:")
    cols = ["ticker", "gics_sector", "pred"]
    print(ranked[cols].head(top).to_string(index=False))


if __name__ == "__main__":
    main()
