"""Container entrypoint for off-box (burst) model training.

Runs inside the ``factor-train`` image on the training box. A thin wrapper over
the same ``src.models.training.train_and_deploy`` that ``pipelines/train.py`` and
the in-process asset call — only the edges differ:

* reads ``panel_monthly`` from the warehouse over the LAN,
* writes the OOS scores to ``predictions`` and the deployment artifact to the
  ``model_registry`` table (store="db"), so nothing depends on this box's disk.

Launched two ways:

* by Dagster via ``PipesDockerClient`` — Pipes context env vars are present, so
  logs and the final materialization (OOS metrics) stream back over stdout;
* by hand (Phase 1) — ``docker run --rm --network host --env-file ~/.factor.env
  factor-train:latest`` — same work, plain stdout logging.

Usage (inside the image): ``python -m orchestration.train_entrypoint
[--model lightgbm] [--horizon 1m] [--no-tune]``
"""
from __future__ import annotations

import argparse
import contextlib
import os
import warnings

from src.config import load_config
from src.data import db, warehouse
from src.factors.panel import _feature_list
from src.models.training import train_and_deploy


class _StdoutPipes:
    """Stand-in for a PipesContext when run outside Dagster."""

    class log:  # noqa: N801 - mirrors PipesContext.log
        info = staticmethod(lambda msg: print(msg, flush=True))

    def report_asset_materialization(self, metadata=None, **_):
        for k, v in (metadata or {}).items():
            print(f"  {k}: {v}", flush=True)


def _pipes():
    from dagster_pipes import DAGSTER_PIPES_CONTEXT_ENV_VAR, open_dagster_pipes

    if os.environ.get(DAGSTER_PIPES_CONTEXT_ENV_VAR):
        return open_dagster_pipes()
    return contextlib.nullcontext(_StdoutPipes())


def main() -> None:
    # The pipeline's imputer hands LightGBM a bare ndarray -> one harmless warning
    # per fold, which would bury the run log streamed into Dagster.
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--horizon", default="1m")
    ap.add_argument("--no-tune", action="store_true", help="disable nested-CV tuning")
    args = ap.parse_args()

    mcfg = load_config("model")
    tune = mcfg["tuning"].get("enabled", False) and not args.no_tune

    with _pipes() as pipes:
        panel = warehouse.load_panel_monthly()
        pipes.log.info(
            f"panel_monthly: {len(panel):,} rows, {panel['date'].min().date()} .. "
            f"{panel['date'].max().date()}; model={args.model} tune={tune}")

        res = train_and_deploy(
            panel, _feature_list(), model_name=args.model, horizon=args.horizon,
            tune=tune, save_model=True, store="db", cfg=mcfg,
        )
        if res.oos_preds.empty or res.manifest is None:
            raise RuntimeError("training produced no predictions / no deployment model")

        version = res.manifest.model_version
        n = db.load_predictions(res.oos_preds, horizon=args.horizon, model_version=version)
        pipes.log.info(f"registered {version} in model_registry; {n:,} prediction rows")
        pipes.report_asset_materialization(metadata={
            "model_version": version,
            "predictions_rows": n,
            "tuned": tune,
            "oos_ic_mean": round(res.summary["ic_mean"], 4),
            "oos_ic_ir": round(res.summary["ic_ir"], 3),
            "oos_t_stat": round(res.summary["t_stat"], 2),
            "oos_months": int(res.summary["n_months"]),
        })


if __name__ == "__main__":
    main()
