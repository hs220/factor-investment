"""One training code path for the pipeline and the Dagster asset.

``train_and_deploy`` runs the walk-forward OOS estimate and (optionally) fits a
deployment model on all data, persisting a versioned artifact. Both
``pipelines/train.py`` and the ``model_predictions`` Dagster asset call this, so
the OOS evaluation, the deployed model, and the saved artifact are produced by
identical logic regardless of how training is launched.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from src.config import load_config
from src.factors import evaluate
from src.models import artifact, rankers, tuning, walkforward


@dataclass
class TrainResult:
    oos_preds: pd.DataFrame          # walk-forward OOS rows (incl. ``pred``)
    summary: dict                    # ic_summary of the OOS predictions
    manifest: artifact.Manifest | None   # deployment artifact manifest (if saved)


def train_and_deploy(
    panel: pd.DataFrame,
    features: list[str],
    *,
    model_name: str = "lightgbm",
    horizon: str = "1m",
    tune: bool = False,
    save_model: bool = True,
    cfg: dict | None = None,
) -> TrainResult:
    """Walk-forward OOS estimate, then fit + persist a deployment model.

    Returns the OOS predictions, their IC summary, and the artifact manifest
    (``None`` when ``save_model`` is false or no predictions were produced).
    """
    cfg = cfg or load_config("model")

    if tune:
        oos = walkforward.walk_forward_predict(
            panel, features=features, model_name=model_name, tune=True, cfg=cfg)
    else:
        oos = walkforward.walk_forward_predict(
            panel, rankers.get_model_factory(model_name), features, cfg=cfg)

    if oos.empty:
        return TrainResult(oos, {}, None)

    summary = evaluate.ic_summary(
        evaluate.information_coefficient(oos, "pred", target="forward_return"))

    manifest = None
    if save_model:
        fit_df = panel.dropna(subset=["target"])
        if tune:
            dep_params, _ = tuning.tune(fit_df, model_name, features, cfg=cfg)
        else:
            dep_params = rankers.default_params(cfg["models"][model_name])
        model = rankers.build_model(model_name, dep_params)
        model.fit(fit_df[features], fit_df["target"])     # pipeline imputes

        manifest = artifact.Manifest(
            model_version=artifact.make_version(model_name, horizon),
            model_name=model_name, horizon=horizon, feature_list=list(features),
            hyperparams=dep_params,
            normalization=load_config("features")["normalization"]["scheme"],
            target="target",
            train_start=str(fit_df["date"].min().date()),
            train_end=str(fit_df["date"].max().date()),
            n_train_rows=int(len(fit_df)),
            oos_metrics={k: float(summary[k])
                         for k in ("ic_mean", "ic_ir", "t_stat", "hit_rate")},
            code_sha=artifact._git_sha(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        artifact.save_artifact(model, manifest)

    return TrainResult(oos, summary, manifest)
