"""Latest investment recommendations from the deployed model.

Shared by ``pipelines/predict.py`` (CLI) and the dashboard's Recommendations
page, so both return identical rows. Scores a panel cross-section with the
deployed artifact — never re-trains.
"""
from __future__ import annotations

import pandas as pd

from src.data import warehouse
from src.models import artifact


def latest_recommendations(
    *,
    horizon: str = "1m",
    version: str = "latest",
    asof: str | pd.Timestamp | None = None,
    panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, artifact.Manifest, pd.Timestamp]:
    """Score the latest (or ``asof``) panel cross-section with the deployed model.

    Returns ``(ranked, manifest, asof)`` — the full cross-section with a ``pred``
    column sorted high→low (the caller slices top-N), the model manifest, and the
    as-of month-end. ``panel`` may be supplied to avoid re-reading the warehouse.
    """
    if panel is None:
        panel = warehouse.load_panel_monthly()
    asof = pd.Timestamp(asof) if asof is not None else panel["date"].max()
    cross = panel[panel["date"] == asof].copy()
    if cross.empty:
        raise ValueError(f"no panel rows for {pd.Timestamp(asof).date()}")

    _, manifest = artifact.load_artifact(horizon, version)
    cross["pred"] = artifact.predict_with_artifact(cross, horizon, version=version)
    ranked = cross.sort_values("pred", ascending=False).reset_index(drop=True)
    return ranked, manifest, asof
