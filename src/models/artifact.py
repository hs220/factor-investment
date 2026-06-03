"""Persist trained models for inference (versioned artifact + manifest).

Training fits one model per walk-forward fold to *estimate* OOS skill; live
inference must not re-train — it loads a single **deployment artifact** fit once
on all data through ``as_of`` and reuses it every rebalance. An artifact is:

    models/<horizon>/<model_version>/
        model.joblib     # the fitted estimator (joblib; uniform across model types)
        manifest.json     # metadata + the train/serve contract

The manifest's ``feature_list`` is the explicit guard against training-serving
skew: ``predict_with_artifact`` asserts the live feature columns match it before
scoring. ``models/<horizon>/latest.txt`` points at the newest version.

joblib is used over raw pickle (efficient for numpy, the sklearn convention) in a
controlled, dependency-pinned runtime; the JSON manifest is the durable,
human-readable record of how the model was built.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODELS_DIR = Path("models")


@dataclass
class Manifest:
    """Everything needed to reproduce/serve a deployment model."""

    model_version: str
    model_name: str
    horizon: str
    feature_list: list[str]
    hyperparams: dict
    normalization: str
    target: str
    train_start: str
    train_end: str
    n_train_rows: int
    oos_metrics: dict = field(default_factory=dict)
    code_sha: str = ""
    created_at: str = ""


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def make_version(model_name: str, horizon: str, created: datetime | None = None) -> str:
    created = created or datetime.now(timezone.utc)
    return f"{model_name}-{horizon}-{created:%Y%m%d%H%M%S}"


def save_artifact(
    model: object,
    manifest: Manifest,
    *,
    models_dir: Path | str = MODELS_DIR,
) -> Path:
    """Write the estimator + manifest under models/<horizon>/<version>/ and bump
    the ``latest`` pointer. Returns the artifact directory."""
    import joblib

    root = Path(models_dir) / manifest.horizon
    dest = root / manifest.model_version
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest / "model.joblib")
    (dest / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    (root / "latest.txt").write_text(manifest.model_version)
    return dest


def _resolve_version(horizon: str, version: str, models_dir: Path | str) -> Path:
    root = Path(models_dir) / horizon
    if version == "latest":
        ptr = root / "latest.txt"
        if not ptr.exists():
            raise FileNotFoundError(f"no deployed model for horizon {horizon!r} in {root}")
        version = ptr.read_text().strip()
    dest = root / version
    if not dest.exists():
        raise FileNotFoundError(f"artifact not found: {dest}")
    return dest


def load_artifact(
    horizon: str,
    version: str = "latest",
    *,
    models_dir: Path | str = MODELS_DIR,
) -> tuple[object, Manifest]:
    """Load (estimator, manifest) for a horizon (default: the latest version)."""
    import joblib

    dest = _resolve_version(horizon, version, models_dir)
    model = joblib.load(dest / "model.joblib")
    manifest = Manifest(**json.loads((dest / "manifest.json").read_text()))
    return model, manifest


def predict_with_artifact(
    features_df: pd.DataFrame,
    horizon: str,
    *,
    version: str = "latest",
    models_dir: Path | str = MODELS_DIR,
) -> pd.Series:
    """Score a feature cross-section with a deployed model.

    Asserts the supplied feature columns cover the manifest's ``feature_list``
    (the training-serving skew guard) and scores in the manifest's column order.
    The model is a pipeline that imputes internally, so raw features are passed —
    the same imputation as training, by construction.
    """
    model, manifest = load_artifact(horizon, version, models_dir=models_dir)
    missing = [c for c in manifest.feature_list if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"live features missing manifest columns (train/serve skew): {missing}"
        )
    x = features_df[manifest.feature_list]
    return pd.Series(model.predict(x), index=features_df.index, name="pred")
