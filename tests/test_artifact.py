"""Tests for model artifact save/load + the train/serve feature-list guard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.models import artifact


def _fit_toy():
    x = pd.DataFrame({"f0": np.arange(20.0), "f1": np.arange(20.0)[::-1]})
    y = x["f0"] * 2 + 1
    return LinearRegression().fit(x, y)


def _manifest(version="m-1m-20260603000000"):
    return artifact.Manifest(
        model_version=version, model_name="m", horizon="1m",
        feature_list=["f0", "f1"], hyperparams={"a": 1}, normalization="rank",
        target="target", train_start="2009-01-31", train_end="2026-05-31",
        n_train_rows=20, oos_metrics={"ic_mean": 0.03}, code_sha="abc1234",
        created_at="2026-06-03T00:00:00+00:00",
    )


def test_save_load_roundtrip(tmp_path):
    model = _fit_toy()
    man = _manifest()
    dest = artifact.save_artifact(model, man, models_dir=tmp_path)
    assert (dest / "model.joblib").exists() and (dest / "manifest.json").exists()
    assert (tmp_path / "1m" / "latest.txt").read_text().strip() == man.model_version

    loaded, man2 = artifact.load_artifact("1m", models_dir=tmp_path)
    assert man2.feature_list == ["f0", "f1"]
    assert man2.oos_metrics["ic_mean"] == 0.03
    x = pd.DataFrame({"f0": [1.0], "f1": [0.0]})
    assert np.allclose(loaded.predict(x), model.predict(x))


def test_latest_pointer_tracks_newest(tmp_path):
    artifact.save_artifact(_fit_toy(), _manifest("m-1m-1"), models_dir=tmp_path)
    artifact.save_artifact(_fit_toy(), _manifest("m-1m-2"), models_dir=tmp_path)
    assert (tmp_path / "1m" / "latest.txt").read_text().strip() == "m-1m-2"


def test_predict_with_artifact_orders_columns(tmp_path):
    artifact.save_artifact(_fit_toy(), _manifest(), models_dir=tmp_path)
    # supply columns in the wrong order + an extra column
    live = pd.DataFrame({"f1": [0.0, 1.0], "extra": [9, 9], "f0": [1.0, 2.0]})
    preds = artifact.predict_with_artifact(live, "1m", models_dir=tmp_path)
    assert list(preds.index) == [0, 1] and preds.notna().all()


def test_predict_with_artifact_skew_guard(tmp_path):
    artifact.save_artifact(_fit_toy(), _manifest(), models_dir=tmp_path)
    live = pd.DataFrame({"f0": [1.0]})        # missing f1
    with pytest.raises(ValueError, match="skew"):
        artifact.predict_with_artifact(live, "1m", models_dir=tmp_path)


def test_load_missing_horizon_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifact.load_artifact("99d", models_dir=tmp_path)
