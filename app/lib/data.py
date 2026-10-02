"""Cached data loaders for the dashboard — thin wrappers over ``src/``.

All real logic lives in ``src/`` (warehouse readers, recommendations, IC
evaluation); Streamlit only caches the reads so interactions don't re-query the
warehouse or re-score the model. Pages import these helpers and render.

The deployed model is read from ``$FACTOR_MODEL_STORE`` — set ``db`` to serve the
artifact the training box registered in ``model_registry`` (default: local fs).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable regardless of where Streamlit launches from.
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from src.config import load_config
from src.data import db, warehouse
from src.factors import evaluate
from src.factors.panel import _feature_list
from src.serving.recommend import latest_recommendations
from src.backtest.run import BacktestResult, run_strategy_backtest

DAGSTER_URL = "http://192.168.68.70:3030"
_MACRO = ("yield_curve", "vix", "credit_spread")


@st.cache_data(ttl=600)
def ping() -> bool:
    return db.ping()


def n_holdings() -> int:
    return int(load_config("model")["portfolio"]["n_holdings"])


@st.cache_data(ttl=3600, show_spinner="Loading feature panel…")
def panel() -> pd.DataFrame:
    return warehouse.load_panel_monthly()


@st.cache_data(ttl=3600, show_spinner="Scoring latest cross-section…")
def recommendations(horizon: str = "1m"):
    """(ranked cross-section, manifest, asof) from the deployed model."""
    return latest_recommendations(horizon=horizon, panel=panel())


@st.cache_data(ttl=3600, show_spinner="Computing single-signal IC…")
def signal_report() -> pd.DataFrame:
    feats = [f for f in _feature_list() if f not in _MACRO]   # stock-level only
    return evaluate.signal_report(panel(), feats)


@st.cache_data(ttl=3600)
def feature_coverage(since: str = "2015-01-01") -> pd.Series:
    p = panel()
    recent = p[p["date"] >= since]
    return (recent[_feature_list()].notna().mean() * 100).round(0).sort_values()


@st.cache_data(ttl=3600, show_spinner="Running backtest on OOS predictions…")
def backtest(n_holdings: int | None = None, horizon: str = "1m") -> tuple[str, BacktestResult] | None:
    """(model_version, result) for the newest registered model; None if no model yet."""
    version = warehouse.latest_model_version(horizon)
    if version is None:
        return None
    preds = warehouse.load_oos_predictions(version, horizon)
    return version, run_strategy_backtest(preds, warehouse.load_ff_factors(),
                                          n_holdings=n_holdings)


@st.cache_data(ttl=3600, show_spinner="Loading latest prices…")
def latest_closes() -> pd.Series:
    return warehouse.latest_closes()


def holdings():
    """(positions, cash, as_of) of the newest saved snapshot (uncached: user-edited)."""
    from src.portfolio.holdings import load_latest_snapshot
    return load_latest_snapshot()
