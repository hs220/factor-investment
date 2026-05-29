"""Cross-sectional feature normalization.

Raw feature *levels* (P/E, ROE, momentum) drift over time and across regimes,
so we never feed them to a model directly. Instead we normalize each feature
*within each date* (and optionally within sector), which:

- makes features stationary and comparable across time,
- expresses "where does this stock stand vs. peers right now", which is exactly
  what a cross-sectional ranking model needs,
- neutralizes sector tilts when ``sector_neutral=True`` so bets reflect
  stock selection rather than an implicit macro sector bet.

All functions operate on a long panel with a ``date`` column (and ``sector``
when neutralizing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _winsorize(s: pd.Series, pct: float) -> pd.Series:
    if pct <= 0:
        return s
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def cross_sectional_rank(
    panel: pd.DataFrame,
    cols: list[str],
    *,
    date_col: str = "date",
    sector_col: str | None = None,
) -> pd.DataFrame:
    """Rank each feature within date (or date x sector) to [0, 1].

    Returns a copy with ``<col>`` replaced by its cross-sectional rank.
    """
    out = panel.copy()
    group = [date_col] + ([sector_col] if sector_col else [])
    for c in cols:
        out[c] = out.groupby(group)[c].rank(pct=True)
    return out


def cross_sectional_zscore(
    panel: pd.DataFrame,
    cols: list[str],
    *,
    date_col: str = "date",
    sector_col: str | None = None,
    winsorize_pct: float = 0.01,
) -> pd.DataFrame:
    """Winsorize then z-score each feature within date (or date x sector)."""
    out = panel.copy()
    group = [date_col] + ([sector_col] if sector_col else [])

    def _z(s: pd.Series) -> pd.Series:
        s = _winsorize(s, winsorize_pct)
        std = s.std()
        return (s - s.mean()) / std if std and not np.isnan(std) else s * 0.0

    for c in cols:
        out[c] = out.groupby(group)[c].transform(_z)
    return out


def normalize_features(
    panel: pd.DataFrame,
    cols: list[str],
    *,
    scheme: str = "rank",
    sector_neutral: bool = True,
    date_col: str = "date",
    sector_col: str = "gics_sector",
    winsorize_pct: float = 0.01,
) -> pd.DataFrame:
    """Dispatch to the configured normalization scheme.

    ``scheme``: ``"rank"`` (default) or ``"zscore"``.
    ``sector_neutral``: group within sector when True.
    """
    sc = sector_col if sector_neutral else None
    if scheme == "rank":
        return cross_sectional_rank(panel, cols, date_col=date_col, sector_col=sc)
    if scheme == "zscore":
        return cross_sectional_zscore(
            panel, cols, date_col=date_col, sector_col=sc, winsorize_pct=winsorize_pct
        )
    raise ValueError(f"Unknown normalization scheme: {scheme}")
