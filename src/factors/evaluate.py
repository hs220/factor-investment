"""Information Coefficient (IC) and quantile-portfolio evaluation.

The IC is the primary metric for a cross-sectional ranking signal: the per-date
Spearman rank correlation between a signal and the subsequent return. We never
judge a signal by return-level accuracy (RMSE) — only by ordering quality.

Key outputs:
- ``information_coefficient`` -> per-date IC series
- ``ic_summary`` -> mean IC, IC IR (mean/std), t-stat, hit rate
- ``quantile_returns`` -> mean forward return by signal quantile (long-only
  focus: the top quantile is what we can actually hold)
- ``signal_report`` -> IC summary table across many signals
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(
    panel: pd.DataFrame,
    signal: str,
    *,
    target: str = "forward_return",
    date_col: str = "date",
    sector_col: str | None = None,
) -> pd.Series:
    """Per-date Spearman rank IC between ``signal`` and ``target``.

    If ``sector_col`` is given, IC is computed within sector then averaged
    across sectors each date (sector-neutral IC).
    """
    def _spearman(df: pd.DataFrame) -> float:
        sub = df[[signal, target]].dropna()
        if len(sub) < 5:
            return np.nan
        return stats.spearmanr(sub[signal], sub[target]).correlation

    if sector_col:
        per = panel.groupby([date_col, sector_col]).apply(_spearman)
        return per.groupby(level=0).mean().dropna()
    return panel.groupby(date_col).apply(_spearman).dropna()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Summarize an IC series: mean, std, IR, t-stat, hit rate, n."""
    ic = ic.dropna()
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    ir = mean / std if std else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ir,                      # information ratio of the IC series
        "t_stat": ir * np.sqrt(n) if n else np.nan,
        "hit_rate": (ic > 0).mean(),      # fraction of months with positive IC
        "n_months": n,
    }


def quantile_returns(
    panel: pd.DataFrame,
    signal: str,
    *,
    fwd_return: str = "forward_return",
    n_quantiles: int = 10,
    date_col: str = "date",
) -> pd.DataFrame:
    """Mean forward return per signal quantile, averaged over time.

    Returns a DataFrame indexed by quantile (1=lowest signal, N=highest) with
    the mean forward return and a top-minus-bottom row.
    """
    def _bucket(df: pd.DataFrame) -> pd.Series:
        sub = df[[signal, fwd_return]].dropna()
        if len(sub) < n_quantiles:
            return pd.Series(dtype=float)
        q = pd.qcut(sub[signal].rank(method="first"), n_quantiles, labels=False) + 1
        return sub.groupby(q)[fwd_return].mean()

    per_date = panel.groupby(date_col).apply(_bucket)
    if per_date.empty:
        return pd.DataFrame()
    means = per_date.groupby(level=-1).mean()
    out = means.to_frame("mean_fwd_return")
    out.index.name = "quantile"
    if n_quantiles in out.index and 1 in out.index:
        out.loc["top_minus_bottom", "mean_fwd_return"] = (
            out.loc[n_quantiles, "mean_fwd_return"] - out.loc[1, "mean_fwd_return"]
        )
    return out


def signal_report(
    panel: pd.DataFrame,
    signals: list[str],
    *,
    target: str = "forward_return",
    date_col: str = "date",
    sector_col: str | None = None,
) -> pd.DataFrame:
    """IC summary table across signals, sorted by IC IR."""
    rows = {}
    for s in signals:
        ic = information_coefficient(
            panel, s, target=target, date_col=date_col, sector_col=sector_col
        )
        rows[s] = ic_summary(ic)
    return pd.DataFrame(rows).T.sort_values("ic_ir", ascending=False)
