"""Performance metrics for a monthly return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

_MONTHS = 12


def cumulative(returns: pd.Series) -> pd.Series:
    """Growth of $1."""
    return (1 + returns.fillna(0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown series from the running peak of the equity curve."""
    eq = cumulative(returns)
    return eq / eq.cummax() - 1.0


def performance_summary(returns: pd.Series, *, rf: pd.Series | float = 0.0) -> dict[str, float]:
    """Annualized return/vol/Sharpe/Sortino, max drawdown, hit rate."""
    r = returns.dropna()
    if r.empty:
        return {}
    excess = r - (rf.reindex(r.index) if isinstance(rf, pd.Series) else rf)
    ann_ret = (1 + r).prod() ** (_MONTHS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(_MONTHS)
    downside = r[r < 0].std() * np.sqrt(_MONTHS)
    sharpe = (excess.mean() * _MONTHS) / ann_vol if ann_vol else np.nan
    sortino = (excess.mean() * _MONTHS) / downside if downside else np.nan
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown(r).min(),
        "hit_rate": (r > 0).mean(),
        "n_months": len(r),
    }
