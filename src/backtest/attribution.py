"""Fama-French factor attribution.

Regress strategy excess returns on FF5 + momentum:

    r_strategy - rf = alpha + b1*Mkt-RF + b2*SMB + ... + b6*MOM + e

The intercept (alpha) is the return the known factors do NOT explain — the part
that is plausibly genuine skill rather than repackaged factor exposure. We
report annualized alpha with a t-stat, the factor betas, and R^2.

OLS is implemented with numpy (no statsmodels dependency).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
_MONTHS = 12


def factor_attribution(strategy_returns: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS of strategy excess return on FF5+MOM. Returns alpha, betas, t-stats, R^2."""
    df = factors.copy()
    df.index = df.index.to_period("M")
    sr = strategy_returns.copy()
    sr.index = pd.PeriodIndex(sr.index, freq="M")

    cols = [c for c in _FACTOR_COLS if c in df.columns]
    data = pd.concat([sr.rename("ret"), df[cols + ["RF"]]], axis=1).dropna()
    if len(data) < len(cols) + 2:
        return {"error": "insufficient overlapping months"}

    y = (data["ret"] - data["RF"]).values
    X = np.column_stack([np.ones(len(data)), data[cols].values])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstats = beta / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot

    names = ["alpha"] + cols
    return {
        "alpha_monthly": beta[0],
        "alpha_annual": (1 + beta[0]) ** _MONTHS - 1,
        "alpha_tstat": tstats[0],
        "betas": dict(zip(cols, beta[1:])),
        "beta_tstats": dict(zip(cols, tstats[1:])),
        "r_squared": r2,
        "n_months": len(data),
        "coef_table": pd.DataFrame(
            {"coef": beta, "t_stat": tstats}, index=names
        ),
    }
