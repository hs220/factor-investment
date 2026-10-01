"""Assemble the daily (ticker, trading day) multi-horizon panel for timing models.

The monthly panel (``panel.assemble_panel``) drives *selection*; this one drives
*timing* at t+5 / t+10 / t+30 trading days. Design (plans/roadmap.md, locked):

- separate from the monthly panel (daily resolution, ~21x the rows);
- ONE panel, N target columns (``target_{h}d``) — features are shared, only the
  target (and, at training time, the embargo) differs per horizon;
- fundamentals still forward-fill by ``availability_date`` (PIT, filing cadence),
  reusing the monthly panel's PIT join and valuation ratios.

No lookahead: every feature at day t uses bars through the close of t; the
target covers (t, t+h]. Implausible daily prints are masked before anything is
derived (``panel_daily.return_bounds``).

Training-time note: overlapping h-day labels mean adjacent days share most of
their target window — walk-forward folds must embargo >= max(horizons) days, and
IC t-stats must come from non-overlapping samples (see ``ic_by_horizon``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.factors.evaluate import ic_summary, information_coefficient
from src.factors.normalize import normalize_features
from src.factors.panel import clean_returns, compute_valuation_ratios, pit_join_fundamentals

_TRADING_YEAR = 252


def compute_daily_features(bars: dict[str, pd.DataFrame], returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wide (date x ticker) technical features from daily bars + cleaned returns."""
    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]
    lr = np.log1p(returns)

    def window_ret(n: int, skip: int = 0) -> pd.DataFrame:
        return np.expm1(lr.shift(skip).rolling(n, min_periods=n).sum())

    dollar_vol = (close * volume).replace(0, np.nan)
    return {
        "rev_1d": returns,
        "rev_5d": window_ret(5),
        "rev_21d": window_ret(21),
        "mom_252_21": window_ret(_TRADING_YEAR - 21, skip=21),     # t-252 .. t-21
        "vol_21d": returns.rolling(21, min_periods=21).std(),
        "vol_63d": returns.rolling(63, min_periods=63).std(),
        "volume_surge": volume.rolling(20, min_periods=20).mean()
        / volume.rolling(60, min_periods=60).mean().replace(0, np.nan),
        "amihud_21d": (returns.abs() / dollar_vol).rolling(21, min_periods=15).mean() * 1e6,
        "range_21d": ((high - low) / close).rolling(21, min_periods=15).mean(),
        "dist_52w_high": close / close.rolling(_TRADING_YEAR, min_periods=126).max(),
    }


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple return over (t, t+h] trading days; NaN unless all h days exist
    (a masked bad print inside the window voids the label rather than faking it)."""
    lr = np.log1p(returns)
    return np.expm1(lr.rolling(horizon, min_periods=horizon).sum().shift(-horizon))


def _stack(wides: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Several aligned wide frames -> one long (date, ticker, *names) frame."""
    long = pd.concat({k: v.stack(future_stack=True) for k, v in wides.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    return long.reset_index()


def assemble_panel_daily(
    *,
    since: str | None = None,
    horizons: list[int] | None = None,
    normalize: bool = True,
    every: int = 1,
    bars: dict[str, pd.DataFrame] | None = None,
    fund: pd.DataFrame | None = None,
    sectors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the daily panel: features + ``fwd_{h}d`` + ``target_{h}d`` per horizon.

    ``since`` trims the *output* dates (lookback windows still use the full bar
    history). ``every`` keeps every k-th trading day — the notebook uses it to
    bound memory; production keeps every day. ``bars``/``fund``/``sectors`` may
    be injected (tests); by default they are read from the warehouse.
    """
    cfg = load_config("features")
    dcfg, ncfg = cfg["panel_daily"], cfg["normalization"]
    sector_col = cfg["panel"]["sector_field"]
    horizons = horizons or list(dcfg["horizons"])
    features = list(dcfg["features"])

    if bars is None or fund is None or sectors is None:
        from src.data import warehouse
        bars = bars if bars is not None else warehouse.load_prices_daily()
        fund = fund if fund is not None else warehouse.load_fundamental_features()
        sectors = sectors if sectors is not None else warehouse.load_sectors()

    raw = bars["close"].pct_change(fill_method=None)
    returns, n_masked = clean_returns(raw, dcfg["return_bounds"])

    wides = compute_daily_features(bars, returns)
    wides["price"] = bars["close"]
    for h in horizons:
        wides[f"fwd_{h}d"] = forward_returns(returns, h)

    keep_dates = bars["close"].index
    if since:
        keep_dates = keep_dates[keep_dates >= pd.Timestamp(since)]
    keep_dates = keep_dates[::every]
    panel = _stack({k: v.reindex(keep_dates) for k, v in wides.items()})
    panel = panel.dropna(subset=["price"])

    # PIT fundamentals -> daily valuation features (same code as the monthly panel).
    panel = pit_join_fundamentals(panel, fund)
    panel = compute_valuation_ratios(panel)
    smap = sectors.dropna(subset=[sector_col]).drop_duplicates("ticker")
    panel = panel.merge(smap[["ticker", sector_col]], on="ticker", how="left")

    fwd_cols = [f"fwd_{h}d" for h in horizons]
    panel = panel[["date", "ticker", sector_col, "market_cap", *features, *fwd_cols]]
    if normalize:
        panel = normalize_features(panel, features, scheme=ncfg["scheme"],
                                   sector_neutral=ncfg["sector_neutral"], sector_col=sector_col)
        for h in horizons:
            panel[f"target_{h}d"] = panel.groupby(["date", sector_col])[f"fwd_{h}d"].rank(pct=True)

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    panel.attrs["returns_masked"] = n_masked
    return panel


def ic_by_horizon(panel: pd.DataFrame, signals: list[str], horizons: list[int]) -> pd.DataFrame:
    """Single-signal IC vs each horizon's forward return, on NON-overlapping dates.

    Daily h-day labels overlap, so consecutive daily ICs are strongly
    autocorrelated and a naive t-stat over every day is inflated ~sqrt(h)x.
    Sampling every h-th date gives independent periods for the summary stats.
    """
    dates = np.sort(panel["date"].unique())
    rows = []
    for h in horizons:
        sub = panel[panel["date"].isin(dates[::h])]
        for s in signals:
            summ = ic_summary(information_coefficient(sub, s, target=f"fwd_{h}d"))
            rows.append({"horizon": f"{h}d", "signal": s, **summ})
    return pd.DataFrame(rows)
