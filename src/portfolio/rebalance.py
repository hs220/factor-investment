"""Turn the model's latest ranking + current holdings into a trade list.

Uses the same hold-buffer rule as the backtest (``select_with_buffer``), so the
live trades are the ones the reported performance assumes:

- **SELL** a holding that fell outside ``buffer_rank``;
- **BUY** the best-ranked names not held, into the slots that frees;
- **ADD / TRIM** a kept holding only when its value drifted more than
  ``drift_band`` from the equal-weight target (avoids churning on small moves);
- **HOLD** otherwise; trades smaller than ``min_trade`` dollars are skipped;
- new **BUY**s must pass the live liquidity floor (``universe.liquidity_filters.
  min_price``): the universe filter ran at ingestion, and a name can since have
  fallen below $5 — still ranked, but not bought;
- **REVIEW** holdings the model doesn't score (ETFs, names outside the
  investable universe) — never traded automatically, excluded from the sleeve.

Whole shares only; the strategy sleeve = scored holdings at market + cash.
"""
from __future__ import annotations

import math

import pandas as pd

from src.config import load_config
from src.portfolio.construct import select_with_buffer

_ORDER = {"SELL": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "HOLD": 4, "REVIEW": 5}


def plan_rebalance(
    positions: pd.DataFrame,
    cash: float,
    ranked: pd.DataFrame,
    prices: pd.Series,
    *,
    n_holdings: int | None = None,
    buffer_rank: int | None = None,
    max_weight: float | None = None,
    drift_band: float = 0.25,
    min_trade: float = 100.0,
) -> tuple[pd.DataFrame, dict]:
    """Return ``(plan, summary)``.

    ``positions``: ticker, shares[, cost_basis]. ``ranked``: the scored
    cross-section best-first (ticker, pred[, gics_sector]). ``prices``: latest
    close by ticker.
    """
    pcfg = load_config("model")["portfolio"]
    costs = load_config("model")["costs"]
    n = n_holdings or pcfg["n_holdings"]
    buf = max(buffer_rank or pcfg.get("buffer_rank", n), n)
    cap = max_weight or pcfg["max_position_weight"]

    ranked = ranked.reset_index(drop=True)
    order = ranked["ticker"].tolist()
    rank = {t: i for i, t in enumerate(order, start=1)}
    info = ranked.set_index("ticker")
    held = positions.groupby("ticker")["shares"].sum()

    def px(t: str) -> float:
        v = prices.get(t)
        return float(v) if v is not None and pd.notna(v) and v > 0 else math.nan

    managed = [t for t in held.index if t in rank]
    sleeve = float(cash) + sum(held[t] * px(t) for t in managed if not math.isnan(px(t)))
    # Buy candidates must be tradable today; held names keep their true rank.
    min_price = load_config("data")["universe"]["liquidity_filters"]["min_price"]
    buyable = [t for t in order if t in held.index or px(t) >= min_price]
    keep, buy, sell = select_with_buffer(buyable, managed, n, buf)
    targets = keep + buy
    w = min(1.0 / len(targets), cap) if targets else 0.0
    target_value = w * sleeve

    rows = []

    def row(t, action, cur, trade, reason):
        p = px(t)
        rows.append({
            "action": action, "ticker": t,
            "sector": info["gics_sector"].get(t) if "gics_sector" in info else None,
            "rank": rank.get(t), "score": info["pred"].get(t) if t in info.index else None,
            "price": p, "current_shares": cur, "trade_shares": trade,
            "target_shares": cur + trade,
            "current_value": cur * p if not math.isnan(p) else math.nan,
            "trade_value": trade * p if not math.isnan(p) else math.nan,
            "reason": reason,
        })

    for t in sell:
        row(t, "SELL", held[t], -held[t], f"rank {rank[t]} fell outside the hold buffer (top {buf})")
    for t in keep:
        p, cur = px(t), held[t]
        if math.isnan(p):
            row(t, "HOLD", cur, 0, "no recent price — not resized")
            continue
        tgt = math.floor(target_value / p)
        drift = (cur * p - target_value) / target_value if target_value else 0.0
        delta = tgt - cur
        if abs(drift) <= drift_band or abs(delta * p) < min_trade:
            row(t, "HOLD", cur, 0, f"rank {rank[t]}, within buffer; weight drift {drift:+.0%}")
        else:
            row(t, "TRIM" if delta < 0 else "ADD", cur, delta,
                f"rank {rank[t]}; weight drift {drift:+.0%} beyond ±{drift_band:.0%} band")
    for t in buy:
        p = px(t)
        if math.isnan(p):
            row(t, "BUY", 0, 0, f"rank {rank[t]} — no recent price, size manually")
            continue
        shares = math.floor(target_value / p)
        if shares * p < min_trade:
            continue
        row(t, "BUY", 0, shares, f"rank {rank[t]}: top-ranked name not held")
    for t in held.index:
        if t not in rank:
            row(t, "REVIEW", held[t], 0, "not scored by the model (ETF / outside universe) — left as is")

    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan, {"sleeve_value": sleeve, "cash": cash}
    plan = plan.sort_values(["action", "rank"], key=lambda s: s.map(_ORDER) if s.name == "action" else s)
    plan = plan.reset_index(drop=True)

    traded = plan["trade_value"].abs().sum(skipna=True)
    bps = (costs["commission_bps"] + costs["spread_bps"]) / 1e4
    summary = {
        "sleeve_value": sleeve,
        "cash": float(cash),
        "target_per_name": target_value,
        "n_sell": int((plan["action"] == "SELL").sum()),
        "n_buy": int((plan["action"] == "BUY").sum()),
        "n_resize": int(plan["action"].isin(["ADD", "TRIM"]).sum()),
        "n_hold": int((plan["action"] == "HOLD").sum()),
        "turnover": traded / 2 / sleeve if sleeve else 0.0,
        "est_cost": traded * bps,
        "cash_after": float(cash) - plan["trade_value"].sum(skipna=True),
    }
    return plan, summary
