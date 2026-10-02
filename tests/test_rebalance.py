"""Holdings CSV parsing + rebalance planning (no warehouse needed)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.portfolio.holdings import parse_positions_csv
from src.portfolio.rebalance import plan_rebalance

SCHWAB = '''"Positions for account Roth IRA ...123 as of 09:30 AM ET, 2026/10/02"

"Symbol","Description","Quantity","Price","Market Value","Cost Basis"
"AAA","AAA CORP","100","$10.00","$1,000.00","$900.00"
"BRK.B","BERKSHIRE","5","$400.00","$2,000.00","$1,500.00"
"VTI","VANGUARD TOTAL","10","$250.00","$2,500.00","$2,000.00"
"Cash & Cash Investments","--","--","--","$3,500.00","--"
"Account Total","--","--","--","$9,000.00","$4,400.00"
'''

FIDELITY = '''Account Number,Account Name,Symbol,Description,Quantity,Last Price,Current Value,Cost Basis Total
X1,ROTH IRA,SPAXX**,HELD IN MONEY MARKET,,,$1250.10,
X1,ROTH IRA,CCC,CCC INC,20,$50.00,"$1,000.00","$1,100.00"
'''


def test_parse_schwab_with_preamble_cash_and_totals():
    pos, cash = parse_positions_csv(SCHWAB.encode())
    assert sorted(pos["ticker"]) == ["AAA", "BRK-B", "VTI"]        # BRK.B -> yfinance form
    assert pos.set_index("ticker").loc["AAA", "shares"] == 100
    assert pos.set_index("ticker").loc["AAA", "cost_basis"] == 900
    assert cash == pytest.approx(3500.0)                            # totals row not cash


def test_parse_fidelity_money_market_is_cash():
    pos, cash = parse_positions_csv(FIDELITY)
    assert pos["ticker"].tolist() == ["CCC"] and cash == pytest.approx(1250.10)


def test_parse_rejects_files_without_symbol_column():
    with pytest.raises(ValueError):
        parse_positions_csv("a,b\n1,2\n")


def _ranked(n=100):
    return pd.DataFrame({"ticker": [f"R{i}" for i in range(1, n + 1)],
                         "pred": [1 - i / 1000 for i in range(1, n + 1)],
                         "gics_sector": "X"})


def test_plan_sells_out_of_buffer_buys_top_and_reviews_unscored():
    ranked = _ranked()
    prices = pd.Series(10.0, index=ranked["ticker"].tolist() + ["VTI"])
    positions = pd.DataFrame({"ticker": ["R2", "R50", "R95", "VTI"], "shares": [100, 100, 100, 10]})
    plan, s = plan_rebalance(positions, 1000.0, ranked, prices,
                             n_holdings=3, buffer_rank=60, max_weight=1.0)
    act = plan.set_index("ticker")["action"].to_dict()
    assert act["R95"] == "SELL"                 # outside buffer 60
    assert act["R1"] == "BUY"                   # one free slot -> best not held
    assert act["R2"] in ("HOLD", "ADD", "TRIM") and act["R50"] in ("HOLD", "ADD", "TRIM")
    assert act["VTI"] == "REVIEW"               # ETF untouched, outside the sleeve
    assert s["sleeve_value"] == pytest.approx(3 * 1000 + 1000)       # VTI excluded
    assert s["cash_after"] >= 0                 # buys funded by sells + cash
    assert (plan.loc[plan.action == "BUY", "trade_shares"] > 0).all()


def test_plan_holds_small_drift_and_skips_tiny_trades():
    ranked = _ranked()
    prices = pd.Series(10.0, index=ranked["ticker"])
    positions = pd.DataFrame({"ticker": ["R1", "R2"], "shares": [105, 95]})   # ~±5% drift
    plan, s = plan_rebalance(positions, 0.0, ranked, prices,
                             n_holdings=2, buffer_rank=10, max_weight=1.0)
    assert set(plan["action"]) == {"HOLD"} and s["turnover"] == 0


def test_plan_never_buys_below_min_price():
    ranked = _ranked()
    prices = pd.Series(10.0, index=ranked["ticker"])
    prices["R1"] = 2.0                                   # top-ranked but a penny stock now
    plan, _ = plan_rebalance(pd.DataFrame(columns=["ticker", "shares"]), 10_000.0, ranked, prices,
                             n_holdings=2, buffer_rank=10, max_weight=1.0)
    assert plan.loc[plan.action == "BUY", "ticker"].tolist() == ["R2", "R3"]
