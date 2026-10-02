"""Current Roth IRA holdings: parse a broker positions CSV, persist snapshots.

Brokers export positions with different headers (Schwab "Symbol/Quantity/Cost
Basis", Fidelity "Symbol/Quantity/Cost Basis Total", ...) plus cash/sweep and
total rows. ``parse_positions_csv`` normalizes them to (ticker, shares,
cost_basis) + a cash amount. Snapshots are stored by upload date in the
``holdings`` table, so the latest one drives the rebalance plan.
"""
from __future__ import annotations

import io
import re

import pandas as pd

_TICKER = ("symbol", "ticker")
_SHARES = ("quantity", "shares", "qty")
_COST = ("cost basis total", "cost basis", "cost_basis", "total cost")
_VALUE = ("market value", "current value", "value")
# Money-market / sweep vehicles count as cash; summary rows are dropped entirely.
_CASH = re.compile(r"(?:cash|money market|sweep|core|\*\*|pending)", re.I)
_TOTAL = re.compile(r"(?:account total|^total$|^--$)", re.I)


def _num(s: pd.Series) -> pd.Series:
    """'$1,234.50' / '(12.0)' / '--' -> float."""
    t = s.astype(str).str.strip()
    neg = t.str.startswith("(") & t.str.endswith(")")
    t = t.str.replace(r"[$,()%\s]", "", regex=True).replace({"": None, "--": None, "N/A": None})
    out = pd.to_numeric(t, errors="coerce")
    return out.where(~neg, -out)


def _col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    cols = {c.strip().lower(): c for c in df.columns}
    for n in names:
        if n in cols:
            return cols[n]
    return None


def parse_positions_csv(raw: bytes | str) -> tuple[pd.DataFrame, float]:
    """Broker positions export -> (positions[ticker, shares, cost_basis], cash).

    Skips preamble lines before the header row (Schwab prefixes an account line),
    folds cash/money-market rows into ``cash`` (by market value), and drops
    summary rows. Raises ``ValueError`` if no symbol/quantity columns are found.
    """
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if re.search(r"\b(symbol|ticker)\b", l, re.I)), None)
    if start is None:
        raise ValueError("no header row with a Symbol/Ticker column")
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), dtype=str, skip_blank_lines=True)

    tcol, qcol = _col(df, _TICKER), _col(df, _SHARES)
    if tcol is None or qcol is None:
        raise ValueError(f"need Symbol and Quantity columns; got {list(df.columns)}")
    ccol, vcol = _col(df, _COST), _col(df, _VALUE)

    df = df[df[tcol].notna()].copy()
    df["ticker"] = df[tcol].str.strip().str.upper()
    df = df[~df["ticker"].str.contains(_TOTAL)]
    is_cash = df["ticker"].str.contains(_CASH) | df[qcol].isna()
    cash = float(_num(df.loc[is_cash, vcol]).sum()) if vcol else 0.0

    pos = df[~is_cash].copy()
    pos["shares"] = _num(pos[qcol])
    pos["cost_basis"] = _num(pos[ccol]) if ccol else float("nan")
    pos = pos[pos["shares"].fillna(0) > 0]
    pos["ticker"] = pos["ticker"].str.replace(".", "-", regex=False)   # BRK.B -> BRK-B (yfinance)
    out = pos.groupby("ticker", as_index=False)[["shares", "cost_basis"]].sum(min_count=1)
    return out, cash


# Snapshot store: one row per (as_of, ticker); cash is the reserved ticker below.
CASH_TICKER = "$CASH"
_HOLDINGS_DDL = """
CREATE TABLE IF NOT EXISTS holdings (
    as_of       date NOT NULL,
    ticker      text NOT NULL,
    shares      double precision NOT NULL,     -- dollars for the $CASH row
    cost_basis  double precision,
    PRIMARY KEY (as_of, ticker)
);
"""


def save_snapshot(positions: pd.DataFrame, cash: float, as_of: str | pd.Timestamp) -> int:
    """Replace the snapshot for ``as_of`` with these positions + cash."""
    from sqlalchemy import text

    from src.data import db

    as_of = pd.Timestamp(as_of).date()
    rows = positions[["ticker", "shares", "cost_basis"]].copy()
    rows = pd.concat([rows, pd.DataFrame([{"ticker": CASH_TICKER, "shares": float(cash),
                                           "cost_basis": None}])], ignore_index=True)
    rows.insert(0, "as_of", as_of)
    with db.get_engine().begin() as conn:
        conn.execute(text(_HOLDINGS_DDL))
        conn.execute(text("DELETE FROM holdings WHERE as_of = :d"), {"d": as_of})
    return db.upsert(rows, "holdings", ["as_of", "ticker"])


def load_latest_snapshot() -> tuple[pd.DataFrame, float, pd.Timestamp | None]:
    """(positions, cash, as_of) of the newest snapshot; empty if none saved yet."""
    from sqlalchemy.exc import ProgrammingError

    from src.data import db

    empty = pd.DataFrame(columns=["ticker", "shares", "cost_basis"])
    try:
        df = db.read_sql("SELECT * FROM holdings WHERE as_of = (SELECT max(as_of) FROM holdings)")
    except ProgrammingError:          # table not created yet
        return empty, 0.0, None
    if df.empty:
        return empty, 0.0, None
    cash = float(df.loc[df["ticker"] == CASH_TICKER, "shares"].sum())
    pos = df[df["ticker"] != CASH_TICKER][["ticker", "shares", "cost_basis"]].reset_index(drop=True)
    return pos, cash, pd.Timestamp(df["as_of"].iloc[0])
