"""Postgres/TimescaleDB connection and upsert helpers (the silver warehouse).

Connection params come from ``config/db.yaml`` with env overrides; the password
is read only from the ``POSTGRES_PASSWORD`` (or ``PGPASSWORD``) environment
variable, never from a committed file.

    from src.data import db
    eng = db.get_engine()
    db.upsert(df, "prices", conflict=["ticker", "date"])
    df = db.read_sql("SELECT * FROM universe")
"""
from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

from src.config import load_config


def _conn_params() -> dict:
    cfg = load_config("db")["postgres"]
    return {
        "host": os.environ.get("FACTOR_DB_HOST", cfg["host"]),
        "port": int(os.environ.get("FACTOR_DB_PORT", cfg["port"])),
        "database": os.environ.get("FACTOR_DB_NAME", cfg["database"]),
        "user": os.environ.get("FACTOR_DB_USER", cfg["user"]),
        "password": os.environ.get("POSTGRES_PASSWORD")
        or os.environ.get("PGPASSWORD", ""),
    }


def database_url() -> str:
    p = _conn_params()
    return (
        f"postgresql+psycopg2://{p['user']}:{p['password']}"
        f"@{p['host']}:{p['port']}/{p['database']}"
    )


@lru_cache(maxsize=1)
def get_engine():
    """SQLAlchemy engine (pooled, pre-ping). Cached per process."""
    from sqlalchemy import create_engine

    return create_engine(database_url(), pool_pre_ping=True, future=True)


def read_sql(query: str, **params) -> pd.DataFrame:
    """Run a read query and return a DataFrame."""
    from sqlalchemy import text

    with get_engine().connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params or None)


def upsert(df: pd.DataFrame, table: str, conflict: list[str], *, chunksize: int = 10000) -> int:
    """Insert rows, updating non-key columns on conflict (Postgres ON CONFLICT).

    Uses psycopg2 ``execute_values`` (one multi-row INSERT per chunk) — far
    faster than row-by-row executemany for the large price/fundamental loads.
    ``conflict`` are the primary-key columns. Returns the number of rows sent.
    """
    if df.empty:
        return 0
    from psycopg2.extras import execute_values

    cols = list(df.columns)
    updates = [c for c in cols if c not in conflict]
    collist = ", ".join(f'"{c}"' for c in cols)
    conflict_cols = ", ".join(f'"{c}"' for c in conflict)
    set_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in updates)
    action = f"DO UPDATE SET {set_clause}" if updates else "DO NOTHING"
    sql = (
        f"INSERT INTO {table} ({collist}) VALUES %s "
        f"ON CONFLICT ({conflict_cols}) {action}"
    )

    # NaN/NaT -> None for SQL NULL; rows as plain tuples.
    clean = df.where(pd.notnull(df), None)
    records = list(clean.itertuples(index=False, name=None))

    raw = get_engine().raw_connection()
    try:
        with raw.cursor() as cur:
            for i in range(0, len(records), chunksize):
                execute_values(cur, sql, records[i : i + chunksize], page_size=1000)
        raw.commit()
    finally:
        raw.close()
    return len(records)


def ping() -> bool:
    """True if the database is reachable."""
    from sqlalchemy import text

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Table-specific loaders (map our DataFrame shapes -> warehouse tables).
# --------------------------------------------------------------------------- #
_FF_RENAME = {
    "Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw",
    "CMA": "cma", "RF": "rf", "MOM": "mom",
}


def load_universe(universe: pd.DataFrame) -> int:
    """Upsert the universe table. Accepts columns ticker[, name, exchange,
    gics_sector]; missing optional columns are filled null."""
    df = universe.copy()
    for c in ("name", "exchange", "gics_sector"):
        if c not in df.columns:
            df[c] = None
    df = df[["ticker", "name", "exchange", "gics_sector"]]
    return upsert(df, "universe", ["ticker"])


def load_prices_wide(close: pd.DataFrame, volume: pd.DataFrame | None = None) -> int:
    """Upsert prices from wide (date x ticker) close/volume frames -> long rows.

    Persists only real observations. ``pd.DataFrame(collected)`` in the downloader
    unions every ticker's trading dates into a dense grid, so a name carries NaN
    closes for months it didn't trade (pre-IPO / post-delisting). We also see
    yfinance adjusted-close artifacts (negative / zero closes on names with heavy
    reverse-split + return-of-capital history). Drop both here so the warehouse
    holds only valid, positive prices — independent of pandas ``stack`` NaN
    behavior across versions.
    """
    # Modern pandas stack() keeps NaN cells (the old dropna=True default is gone,
    # and passing dropna now raises) — which is how NaN grid-padding got loaded.
    # Filter explicitly instead: keep only present, positive closes.
    long = close.stack().rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    long = long[long["close"].notna() & (long["close"] > 0)]
    if volume is not None:
        vlong = volume.stack().rename("volume").reset_index()
        vlong.columns = ["date", "ticker", "volume"]
        long = long.merge(vlong, on=["date", "ticker"], how="left")
    else:
        long["volume"] = None
    long["date"] = pd.to_datetime(long["date"]).dt.date
    return upsert(long[["ticker", "date", "close", "volume"]], "prices", ["ticker", "date"])


def load_ff_factors(ff: pd.DataFrame) -> int:
    """Upsert FF5+MOM (date-indexed, FF column names) -> ff_factors."""
    df = ff.rename(columns=_FF_RENAME).reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    cols = ["date"] + [c for c in _FF_RENAME.values() if c in df.columns]
    return upsert(df[cols], "ff_factors", ["date"])


def load_macro(macro: pd.DataFrame) -> int:
    """Upsert macro (date-indexed: yield_curve, vix, hy_spread) -> macro."""
    df = macro.reset_index().rename(columns={macro.index.name or "index": "date"})
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    cols = ["date"] + [c for c in ("yield_curve", "vix", "hy_spread") if c in df.columns]
    return upsert(df[cols], "macro", ["date"])


def load_fundamental_facts(facts: pd.DataFrame) -> int:
    """Upsert raw fundamental facts (already in fundamental_facts column shape)."""
    df = facts.copy()
    for c in ("period_end", "filed_date"):
        df[c] = pd.to_datetime(df[c]).dt.date
    return upsert(df, "fundamental_facts",
                  ["ticker", "concept", "period_end", "filed_date"])


def set_active(active_tickers: list[str]) -> int:
    """Mark the investable (liquidity-passing) set: is_active=true for the given
    tickers, false for all others. universe_table loads all listed names; this
    is how the liquidity filter is recorded so downstream reads WHERE is_active."""
    active = list(active_tickers)
    raw = get_engine().raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("UPDATE universe SET is_active = FALSE")
            if active:
                cur.execute(
                    "UPDATE universe SET is_active = TRUE WHERE ticker = ANY(%s)",
                    (active,),
                )
        raw.commit()
    finally:
        raw.close()
    return len(active)
