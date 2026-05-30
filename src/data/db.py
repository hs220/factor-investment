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


def upsert(df: pd.DataFrame, table: str, conflict: list[str], *, chunksize: int = 5000) -> int:
    """Insert rows, updating non-key columns on conflict (Postgres ON CONFLICT).

    ``conflict`` are the primary-key columns. Returns the number of rows sent.
    """
    if df.empty:
        return 0
    from sqlalchemy import text

    cols = list(df.columns)
    updates = [c for c in cols if c not in conflict]
    collist = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    conflict_cols = ", ".join(f'"{c}"' for c in conflict)
    set_clause = (
        ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in updates)
        if updates
        else None
    )
    action = f"DO UPDATE SET {set_clause}" if set_clause else "DO NOTHING"
    stmt = text(
        f'INSERT INTO {table} ({collist}) VALUES ({placeholders}) '
        f"ON CONFLICT ({conflict_cols}) {action}"
    )

    records = df.where(pd.notnull(df), None).to_dict("records")
    sent = 0
    with get_engine().begin() as conn:
        for i in range(0, len(records), chunksize):
            conn.execute(stmt, records[i : i + chunksize])
            sent += len(records[i : i + chunksize])
    return sent


def ping() -> bool:
    """True if the database is reachable."""
    from sqlalchemy import text

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False
