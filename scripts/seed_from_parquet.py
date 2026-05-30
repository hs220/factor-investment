"""One-time migration: load existing backfilled parquet into Postgres.

Prices / universe / FF / macro were already fetched (data/processed/*.parquet),
so we load them straight into the warehouse rather than re-downloading.
Fundamentals are NOT seeded here — they are re-fetched with the restatement-
aware ingester (via Dagster) into fundamental_facts.

Usage:
    export POSTGRES_PASSWORD=...        # from the NAS .env
    python -m scripts.seed_from_parquet
"""
from __future__ import annotations

from src.data import cache, db


def main() -> None:
    if not db.ping():
        raise SystemExit("Database not reachable — set POSTGRES_PASSWORD and check host.")

    # Universe (attach sectors later from EDGAR; seed names/exchange if present).
    uni = cache.load("universe.parquet")
    n = db.load_universe(uni)
    print(f"universe:    {n} rows")

    # Prices (monthly close for now; volume not persisted in the backfill).
    close = cache.load("stock_prices_monthly.parquet")
    n = db.load_prices_wide(close)
    print(f"prices:      {n} rows")

    n = db.load_ff_factors(cache.load("ff5_monthly.parquet"))
    print(f"ff_factors:  {n} rows")

    n = db.load_macro(cache.load("macro_monthly.parquet"))
    print(f"macro:       {n} rows")

    print("\nSeed complete. Fundamentals come from the Dagster ingester.")


if __name__ == "__main__":
    main()
