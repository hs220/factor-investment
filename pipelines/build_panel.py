"""Assemble the monthly feature panel and materialize it to the warehouse.

Schedulable entry point for the gold training matrix. Reads the warehouse
(prices, fundamental_features, macro, universe sectors) via
``src.factors.panel.assemble_panel(source="db")`` — the same code the notebook
and the Dagster ``panel_monthly`` asset use — and upserts the result into the
``panel_monthly`` gold table. The model stage reads that table instead of
recomputing the panel.

Usage:
    python -m pipelines.build_panel              # build from the warehouse -> DB
    python -m pipelines.build_panel --parquet    # also cache panel.parquet (legacy
                                                 # downstream stages 03-05)
    python -m pipelines.build_panel --no-db      # parquet only, skip the DB write

Requires ``POSTGRES_PASSWORD`` (and ``FACTOR_DB_HOST`` if off-LAN).
"""
from __future__ import annotations

import argparse

from src.data import cache, db
from src.factors.panel import assemble_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", action="store_true",
                    help="also cache data/processed/panel.parquet for stages 03-05")
    ap.add_argument("--no-db", action="store_true",
                    help="skip the warehouse write (parquet only)")
    args = ap.parse_args()

    print("Assembling monthly feature panel from the warehouse...")
    panel = assemble_panel(source="db")
    print(f"  panel: {len(panel):,} rows | {panel['ticker'].nunique()} tickers "
          f"| {panel['date'].nunique()} months | {panel['gics_sector'].nunique()} sectors")

    if not args.no_db:
        n = db.load_panel_monthly(panel)
        print(f"  wrote {n:,} rows -> panel_monthly")

    if args.parquet or args.no_db:
        cache.save(panel, "panel.parquet")
        print("  cached data/processed/panel.parquet")

    print("Done.")


if __name__ == "__main__":
    main()
