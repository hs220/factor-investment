"""Re-derive universe.gics_sector from the stored raw SIC — no EDGAR re-fetch.

Run after tweaking the SIC->sector mapping in src/data/fundamentals.py:
    export POSTGRES_PASSWORD=...
    python -m scripts.remap_sectors
"""
from __future__ import annotations

from src.data import db
from src.data.fundamentals import _sic_to_sector


def main() -> None:
    df = db.read_sql("SELECT ticker, sic FROM universe WHERE sic IS NOT NULL")
    if df.empty:
        print("No stored SIC codes — run the sectors asset first.")
        return
    df["gics_sector"] = df["sic"].map(_sic_to_sector)
    n = db.upsert(df[["ticker", "gics_sector"]], "universe", ["ticker"])
    mapped = int(df["gics_sector"].notna().sum())
    print(f"Re-mapped {n} tickers from stored SIC; {mapped} have a sector "
          f"({mapped / len(df) * 100:.1f}%).")


if __name__ == "__main__":
    main()
