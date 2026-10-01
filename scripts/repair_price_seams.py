"""One-time repair: re-fetch full price history for tickers with adjustment seams.

Before ``prices.find_rebased`` existed, the incremental prices load appended
freshly (re-)adjusted yfinance closes to history stored on an older adjustment
basis, so every split/reverse split since the backfill left a fake jump (up to
+16,000%) in ``prices``. This finds tickers whose stored monthly returns contain
a suspicious move and replaces their whole history with one fresh download, so
each series is back on a single adjustment basis.

Run where yfinance + the warehouse are reachable (e.g. the NAS dagster container):
    python -m scripts.repair_price_seams            # dry run: list suspects
    python -m scripts.repair_price_seams --apply
"""
from __future__ import annotations

import argparse

from src.config import load_config
from src.data import db, prices, warehouse

SUSPECT_UP, SUSPECT_DOWN = 1.0, -0.75     # wider net than the panel's mask band


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    px = warehouse.load_prices_wide()
    r = px.pct_change(fill_method=None)
    suspects = sorted(r.columns[((r > SUSPECT_UP) | (r < SUSPECT_DOWN)).any()])
    print(f"{len(suspects)} of {px.shape[1]} tickers have a month > +{SUSPECT_UP:.0%} "
          f"or < {SUSPECT_DOWN:.0%}")
    if not args.apply or not suspects:
        print("dry run — pass --apply to re-fetch their full history")
        return

    start = load_config("data")["prices"]["start_date"]
    end = __import__("pandas").Timestamp.today().strftime("%Y-%m-%d")
    full = prices.download_prices(suspects, start, end, field="Close")
    monthly = prices.to_monthly_close(full)
    n = db.load_prices_wide(monthly)
    missing = sorted(set(suspects) - set(monthly.columns))
    print(f"re-fetched {monthly.shape[1]} tickers, upserted {n} rows; "
          f"{len(missing)} returned no data (left as stored)")

    after = monthly.pct_change(fill_method=None)
    still = int(((after > 3.0) | (after < -0.95)).sum().sum())
    print(f"out-of-band (+300%/-95%) months in the re-fetched series: {still}")


if __name__ == "__main__":
    main()
