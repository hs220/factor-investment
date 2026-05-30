# Production Architecture

Target architecture for the daily production system. Supersedes the local-parquet
+ cron approach in `modeling.md` (which remains the description of the *modeling*
logic). Built incrementally — see Migration Phases.

## Decisions (locked)

- **Orchestration:** Dagster — software-defined assets, partitions (daily backfill),
  lineage / staleness tracking. Per-dataset clean reruns are native.
- **Storage:** Postgres + TimescaleDB as the curated **silver** warehouse;
  parquet remains the immutable **bronze** landing zone and **gold** training-matrix
  export. (Medallion: bronze raw → silver DB → gold features.)
- **Modeling:** two roles, not redundant pickers —
  - **Selection (30d / monthly):** cross-sectional ranker over fundamentals +
    quality/value/momentum → *what to own*.
  - **Timing (5–10d):** technical/microstructure/reversal model → *when to enter/
    exit* within the selected set. Optional overlay; the 30d core ships first.

## Data Model (Postgres / Timescale)

| Table | Type | Key | Notes |
|---|---|---|---|
| `universe` | table | ticker | name, exchange, gics_sector, first/last_seen, is_active |
| `prices` | **hypertable** | (ticker, date) | daily close/volume; Timescale partition by date |
| `fundamentals` | table | (ticker, period_end, filed_date) | **PIT**: as-of queries via `filed_date`; handles restatements |
| `ff_factors` | table | date | monthly FF5+MOM |
| `macro` | table | date | monthly yield curve, VIX, HY spread |
| `predictions` | **hypertable** | (date, ticker, horizon, model_version) | OOS scores per horizon |
| `portfolios` | table | (date, ticker, strategy) | target weights |

The `fundamentals` PIT key is the main reason for the DB: "value as known on date X"
is a native query (latest `filed_date <= X`), which forward-filled parquet cannot
express across restatements.

## Asset Graph (Dagster)

```
ingest (partitioned):
  universe ──┐
  prices(daily) ──┬─► panel_monthly ─► pred_30d ─► selection_portfolio ─┐
  fundamentals(incremental) ──┘                                          ├─► target_book
  prices(daily) ──────────────► panel_daily ─► pred_10d/pred_5d ─► timing_overlay ─┘
  ff_factors(monthly), macro(monthly) ─► (feed both panels + attribution)
```

- Two feature cadences from the **same** source tables: `panel_monthly` (selection)
  and `panel_daily` (timing). Adding a horizon = another `pred_*` asset on a panel,
  not a new pipeline.
- Each asset idempotent + date-partitioned → clean reruns and backfill.

## Deployment (Synology, Docker)

- `timescaledb` container (volume-backed Postgres data) — source of truth.
- `dagster` container(s) (webserver + daemon) — schedules + UI.
- pipeline image (existing) provides the asset compute.
- **Laptop** connects to Postgres over LAN `:5432` for ad-hoc SQL (this replaces the
  NFS-for-querying need); training reads a **gold parquet export** for fast bulk loads.
- Secrets/conn via env / `config/db.yaml` (not committed).

## Cadence

| Asset | Schedule |
|---|---|
| prices | daily (after close) |
| fundamentals | daily *check*, fetch only new filings (incremental) |
| ff_factors / macro | monthly |
| panel_monthly, pred_30d, selection_portfolio | monthly (or daily recompute, acts monthly) |
| panel_daily, pred_5/10d, timing_overlay | daily |

## Migration Phases

1. **Storage foundation** — stand up Timescale container; SQL schema/migrations;
   `src/data/db.py` repository layer (read/write tables) abstracting the current
   `cache.py`; one-time ETL loading the existing backfilled parquet into Postgres;
   point pipelines at the DB. *Everything else depends on this.*
2. **Dagster** — wrap existing `src/` functions as partitioned assets; run on the NAS;
   IO managers (DB-backed); schedules above.
3. **Daily panel + Timing model** — build `panel_daily`; port current 30d as `pred_30d`;
   add `pred_10d/5d` timing overlay (new daily price/microstructure features).
4. **Portfolio + backtest** on the new store; selection book + timing overlay; FF
   attribution unchanged.

## Implications / watch-items

- **Selection stays ~monthly** (our current panel); **Timing needs a new daily panel** —
  two cadences, one set of source tables.
- **Walk-forward embargo uses the longest horizon (30d)** to avoid overlapping-label
  leakage across the 5/10/30d models.
- DB-as-source-of-truth makes laptop querying a SQL connection, not a file mount.
- Don't boil the ocean: Phase 1 is the unlock; build it before touching Dagster.
