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

## Training vs. Inference (one feature path, two modes)

Every stage here serves **both** training and live inference. We are building the
training side first, but the code must be written so inference reuses it verbatim.
The chief failure mode is **training-serving skew** — live features computed by
different code than training features. The only defense: a single feature code
path, with the differences between modes isolated to the edges.

**Shared, identical in both modes (no parallel "live" implementation allowed):**
- `src/data` warehouse reads and `src/factors` panel assembly + normalization.
  Normalization is **cross-sectional within each date** → stateless across time,
  so there is no fitted scaler that can drift between train and serve.
- the model's `predict`, `src/portfolio/construct`, and the `predictions` /
  `portfolios` tables (the junction where the two modes meet).

**Training-only:**
- the **target** (forward return) — unknowable at the live edge. It is just an
  extra column, NaN for the most recent unrealized period; `assemble_panel`
  already yields feature rows with a NaN target there.
- walk-forward CV (`src/models/walkforward.py`) — itself a fit→predict loop, so
  its per-step predict is the *same function* live inference calls.
- model **fit**.

**Inference-only:** a thin driver — take the latest cross-section of features
(target absent) → load the deployed model → predict → write `predictions` →
`portfolio.construct` → recommendations.

### Rules that keep train == serve
1. **One feature function.** `assemble_panel(...)` builds features the same way for
   both; expose a feature-only view (`with_target=False` / `as_of=<date>`) so
   inference never depends on the target existing.
2. **Any *fitted* transform must live in the model artifact.** Today none does
   (normalization is per-date cross-sectional). If that ever changes (global
   z-score, quantile bins, imputation means), the fitted state is saved with the
   model and reloaded at inference — never recomputed on live data.
3. **Identical PIT discipline.** `availability_date` as-of joins and `.shift(1)`
   apply unchanged; inference's `as_of = today` gates exactly like a training row.
4. **`predict` is one function**, called by the walk-forward harness and the live
   driver alike. Factor `walk_forward_predict` into `fit(train) -> model` +
   `predict(model, features) -> scores` (in `src/models/rankers.py`) so the live
   path reuses `predict` directly.

### Hyperparameter tuning (nested walk-forward) — built
`config/model.yaml` model params may be **lists** (tuning candidates) or scalars
(fixed); `config/model.yaml::tuning` controls the search. Each outer fold tunes
**inside** its training window via inner time-ordered validation splits
(`src/models/tuning.py`), maximizing inner-validation IC, so the outer test block
is never seen — tuning stays causal. Re-tuned every `retune_every` folds (default
12) to bound cost. `walk_forward_predict(tune=True, model_name=...)` records the
per-fold chosen params on `result.attrs["tuning_history"]`.

### Model artifact + registry — built
`pipelines/train.py` now, after walk-forward *validation*, tunes on all data and
fits a **deployment model** on every row through `as_of`, persisting a versioned
**artifact** (`src/models/artifact.py`) under a gitignored `models/<horizon>/
<model_version>/`:
- `model.joblib` — the fitted estimator (joblib; uniform across model types).
- `manifest.json` — `model_version, model_name, horizon, feature_list,
  hyperparams, normalization, target, train_window, n_train_rows, oos_metrics,
  code_sha, created_at`. `models/<horizon>/latest.txt` points at the newest.
- `pipelines/predict.py` loads the latest artifact, **asserts the live feature set
  covers the manifest `feature_list`** (`predict_with_artifact` — the explicit
  skew guard), and scores the latest cross-section without re-training.
- A `model_registry` table is still optional; the filesystem `latest.txt` pointer
  plus the `predictions.model_version` key suffice for now.

### Unified predictions sink
Walk-forward OOS predictions (historical) and live predictions (latest) have the
**same shape** → both land in `predictions` `(date, ticker, horizon,
model_version)`. **Backtest** reads all history; **live** reads `max(date)`; both
feed the same `portfolio.construct`. Action: add `db.load_predictions` /
`db.load_portfolios` and migrate `train.py` off `predictions.parquet` onto the
table (the current parquet is a temporary stand-in).

### Cadence decoupling
Refit (training) is periodic and expensive (monthly/quarterly expanding window);
inference (predict + construct) runs every rebalance and is cheap, reusing the
last deployed artifact. A stale model still infers daily; refit on schedule. Each
horizon is its own model family — `predictions.horizon` already separates them, so
one panel → N targets → N artifacts → N prediction streams, combined in the book.

### Inference lane in the asset graph
```
panel_* (features, as_of=today) ─► predict_<h> (load model_<h>) ─► predictions[today]
                                                                       │
                                              portfolio.construct ─► portfolios / recommendations
```
The training lane produces the `model_<h>` artifacts + historical `predictions`;
the inference lane consumes an artifact and emits today's row. Same code, same
tables.

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
5. **Inference / live recommendations** — persist a versioned model artifact from
   training; add `pipelines/predict.py` (then a Dagster `predict_<h>` asset) that
   assembles the latest feature cross-section, loads the artifact, writes today's
   `predictions`, and constructs the recommendation book. Reuses the *exact*
   feature + predict + portfolio code from training (see Training vs. Inference).

## Implications / watch-items

- **Selection stays ~monthly** (our current panel); **Timing needs a new daily panel** —
  two cadences, one set of source tables.
- **Walk-forward embargo uses the longest horizon (30d)** to avoid overlapping-label
  leakage across the 5/10/30d models.
- **Training-serving skew is the top risk:** one feature code path for train and
  serve, fitted transforms (if any) saved in the model artifact, and an explicit
  feature-list assertion at inference. See *Training vs. Inference*.
- DB-as-source-of-truth makes laptop querying a SQL connection, not a file mount.
- Don't boil the ocean: Phase 1 is the unlock; build it before touching Dagster.
