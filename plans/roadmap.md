# Roadmap: Factor Analysis for the Roth IRA (Strategy B)

The canonical, end-to-end plan — keyed to the six steps of the factor-research
workflow. This is the index that ties together the two design docs:
- **`modeling.md`** — the *modeling logic* (problem formulation, features, eval).
- **`architecture.md`** — the *production infra* (Postgres/Timescale warehouse,
  Dagster assets, medallion bronze→silver→gold).

This doc is the source of truth for **what we're building, in what order, and
what's already done**. Update the status markers as we go.

## Goal

Pick the stocks that will **out-rank** their peers over the next month, and hold
a **long-only, sector-neutral** basket of the top names in a Roth IRA. Classical
Fama-French factor analysis as the backbone, ML-enhanced cross-sectional ranking
on top.

### Locked decisions (do not relitigate without cause)
- **Long-only** (Roth prohibits shorting) → we care most about ranking the *top*
  of the distribution correctly.
- **Target: 1-month forward return**, as a within-sector cross-sectional rank.
- **Sector-neutral** ranking/selection (within GICS sector).
- **Universe: Russell 3000** (~3,000 names; small-cap breadth where premia live).
- **Strategy B (stock selection) first**; Strategy A (ETF allocation) reuses the
  same infra later.
- **Roth advantage:** no capital-gains drag → factor rotation / turnover is viable.

### Invariants that must hold at every step
- **No lookahead / PIT:** features at month `t` use only data known by end of `t`;
  the target covers `(t, t+1]`. Fundamentals are stamped with a filing-availability
  date and forward-filled from there. `.shift(1)` discipline on all features.
- **Walk-forward only:** expanding window + a **1-month embargo** (sized to the
  *longest* horizon = 30d, so 5/10/30d labels never leak). Never random K-fold,
  never in-sample evaluation.
- **Evaluate on rank quality, not return levels:** Information Coefficient
  (per-date Spearman corr of preds vs realized), its IR (`mean(IC)/std(IC)`), and
  long-only top-quantile spread vs benchmark — never RMSE on returns.
- **Caveats stated on every result:** survivorship bias (current-membership
  universe), ~5y fundamental depth (yfinance/EDGAR limit).

---

## The six-step pipeline

> **The big picture on status.** The original build (`modeling.md`, stages 0–5)
> implemented all six steps **on local parquet**. The warehouse/Dagster migration
> (`architecture.md` Phases 1–2) so far has only rebuilt **Step 1** on the new
> infra. The model/portfolio/backtest *logic* (Steps 3–6) is **pure
> DataFrame-in/out compute and is reusable as-is** — the parquet coupling lives
> only in `panel.py` (`cache.load`), `query.py` (DuckDB), and the three
> `pipelines/*.py` scripts. So the next phase is: **port the feature/panel layer
> (Step 2) onto the warehouse — including the one genuinely missing piece, the
> raw-facts→ratios bridge — then re-wire Steps 3–6 to read the warehouse-built
> panel and run as Dagster assets.**

### 1. Pull data  —  ✅ DONE (on the warehouse)
Sources & landing tables (Postgres/Timescale `factor` DB):

| Source | Table | Notes |
|---|---|---|
| iShares **IWV** holdings | `universe` | ticker, name, exchange, `gics_sector` (from SEC SIC), `is_active` (liquidity-passing set) |
| **yfinance** | `prices` (hypertable) | monthly close/volume; liquidity filters applied |
| **SEC EDGAR** | `fundamental_facts` | **restatement-aware raw facts**, PIT-keyed `(ticker, concept, period_end, filed_date)` |
| **Ken French** | `ff_factors` | monthly FF5 + MOM |
| **FRED** | `macro` | yield curve, VIX, HY spread |

- Dagster ingestion assets + schedules (daily prices/fundamentals incremental;
  monthly universe/FF/macro/sectors), `RetryPolicy`, `run_monitoring`.
- **Data-quality asset checks** in `orchestration/checks.py` (PIT no-lookahead,
  positive prices, freshness, coverage, unique keys).
- **Recently fixed:** the `fundamentals_no_lookahead` failure — nearest-quarter
  `_snap_q` + drop future-dated facts; `fundamental_facts` re-materialized.

### 2. Create signals / features  —  🟡 THE MAIN GAP
Two sub-parts. The first does not exist yet for the new format; the second exists
but reads parquet.

**2a. Fundamentals derivation bridge — ⛔ MISSING (the linchpin).**
The new `fundamental_facts` table is *long, raw, per-concept* (revenue, net_income,
assets, equity, cash, debt, shares, …) with change-points across filings. The
panel join (`pit_join_fundamentals`) expects *per-(ticker, quarter)* **TTM
aggregates + ratios** (`net_income_ttm`, `ebitda_ttm`, `roe`, `gross_margin`,
`profit_margin`, `accruals`, `revenue_growth_yoy`, `asset_growth_yoy`, …) each
stamped with an **`availability_date`**. We must build the transform:
`fundamental_facts (long) → quarterly ratios (wide) + availability_date`.
This unblocks everything downstream. → new `src/factors/fundamentals_features.py`.

**2b. Panel assembly — ✅ logic exists, 🟡 reads parquet.**
`src/factors/panel.py::assemble_panel` already does: price/technical features
(momentum 12-2 / 6-1, volatility), PIT `merge_asof` fundamental join, valuation
ratios from month-end market cap (earnings yield, book/price, EV/EBITDA, size),
macro broadcast, target construction, and cross-sectional **sector-neutral
normalization** (`normalize.py`). Re-wire its inputs from `cache.load(...)` →
warehouse reads (via `src/data/db.py`), fed by 2a.

**Feature families:** value, quality, growth, momentum, size, low-vol, macro regime.

### 3. Examine signal strength  —  ✅ logic exists, runs on the panel
`src/factors/evaluate.py` — IC utilities. Single-signal IC baseline + **sign
sanity** (value/momentum positive, high-vol negative), IC IR, quantile spread.
Run it on the *warehouse-built* panel as the first validation gate of Step 2.

### 4. Modeling  —  ✅ logic exists (source-agnostic)
`src/models/walkforward.py` + `rankers.py` — expanding-window CV; **ElasticNet
baseline → LightGBM/XGBoost** (regression, then `lambdarank`). Cross-sectional
ranker. DataFrame-in/out, so no rewrite of model code — only the panel feeding it
changes.

### 5. Model validation  —  ✅ logic exists (part of the harness)
Walk-forward folds never overlap; **1-month embargo** enforced; **shuffled-target
run yields IC ≈ 0** (leakage test). PIT-integrity assertion: zero rows where a
fundamental's `availability_date > observation_month`.

### 6. Backtesting  —  ✅ logic exists (source-agnostic)
- `src/portfolio/construct.py` — long-only top-N, sector-neutral, position caps.
- `src/backtest/{engine,metrics,attribution}.py` — cost-aware sim; Sharpe /
  Sortino / max-DD / turnover; **FF5+MOM-residual alpha** attribution.
Re-wire to read warehouse predictions; write a `portfolios` book.

---

## Next phase — concrete build sequence (the critical path)

Maps to `architecture.md` Phase 3–4. Each item = `src/` work + a thin
notebook/validation + a Dagster asset (warehouse-native, gold parquet export for
fast training reads).

1. **Fundamentals-derivation bridge** (`src/factors/fundamentals_features.py`) —
   `fundamental_facts → quarterly TTM ratios + availability_date`. Unit-tested
   against a hand-checked ticker. *Unblocks all downstream.*
2. **`panel_monthly` asset** — re-wire `assemble_panel` to read the warehouse
   (universe/prices/macro/ff + the bridge output). Write panel to gold parquet
   export (+ optional `panel` table). PIT-integrity assert.
3. **Signal-strength gate** — run `evaluate.py` single-signal IC on the new panel;
   confirm signs/magnitudes are sane before training. (Step 3.)
4. **`pred_30d` asset** — port `walkforward` inference → `predictions` hypertable
   `(date, ticker, horizon, model_version)`. Include the **shuffle leakage test**
   as an asset check / CI gate. (Steps 4–5.)
5. **`selection_portfolio` asset** — `construct` top-N sector-neutral → `portfolios`
   table. (Step 6.)
6. **Backtest report** — cost-aware sim + FF5+MOM attribution over the new store;
   `notebooks/05_backtest_report.ipynb` as the thin client.
7. **Later — Timing track** (`architecture.md` Phase 3 tail): `panel_daily` +
   `pred_10d/5d` overlay (daily price/microstructure features). Ships *after* the
   30d selection core proves out. Embargo already sized for 30d.

### New tables to add (per `architecture.md` data model)
- `predictions` (hypertable) — `(date, ticker, horizon, model_version)` → OOS score.
- `portfolios` — `(date, ticker, strategy)` → target weight.

---

## Status at a glance

| Step | Logic | On warehouse? | Action |
|---|---|---|---|
| 1. Pull data | ✅ | ✅ | done (DQ checks live) |
| 2a. Fundamentals→ratios bridge | ⛔ | — | **build (linchpin)** |
| 2b. Panel assembly | ✅ | 🟡 parquet | re-wire to DB |
| 3. Signal strength (IC) | ✅ | 🟡 | run on new panel |
| 4. Modeling | ✅ | 🟡 | port to `pred_30d` asset |
| 5. Validation | ✅ | 🟡 | embargo + shuffle gate |
| 6. Backtest | ✅ | 🟡 | re-wire to predictions/portfolios |

## Out of scope (this phase)
Strategy A (ETF allocation), torch/deep models, RL allocation, paid PIT data
(survivorship fix), live monitoring/rebalancing. The data layer, walk-forward
harness, backtester, and attribution are built so all of these reuse them.

## Open decisions
- Persist `panel_monthly` to a Postgres table **and** gold parquet, or gold
  parquet only (training reads bulk)? *Leaning: gold parquet primary, table optional.*
- `predictions` model-versioning scheme (git SHA vs semantic tag).
- When to invest in the survivorship-bias fix (paid PIT source) vs ship v1 with
  the caveat.
