# Plan: Strategy B — Cross-Sectional Stock-Selection ML Pipeline

## Context

The repo currently has an ad-hoc data notebook (`notebooks/01_data_pipeline.ipynb`), a `data/` cache, and a `config/universe.yaml` that the user has judged not worth keeping. We're tearing those down and rebuilding the data layer properly as a **src-first library**, then building Strategy B end-to-end.

Strategy B = pick the stocks that will **out-rank** their peers over the next month, and hold a long-only, sector-neutral basket of the top names in a Roth IRA.

**Decisions locked with the user:**
- **Long-only** (Roth prohibits shorting) — we care most about ranking the *top* of the distribution correctly.
- **Target: 1-month forward return**, as a within-sector cross-sectional rank.
- **Sector-neutral** ranking/selection (within GICS sector).
- **Strategy B first**; Strategy A (ETF allocation) reuses the same infra later.
- **Universe: Russell 3000** (~3,000 names) — ~6× the breadth of the S&P 500 and includes the small-cap segment where factor premiums are strongest.

## Architecture (src-first, the workflow correction)

- **`src/` is the library** — all real logic is written here as functions/classes from the start, reviewed as normal code.
- **`notebooks/` are thin clients** — they `import` from `src/` for EDA, visualization, and reporting only. No reusable logic lives in a notebook.
- **`pipelines/` are production** — schedulable `.py` entry points that orchestrate `src/` functions end-to-end (cron-friendly). What a notebook *walks through*, a pipeline script *runs*.

```
src/
  data/      universe.py  prices.py  fundamentals.py  factors.py  cache.py
  factors/   panel.py  normalize.py  evaluate.py
  models/    walkforward.py  rankers.py
  portfolio/ construct.py
  backtest/  engine.py  metrics.py  attribution.py
pipelines/   build_dataset.py  train.py  backtest.py
notebooks/   thin EDA/reporting clients that import from src/
config/      data.yaml  features.yaml  model.yaml
data/        raw/  processed/   (regenerated; gitignored)
```

`CLAUDE.md`'s "Notebook-First Development" section is **rewritten** to this library-first model as part of step 0.

## Problem Formulation (unchanged — the part that must stay right)

- **Unit:** panel keyed by `(ticker, month)`.
- **Features `X[(i,t)]`:** point-in-time signals known at `t`, **cross-sectionally rank/z-scored within each date**, sector-neutralized.
- **Target `y[(i,t)]`:** 1-month forward return, as a within-sector cross-sectional rank.
- **Evaluate on Information Coefficient** (per-date Spearman rank corr of preds vs. realized), its IR (`mean(IC)/std(IC)`), and the **long-only top-quantile spread over a benchmark** — never RMSE on return levels.
- **Walk-forward validation** (expanding window) with a 1-month **embargo**; never random K-fold.
- **Integrity:** PIT features only; fundamentals lagged by fiscal-period-end + ~3 months; `.shift(1)` discipline on all features.

## Universe & Data Layer (`src/data/`)

- **`universe.py`** — fetch Russell 3000 constituents from the iShares **IWV** holdings CSV; apply **liquidity filters** (min price ≥ $5, min 60-day avg dollar-volume threshold, common-stock only — drop ADRs/ETFs/units). Emit the point-in-time tradeable list per rebalance date.
- **`prices.py`** — chunked monthly price/volume download (reuse the chunked-download pattern), cached to parquet.
- **`fundamentals.py`** — historical **quarterly** financials (`yf.Ticker.quarterly_financials/_balance_sheet/_cashflow`); compute valuation/quality/growth ratios; align each to its **filing-availability date**. This is the slow path (~3,000 tickers ≈ 30–45 min, rate-limited) — fetch once, cache aggressively, support incremental refresh.
  - **Why quarterly, not monthly:** companies only file financials quarterly (10-Q) / annually (10-K) — monthly fundamentals do not exist at the source. In the monthly panel, each fundamental is **forward-filled from its filing-availability date** until the next filing supersedes it, so a `(ticker, month)` row always holds the most recent value publicly known as of that month. Price-derived features (momentum, vol, size) update monthly; fundamentals update on filing cadence. This mixed-cadence join is the normal structure for equity factor panels.
- **`factors.py`** — FF5+MOM (direct Ken French CSV download) and FRED macro series (yield curve, VIX, HY spread).
- **`cache.py`** — parquet read/write helpers; `data/` is regenerated and gitignored.

**Known limitations (documented, not blocking v1):**
- Fundamental history is ~5y (yfinance limit) regardless of universe size — expanding helps *breadth*, not fundamental *time depth*. Price/technical features still get full history.
- Survivorship bias worsens with a current-membership universe; true fix (delisted names) needs a paid PIT source (Sharadar/EOD/Norgate). v1 accepts it with the caveat stated in every result.
- Two feature tiers: train v1 on the recent window where all features exist (~3,000 × ~60 months ≈ 180k rows); a price-only long-history track is a later option.

## Build Sequence

Each stage = `src/` module(s) + a thin notebook (EDA/validation) + a pipeline entry point where it makes sense.

| Stage | `src/` work | Thin client / pipeline |
|---|---|---|
| **0. Teardown & rework** | Delete `notebooks/01_data_pipeline.ipynb`, `data/`, `config/`. Rewrite `CLAUDE.md` workflow section. Add new `config/*.yaml`. | — |
| **1. Data layer** | `src/data/{universe,prices,fundamentals,factors,cache}.py` | `pipelines/build_dataset.py`; `notebooks/01_data_eda.ipynb` |
| **2. Feature panel** | `src/factors/{panel,normalize}.py` — assemble `(ticker,month)` panel, PIT join + forward-fill, cross-sectional rank/z-score, sector-neutralize, build 1-month target | `notebooks/02_panel_eda.ipynb` |
| **3. Signal analysis** | `src/factors/evaluate.py` — IC utilities | `notebooks/03_signal_ic.ipynb` (single-signal IC baseline + sign sanity) |
| **4. ML ranking** | `src/models/{walkforward,rankers.py}` — expanding-window CV w/ embargo; ElasticNet baseline → LightGBM/XGBoost (regression, then `lambdarank`) | `pipelines/train.py`; `notebooks/04_model_ic.ipynb` |
| **5. Backtest** | `src/portfolio/construct.py` (long-only top-N, sector-neutral, position caps); `src/backtest/{engine,metrics,attribution}.py` (cost-aware sim; Sharpe/Sortino/DD/turnover; FF5+MOM attribution) | `pipelines/backtest.py`; `notebooks/05_backtest_report.ipynb` |

The earlier idea of a standalone `02_ff5_regression` notebook is **repurposed** into `src/backtest/attribution.py` (used in stage 5 and later by Strategy A).

## Verification

- **Pipelines:** each `pipelines/*.py` runs clean to completion and writes expected parquet artifacts; notebooks execute top-to-bottom via `jupyter nbconvert --execute` with no cell errors.
- **PIT integrity:** assert zero rows where a fundamental's `availability_date > observation_month`; spot-check one ticker/month by hand.
- **Leakage test:** walk-forward folds never overlap; embargo enforced; a **shuffled-target** run yields IC ≈ 0.
- **Signal sanity:** single-signal IC signs match theory (value/momentum positive, high-vol negative).
- **Strategy validity:** report IC mean/IR, top-quantile spread vs. benchmark, turnover, and FF5+MOM-**residual** alpha — all annotated with the survivorship caveat.

## Next Phase: Daily Multi-Horizon Panel + Timing Models

The monthly panel (`assemble_panel`, source="db") drives the **selection** model
(1-month forward, ≈ t+21 trading days). The next phase adds a **daily** panel and
**multiple prediction horizons** — t+5 / t+10 / t+30 day forward returns — so we
can train a timing overlay on top of selection (architecture.md Phase 3,
`pred_30d`/`pred_10d`/`pred_5d`).

Design decisions (locked):
- **New `panel_daily`, separate from the monthly panel** — t+5/t+10 need daily
  resolution; do **not** retrofit the monthly `02` panel. ~21× the rows
  (3,976 × daily). Price/microstructure features recompute daily; fundamentals
  still forward-fill by `availability_date` (filing cadence unchanged).
- **One panel, N target columns** (`target_5d, target_10d, target_30d`), not N
  panels — features are shared across horizons; only the target + embargo differ.
  Parameterize: `build_target(returns, horizon)` and
  `assemble_panel(horizons=[5,10,30])`.
- **Embargo = the longest horizon (30d)** in the walk-forward harness, so an
  overlapping short-horizon label can't leak across folds into a longer one
  (`src/models/walkforward.py`). This is the key leakage trap.
- N models train on the same panel, each selecting its target column.

Build it via the **promotion path** (CLAUDE.md): prototype as a `panel_daily`
notebook over `src/` → user review → fix → productionize. Long-term target: a
`panel_daily` Dagster asset (gold table + checks), same pattern as
`fundamental_features`.

## Out of Scope (this phase)

Strategy A (ETF allocation), torch/deep models, RL, live monitoring/rebalancing, paid PIT data. The data layer, walk-forward harness, backtester, and attribution built here are designed for all of them to reuse.
