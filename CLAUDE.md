# Factor Investment System

## Project Overview
Factor analysis system for a Roth IRA — combining ETF-based (A) and individual stock (B) strategies, with ML-enhanced factor models layered on top of classical Fama-French analysis.

**Strategy B (stock selection) is built** — see `plans/modeling.md`. The end-to-end
pipeline (all logic in `src/`, schedulable via `pipelines/`, demoed in `notebooks/`):

```
pipelines/build_dataset.py  → universe + prices + EDGAR PIT fundamentals + FF/macro
   (src/data)                 cached to data/processed/
pipelines/train.py          → assemble panel (src/factors) → walk-forward ML
   (src/models)               (src/models) → OOS predictions  [--shuffle leakage test]
pipelines/backtest.py       → long-only top-N portfolio (src/portfolio) → cost-aware
   (src/backtest)             backtest + FF5/MOM attribution

Run full universe: python -m pipelines.build_dataset   (drop --quick)
                   python -m pipelines.train --model lightgbm
                   python -m pipelines.backtest
```

## Working Principles

### Library-First Development
- **`src/` is the library** — all real logic is written here as clean functions/classes from the start, reviewed as normal code. This is the source of truth.
- **`notebooks/` are thin clients** — they `import` from `src/` for EDA, visualization, and reporting only. No reusable logic ever lives in a notebook; if a notebook cell grows logic worth keeping, it belongs in `src/`.
- **`pipelines/` are production** — schedulable `.py` entry points that orchestrate `src/` functions end-to-end (cron-friendly). What a notebook *walks through* interactively, a pipeline script *runs* unattended. A notebook's job is exploration/reporting; a pipeline's job is repeatable execution.

### Code Structure
```
factor-investment/
├── src/                # THE LIBRARY — all logic lives here
│   ├── data/           # universe, prices, fundamentals, factors, cache
│   ├── factors/        # panel assembly, normalization, IC evaluation
│   ├── models/         # walk-forward harness, model wrappers
│   ├── portfolio/      # long-only construction
│   ├── backtest/       # cost-aware sim, metrics, FF attribution
│   └── monitor/        # drift tracking & rebalancing alerts
├── pipelines/          # schedulable entry points (build_dataset, train, backtest)
├── notebooks/          # thin EDA/reporting clients that import from src/
├── data/
│   ├── raw/            # downloaded source data (gitignored)
│   └── processed/      # cleaned, aligned, feature-engineered (gitignored)
├── config/             # data.yaml, features.yaml, model.yaml
└── plans/              # design docs (e.g., modeling.md)
```

### Development Flow
1. Write/extend the logic as functions in the appropriate `src/` module
2. Use a notebook to `import` and exercise it — explore, visualize, validate
3. For repeatable runs, wire the `src/` calls into a `pipelines/*.py` entry point
4. Notebooks never hold logic worth keeping — they are demos/reports over `src/`

### Promotion Path (notebook → review → production)
Each stage graduates through three gates. **Do not skip ahead to productionizing
before the notebook has been reviewed.**

1. **Prototype (Claude writes).** Implement the logic as functions in `src/`, then
   build/extend a thin `notebooks/NN_*.ipynb` that imports and exercises it —
   reads inputs from the **warehouse** (`source="db"`), shows the EDA/validation
   that proves it works, and writes its output artifact for the next stage.
2. **Review (user reviews).** The user reads the notebook and its outputs. This is
   the human gate — surface assumptions, data-quality caveats, and design choices
   here, not after productionizing.
3. **Fix.** Resolve outstanding issues found in review (logic, leakage, coverage,
   edge cases). Iterate on `src/` + notebook until the user signs off.
4. **Productionize.** Wire the reviewed `src/` calls into a schedulable
   `pipelines/*.py` entry point. **Long-term target: every stage becomes a Dagster
   asset in `orchestration/`** — materialized into a warehouse table (gold layer)
   with asset checks, the same pattern as `fundamental_features`. The notebook
   remains as the living EDA/report over that stage; the asset is what runs
   unattended. Deploy per [[nas-deploy-ordering]] (push before deploy).

Rule of thumb: **notebook = the reviewable prototype + report; pipeline/asset =
the productionized, scheduled, checked version of the same `src/` logic.**

### Modeling: sklearn components for standard ops; inference-time parity
Any transform that **fits parameters on the training data and is applied per
sample** — imputation fills, scaler means/σ, encoder vocabularies, binning edges
— must use the corresponding **sklearn component** (`SimpleImputer`,
`StandardScaler`, `OneHotEncoder`, …) **bundled into the estimator via an sklearn
`Pipeline`** (`ColumnTransformer` for per-column ops). Never hand-roll these as
ad-hoc DataFrame calls (`df.fillna(...)`, `pd.get_dummies`, inline z-scores) in
the training code.

**Why:** the *fitted* transform is then serialized *inside the model artifact* and
replayed identically at inference — training-serving skew is eliminated by
construction. `Pipeline` works with LightGBM/XGBoost (both sklearn-compatible).
The factories in `src/models/rankers.py` return such pipelines; training and
inference pass **raw** feature columns. See `plans/architecture.md`
"Training vs. Inference".

**Three categories — know which bucket a step is in:**
1. **Fitted, per-sample preprocessing** (imputation, scaling, encoding) → **in the
   model `Pipeline`** (travels in the artifact). It learns from the training
   distribution, so it *must* be carried to serve.
2. **Stateless, cross-sectional normalization** — our rank / z-score **within each
   date** (`src/factors/normalize.py`). It fits *nothing* from history (a stock's
   rank comes only from that date's cross-section) and is a cross-row groupby, not
   a per-sample op — so it does **not** go in the Pipeline. Parity instead comes
   from living in the single shared `assemble_panel` path that both training and
   inference call. ⚠️ If normalization ever switches to a statistic fitted on the
   *training set* (global z-score, fitted quantile bins), it becomes category 1 and
   must move into the artifact.
3. **Domain feature engineering** — a `fillna(0)` encoding a real fact (missing
   debt component = 0 in an EV sum; missing monthly return = 0 contribution).
   Correct domain logic; stays in `src/factors` / `src/backtest`.

The line between 1 and 3: if a step *normalizes/learns the feature distribution*
for the model it is preprocessing (Pipeline); if it *constructs the feature's
economic value* it is feature engineering.

## Tech Stack

### Active
- **Data**: `yfinance` (prices/fundamentals), direct HTTP fetch from Ken French library + FRED CSV API (no pandas-datareader)
- **Core**: `pandas`, `numpy`, `scipy`
- **ML**: `scikit-learn`, `xgboost`, `lightgbm`
- **Optimization**: `cvxpy`, `PyPortfolioOpt`
- **Viz**: `plotly`, `matplotlib`

### Future (needs more data + compute)
- `torch` — LSTM, Transformer, RL-based allocation

## Data Sources

### Canonical (always use)
- **Ken French Data Library**: Fama-French 3/5 factors + momentum — free, canonical benchmark
- **iShares IWV holdings CSV**: Russell 3000 constituents (the investable universe), with liquidity filters applied
- **yfinance**: Price/volume history (full) + **historical quarterly** financials (`quarterly_financials/_balance_sheet/_cashflow`, ~5y) used point-in-time via filing-availability lag
- **FRED CSV API**: Macro series (yield curve, credit spreads) — free, no key needed

### Future: Paid Sources (add when backtesting engine needs point-in-time data)
- **Tiingo** (~$10/mo): historical EOD prices + daily fundamentals snapshots — fixes point-in-time bias for backtesting
- **Polygon.io** (~$29/mo Starter): quarterly financial statements (10yr history), news sentiment with ticker tags

### Fundamental & Quality Signals
- Point-in-time financials (P/B, P/E, EV/EBITDA, ROE, ROA, gross margins)
- Asset growth, accruals, capex intensity
- Earnings surprise (SUE — Standardized Unexpected Earnings)
- Analyst estimate revision direction and magnitude

### Market Microstructure
- Relative volume (vs. 30/90-day average) — liquidity proxy
- Short interest ratio / days to cover — crowding & mean-reversion signal
- Bid-ask spread (for transaction cost modeling)

### Macro & Regime
- Yield curve slope (10y–2y spread) — regime signal
- Credit spreads (HY–IG) — risk appetite
- VIX level + VIX term structure — volatility regime
- Fed funds rate / rate of change

### Sentiment & Alternative
- Insider buying/selling (SEC Form 4 filings — free via SEC EDGAR)
- Share buyback announcements
- News/earnings call sentiment (NLP on 10-K/10-Q via SEC EDGAR)

## Key Constraints & Caveats
- **No lookahead bias**: all features must use only data available at prediction time (point-in-time)
- **Survivorship bias**: backtests must include delisted stocks; use CRSP or equivalent where possible
- **Factor zoo**: with many features, use regularization (Lasso, Ridge, ElasticNet) and walk-forward validation — never in-sample evaluation
- **Roth IRA advantage**: no capital gains tax on rebalancing → factor rotation strategies are viable without tax drag

## Data Lag & Temporal Alignment

### FF5 Factor Construction Lags
FF accounting-based factors (HML, RMW, CMA) are rebalanced every June using fiscal year-end data from the *prior* December — a **minimum 6-month lag** by design, ensuring data was publicly filed before portfolio formation. SMB uses June market cap (~0 lag). Momentum (MOM) uses 12-2 returns (t-12 to t-2), skipping the most recent month.

### Two Valid Use Cases
- **Factor loading regression** (attribution): regress `returns[t]` on `factors[t]` — both are realized, no lookahead. Purely descriptive; used in `src/backtest/attribution.py`.
- **Return prediction** (the ML pipeline): features for month `t` must use only data known as of `t`; the target is the forward return over `(t, t+1]`. No feature may peek at `t+1`.

### Fundamental Signal Lag
We use yfinance **historical quarterly** statements (not the live snapshot). Each fundamental is stamped with a **filing-availability date** (fiscal period-end + ~3-month lag) and forward-filled into the monthly panel from that date until the next filing supersedes it. A `(ticker, month)` row therefore holds only values publicly known as of that month. Using a fiscal period's data before its availability date is lookahead bias and will inflate backtest results.

### Rule: `.shift(1)` is mandatory on all feature DataFrames in any predictive model.
