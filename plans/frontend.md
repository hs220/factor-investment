# Plan: Streamlit Dashboard (data presentation layer)

## Goal
A personal, LAN-only dashboard to present the system's outputs: **what to hold**
(recommendations), **does it work** (performance/backtest), and **why** (signal/IC
analytics). Pipeline health links out to the existing Dagster UI rather than being
rebuilt.

## Decisions (locked with user)
- **Streamlit**, pure Python, **in this repo** (new top-level `app/`). It is "another
  thin client over `src/`" — same status as `notebooks/` and `pipelines/`.
- **No separate repo, no separate API.** Streamlit runs in-process and imports
  `src/` directly (warehouse readers, `predict_with_artifact`, backtest metrics,
  `evaluate`). Presentation (charts/layout) lives in `app/`; all computation stays
  in `src/` — if a page grows real logic, it moves to `src/`.
- v1 surfaces: **Recommendations, Performance, Signal/IC**. Pipeline = links to
  Dagster UI (`http://192.168.68.70:3030`).

## The principle (same as train/serve parity)
The UI never re-derives data. Recommendations come from the *same*
`predict_with_artifact`; performance from the *same* `src/backtest`; IC from the
*same* `src/factors/evaluate`. The dashboard reads the warehouse via
`src/data/{db,warehouse}.py` and the deployed model via `src/models/artifact.py`.

## Structure
```
app/
  Home.py                  # overview + links to Dagster UI / model card summary
  pages/
    1_Recommendations.py   # latest top-N from the deployed model + model card
    2_Performance.py       # backtest: returns vs benchmark, Sharpe/DD, attribution
    3_Signal_IC.py         # single-signal IC + cumulative IC + feature coverage
  lib/
    data.py                # cached loaders (st.cache_data/resource) over src/
config/  (reuse)           # db.yaml etc.
deploy/streamlit/          # Dockerfile + docker-compose.yml (NAS container)
requirements-app.txt       # streamlit + plotly + the src/ runtime deps
```

### Caching
- `@st.cache_resource`: the DB engine and the loaded model artifact (heavy, stable).
- `@st.cache_data(ttl=...)`: warehouse reads (`panel_monthly`, `predictions`,
  `prices`, `ff_factors`) and the backtest result — so interactions don't re-query.

## Pages (data sources, all via `src/`)
1. **Recommendations** — load the latest `panel_monthly` cross-section →
   `artifact.predict_with_artifact(horizon="1m")` → top-N by score (config
   `portfolio.n_holdings`), with sector + key feature ranks. Show the **model card**
   from the manifest: `model_version`, OOS IC/IR/t, train window, hyperparams.
   *(Reuses the exact logic in `pipelines/predict.py`.)*
2. **Performance** — read OOS scores from the `predictions` table → long-only
   top-N, sector-neutral construction (`src/portfolio/construct`) → cost-aware
   backtest (`src/backtest/engine,metrics`) → cumulative return vs SPY,
   Sharpe/Sortino/max-DD/turnover, and FF5+MOM **residual alpha**
   (`src/backtest/attribution`). Cached (the backtest is the heaviest call).
3. **Signal / IC** — `panel_monthly` → `evaluate.signal_report` (per-signal IC, IR,
   t, hit-rate) + cumulative IC chart + feature-coverage bars (notebook-03 content).

## Small `src/` extractions (keep pages thin)
- `src/serving/recommend.py::latest_recommendations(horizon, top_n)` — load latest
  cross-section + score; shared by `pipelines/predict.py` and the Recommendations
  page (so the CLI and UI return identical rows).
- `src/backtest/run.py::run_backtest(predictions, ...) -> results` — one entry that
  notebook 05, `pipelines/backtest.py`, and the Performance page all call (if not
  already factored this way).

## Deployment
- A **Streamlit container on the NAS** (`deploy/streamlit/`), same pattern as
  Dagster: shares the repo image (Python + `src/` + `app/`), connects to Postgres
  over the LAN (`POSTGRES_PASSWORD`/`FACTOR_DB_HOST` env), serves on `:8501`.
- **Mount the `factor-models` volume read-only** so the Recommendations page can
  load the deployment artifact the `model_predictions` asset wrote.
- Deploy via a `deploy/deploy_streamlit.sh` (mirror `deploy_dagster.sh`,
  push-before-deploy per [[nas-deploy-ordering]]).
- **Auth:** LAN-only for v1 (no auth). If ever exposed: front with the NAS reverse
  proxy / a Tailscale tunnel + basic auth — not before.

## Build sequence
| Step | Work |
|---|---|
| 0 | Scaffold `app/` + `lib/data.py` cached loaders + `requirements-app.txt`; `Home.py` with Dagster links + a one-line model-card summary. Run locally (`streamlit run`). |
| 1 | **Recommendations** page (+ extract `src/serving/recommend.py`). The quickest win — depends only on `panel_monthly` + the artifact. |
| 2 | **Signal/IC** page (reuses `evaluate.signal_report`; cheap). |
| 3 | **Performance** page (+ factor `run_backtest`); cache the backtest. |
| 4 | `deploy/streamlit/` container + `deploy_streamlit.sh`; deploy to NAS, mount `factor-models` ro; verify LAN access. |

## Risks / watch-items
- **Backtest cost on page load** → cache; longer-term, persist backtest results to
  the warehouse (a `backtest` asset) and have the page read them. Ties to the
  pending predictions/backtest-DB-sink follow-up.
- **Artifact availability** → the dashboard depends on a deployed model existing;
  handle "no artifact yet" gracefully (show a message + link to trigger
  `model_train`).
- **Heavy deps in the image** — the app needs lightgbm (to load/predict the
  pipeline) + plotly; reuse the orchestration image base or a shared requirements.

## Out of scope (v1)
Multi-user/auth, internet exposure, order execution / broker integration, editing
config from the UI, real-time data. Read-only presentation of what the warehouse +
deployed model already produce.
