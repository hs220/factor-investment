-- Factor-investment warehouse schema (Postgres + TimescaleDB).
-- Runs once on first container init (mounted into docker-entrypoint-initdb.d).
--
-- Design: medallion layers. ``fundamental_facts`` is the SILVER layer — raw XBRL
-- facts with FULL restatement history (every filing), so any point-in-time /
-- as-of query is correct even when a company restates a prior period.
-- ``fundamental_features`` is the GOLD/feature layer — the derived per-quarter
-- metrics (TTM, ROE, accruals, growth) rolled up from the raw facts with as-of
-- logic and stamped with an availability_date; it is MATERIALIZED (refreshed by
-- the fundamental_features Dagster asset) rather than recomputed on every read.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------------
-- Universe: current investable names + sector (one row per ticker).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS universe (
    ticker       text PRIMARY KEY,
    name         text,
    exchange     text,
    sic          text,          -- raw SEC SIC code (re-map sectors without re-fetch)
    gics_sector  text,
    first_seen   date,
    last_seen    date,
    is_active    boolean NOT NULL DEFAULT true,
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Prices: daily close/volume (Timescale hypertable, partitioned by date).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices (
    ticker  text             NOT NULL,
    date    date             NOT NULL,
    close   double precision,
    volume  double precision,
    PRIMARY KEY (ticker, date)
);
SELECT create_hypertable('prices', 'date',
                         chunk_time_interval => INTERVAL '1 year',
                         if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices (date);

-- ---------------------------------------------------------------------------
-- Fundamental facts: RAW XBRL values with full restatement history.
-- One row per (ticker, concept, period_end, filed_date): a later filing that
-- restates an earlier period inserts a NEW row (new filed_date), so the prior
-- value is preserved. As-of query = latest filed_date <= as_of per
-- (ticker, concept, period_end).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fundamental_facts (
    ticker         text             NOT NULL,
    cik            text             NOT NULL,
    concept        text             NOT NULL,   -- normalized field (e.g. revenue, equity)
    gaap_tag       text,                        -- original us-gaap concept name
    period_end     date             NOT NULL,
    fiscal_period  text,                        -- Q1..Q4 / FY
    duration_days  integer,                     -- ~91 quarter, ~365 annual, 0 instant
    filed_date     date             NOT NULL,   -- availability date (PIT)
    form           text,                        -- 10-Q / 10-K
    value          double precision,
    unit           text,
    PRIMARY KEY (ticker, concept, period_end, filed_date)
);
CREATE INDEX IF NOT EXISTS idx_ff_asof
    ON fundamental_facts (ticker, concept, period_end, filed_date DESC);
CREATE INDEX IF NOT EXISTS idx_ff_filed ON fundamental_facts (filed_date);

-- ---------------------------------------------------------------------------
-- Fundamental features: GOLD layer. One per-quarter row per (ticker, period_end)
-- derived from fundamental_facts (TTM flows, balance-sheet levels, ratios),
-- stamped with availability_date for the panel's point-in-time as-of join.
-- Materialized by the fundamental_features Dagster asset.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fundamental_features (
    ticker              text NOT NULL,
    period_end          date NOT NULL,
    availability_date   date,            -- when the whole row became public (PIT)
    gics_sector         text,
    revenue_ttm         double precision,
    net_income_ttm      double precision,
    gross_profit_ttm    double precision,
    op_cashflow_ttm     double precision,
    ebitda_ttm          double precision,
    equity              double precision,
    assets              double precision,
    debt                double precision,
    cash                double precision,
    shares              double precision,
    roe                 double precision,
    gross_margin        double precision,
    profit_margin       double precision,
    accruals            double precision,
    revenue_growth_yoy  double precision,
    asset_growth_yoy    double precision,
    PRIMARY KEY (ticker, period_end)
);
CREATE INDEX IF NOT EXISTS idx_ffeat_asof
    ON fundamental_features (ticker, availability_date);

-- ---------------------------------------------------------------------------
-- Fama-French 5 + momentum (monthly) and macro regime series (monthly).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ff_factors (
    date    date PRIMARY KEY,
    mkt_rf  double precision,
    smb     double precision,
    hml     double precision,
    rmw     double precision,
    cma     double precision,
    rf      double precision,
    mom     double precision
);

CREATE TABLE IF NOT EXISTS macro (
    date         date PRIMARY KEY,
    yield_curve   double precision,
    vix           double precision,
    credit_spread double precision   -- Moody's Baa - 10y Treasury (BAA10Y)
);

-- ---------------------------------------------------------------------------
-- panel_monthly: GOLD training matrix. One row per (ticker, month-end) with the
-- normalized, sector-neutral feature set + forward-return target. Materialized
-- by the panel_monthly Dagster asset from prices + fundamental_features + macro
-- + universe sectors (src/factors/panel.assemble_panel). The model stage reads
-- this directly instead of recomputing the panel.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS panel_monthly (
    date                date             NOT NULL,
    ticker              text             NOT NULL,
    gics_sector         text,
    forward_return      double precision,
    market_cap          double precision,
    -- price / technical
    momentum_12_2       double precision,
    momentum_6_1        double precision,
    volatility_12m      double precision,
    size_log_mktcap     double precision,
    -- value
    earnings_yield      double precision,
    book_to_price       double precision,
    ev_ebitda_inv       double precision,
    -- quality
    roe                 double precision,
    gross_margin        double precision,
    profit_margin       double precision,
    accruals            double precision,
    -- growth
    revenue_growth_yoy  double precision,
    asset_growth_yoy    double precision,
    -- macro regime
    yield_curve         double precision,
    vix                 double precision,
    credit_spread       double precision,
    -- target: within-sector forward-return rank
    target              double precision,
    PRIMARY KEY (date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_panel_date ON panel_monthly (date);

-- ---------------------------------------------------------------------------
-- Model outputs: OOS predictions per horizon (hypertable) and portfolios.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    date           date             NOT NULL,
    ticker         text             NOT NULL,
    horizon        text             NOT NULL,   -- e.g. 5d / 10d / 30d
    model_version  text             NOT NULL,
    score          double precision,
    PRIMARY KEY (date, ticker, horizon, model_version)
);
SELECT create_hypertable('predictions', 'date',
                         chunk_time_interval => INTERVAL '1 year',
                         if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS portfolios (
    date      date             NOT NULL,
    ticker    text             NOT NULL,
    strategy  text             NOT NULL,   -- selection / timing / combined
    weight    double precision,
    PRIMARY KEY (date, ticker, strategy)
);
