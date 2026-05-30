-- Factor-investment warehouse schema (Postgres + TimescaleDB).
-- Runs once on first container init (mounted into docker-entrypoint-initdb.d).
--
-- Design: silver layer. Raw XBRL facts are stored with FULL restatement history
-- (every filing), so any point-in-time / as-of query is correct even when a
-- company restates a prior period. Derived metrics (TTM, ROE, accruals, growth)
-- are NOT stored here — they are computed in the feature layer with as-of logic.

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
    yield_curve  double precision,
    vix          double precision,
    hy_spread    double precision
);

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
