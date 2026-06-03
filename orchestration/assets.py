"""Dagster ingestion assets — fetch from source, write to the Postgres warehouse.

Each asset is independently materializable and reads its upstream inputs from the
DB where possible, so "rerun just fundamentals" or "rerun just prices" is one
click. Assets are incremental where the source supports it (prices extend from
the last stored month; fundamentals re-pull only tickers with a new filing).
"""
from __future__ import annotations

import pandas as pd
from dagster import Backoff, RetryPolicy, asset

from src.config import load_config
from src.data import db, factors, fundamentals, prices, universe, warehouse

# Transient external sources (FRED/yfinance/EDGAR) self-heal via step retry.
_RETRY = RetryPolicy(max_retries=3, delay=30, backoff=Backoff.EXPONENTIAL)


def _date_range() -> tuple[str, str]:
    start = load_config("data")["prices"]["start_date"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    return start, end


@asset(group_name="ingest", compute_kind="edgar", retry_policy=_RETRY)
def universe_table(context) -> None:
    """Current US-listed common stocks (names + exchange) -> universe table."""
    uni = universe.fetch_us_listed().reset_index()
    n = db.load_universe(uni)
    context.add_output_metadata({"rows": n})


@asset(group_name="ingest", deps=[universe_table], compute_kind="yfinance", retry_policy=_RETRY)
def prices_table(context) -> None:
    """Incremental monthly prices + liquidity filter -> prices/universe tables."""
    start, end = _date_range()
    tickers = db.read_sql("SELECT ticker FROM universe ORDER BY ticker")["ticker"].tolist()

    last = db.read_sql("SELECT MAX(date) AS d FROM prices")["d"].iloc[0]
    if last is not None:
        start = (pd.Timestamp(last) - pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d")

    close = prices.download_prices(tickers, start, end, field="Close")
    volume = prices.download_prices(tickers, start, end, field="Volume")
    adv = prices.avg_dollar_volume(close, volume.reindex_like(close))
    tradeable = universe.apply_liquidity_filters(
        pd.DataFrame(index=close.columns), dollar_volume=adv
    ).index.tolist()
    keep = [t for t in tradeable if t in close.columns]

    monthly_close = prices.to_monthly_close(close[keep])
    n = db.load_prices_wide(monthly_close)
    # Record the investable set (liquidity filter) -> universe.is_active.
    db.set_active(keep)
    context.add_output_metadata({"tickers": len(keep), "rows": n})


@asset(group_name="ingest", deps=[universe_table], compute_kind="edgar", retry_policy=_RETRY)
def fundamental_facts(context) -> None:
    """EDGAR raw facts (restatement history), incremental on new filings.

    Scopes to the investable set via is_active (persisted by the prior prices
    run) — no hard dependency on prices_table, so in the daily job the two run
    in parallel. A name that newly crosses the liquidity threshold simply gets
    its fundamentals on the next run.
    """
    tickers = db.read_sql(
        "SELECT ticker FROM universe WHERE is_active ORDER BY ticker"
    )["ticker"].tolist()

    existing = db.read_sql(
        "SELECT ticker, MAX(filed_date) AS f FROM fundamental_facts GROUP BY ticker"
    )
    existing_latest = (
        dict(zip(existing["ticker"], pd.to_datetime(existing["f"])))
        if not existing.empty
        else None
    )

    facts = fundamentals.fetch_fundamentals(tickers, existing_latest=existing_latest)
    n = db.load_fundamental_facts(facts) if not facts.empty else 0
    context.add_output_metadata({"rows": n, "tickers": int(facts["ticker"].nunique()) if n else 0})


@asset(group_name="ingest", deps=[universe_table], compute_kind="edgar", retry_policy=_RETRY)
def sectors(context) -> None:
    """GICS sector (SEC SIC, yfinance fallback) -> universe.gics_sector.

    Independent of prices: SIC classification has nothing to do with liquidity.
    Scopes to the last-known investable set via is_active (persisted by the
    prior prices run), so it needs no hard dependency on prices_table. Names
    EDGAR leaves with a blank SIC (banks, BDCs/closed-end funds, some foreign
    issuers) fall back to yfinance's sector so they aren't left NULL.
    """
    tickers = db.read_sql(
        "SELECT ticker FROM universe WHERE is_active ORDER BY ticker"
    )["ticker"].tolist()
    sec = fundamentals.fetch_sectors(tickers)
    # Store raw sic too, so future mapping changes are an instant re-map
    # (scripts/remap_sectors.py) with no EDGAR re-fetch.
    n = db.upsert(sec[["ticker", "sic", "gics_sector"]], "universe", ["ticker"]) if not sec.empty else 0
    src = sec["sector_source"].value_counts().to_dict() if not sec.empty else {}
    context.add_output_metadata({
        "updated": n,
        "with_sector": int(sec["gics_sector"].notna().sum()),
        "via_sec_sic": int(src.get("sec_sic", 0)),
        "via_yfinance": int(src.get("yfinance", 0)),
        "unmapped": int(sec["gics_sector"].isna().sum()),
    })


@asset(group_name="ingest", deps=[fundamental_facts, sectors], compute_kind="pandas")
def fundamental_features(context) -> None:
    """Derive quarterly TTM/ratio features from raw facts -> fundamental_features.

    Gold/feature layer: rolls the silver ``fundamental_facts`` (raw, restatement-
    versioned) up into one per-quarter row per (ticker, period_end) with TTM
    flows, balance-sheet levels, ratios, and an availability_date for the panel's
    point-in-time join. Pure pandas compute over the warehouse — no external call.
    """
    from src.factors.fundamentals_features import derive_fundamental_features

    facts = db.read_sql(
        "SELECT ticker, concept, period_end, filed_date, value, duration_days "
        "FROM fundamental_facts"
    )
    secs = db.read_sql("SELECT ticker, gics_sector FROM universe")
    feats = derive_fundamental_features(facts, secs)
    n = db.load_fundamental_features(feats)
    context.add_output_metadata(
        {"rows": n, "tickers": int(feats["ticker"].nunique()) if not feats.empty else 0}
    )


@asset(group_name="ingest", compute_kind="ken_french", retry_policy=_RETRY)
def ff_factors(context) -> None:
    """Fama-French 5 + momentum (monthly) -> ff_factors table."""
    start, end = _date_range()
    n = db.load_ff_factors(factors.load_factors(start, end))
    context.add_output_metadata({"rows": n})


@asset(group_name="ingest", compute_kind="fred", retry_policy=_RETRY)
def macro(context) -> None:
    """Macro regime series (yield curve, VIX, credit spread) -> macro table."""
    start, end = _date_range()
    n = db.load_macro(factors.load_macro(start, end))
    context.add_output_metadata({"rows": n})


@asset(
    group_name="features",
    deps=[prices_table, fundamental_features, macro, sectors],
    compute_kind="pandas",
)
def panel_monthly(context) -> None:
    """Assembled (ticker, month) feature panel -> panel_monthly (gold).

    The training matrix: PIT-joined fundamentals + price/technical + macro
    features, cross-sectionally rank-normalized within sector, plus the
    within-sector forward-return rank target. Pure pandas over the warehouse via
    src.factors.panel.assemble_panel(source="db") — the same code the notebook
    and model stage use, so there is one feature definition for train and serve.
    """
    from src.factors.panel import assemble_panel

    panel = assemble_panel(source="db")
    n = db.load_panel_monthly(panel)
    context.add_output_metadata({
        "rows": n,
        "tickers": int(panel["ticker"].nunique()),
        "dates": int(panel["date"].nunique()),
        "sectors": int(panel["gics_sector"].nunique()),
    })


@asset(group_name="model", deps=[panel_monthly], compute_kind="lightgbm")
def model_predictions(context) -> None:
    """Walk-forward OOS scores -> predictions, + a deployment model artifact.

    Reads ``panel_monthly``, runs the (fixed-config) LightGBM walk-forward to
    produce out-of-sample scores, writes them to the ``predictions`` table keyed
    by the run's ``model_version``, and fits + persists a deployment model
    artifact (``models/1m/<version>/``) for inference. Same code path as
    ``pipelines/train.py`` (tuning off here — it's too heavy for the NAS; run a
    tuned retrain off-box). The OOS IC is surfaced as asset metadata.
    """
    from src.factors.panel import _feature_list
    from src.models.training import train_and_deploy

    panel = warehouse.load_panel_monthly()
    res = train_and_deploy(
        panel, _feature_list(), model_name="lightgbm", horizon="1m",
        tune=False, save_model=True,
    )
    if res.oos_preds.empty or res.manifest is None:
        raise Exception("training produced no predictions / no deployment model")

    version = res.manifest.model_version
    n = db.load_predictions(res.oos_preds, horizon="1m", model_version=version)
    context.add_output_metadata({
        "model_version": version,
        "predictions_rows": n,
        "oos_ic_mean": round(res.summary["ic_mean"], 4),
        "oos_ic_ir": round(res.summary["ic_ir"], 3),
        "oos_t_stat": round(res.summary["t_stat"], 2),
        "oos_months": int(res.summary["n_months"]),
    })
