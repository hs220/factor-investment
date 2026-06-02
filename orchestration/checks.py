"""Dagster asset checks — data-quality validation on the warehouse.

Each check runs after its asset materializes (and on demand), querying Postgres
and surfacing pass/fail in the lineage UI. Severity:
- ERROR: integrity violations that must block (lookahead, negative prices).
- WARN:  quality signals that shouldn't fail the pipeline (coverage dips).

These replace the ad-hoc asserts/guards scattered through the code with visible,
historised checks.
"""
from __future__ import annotations

import pandas as pd
from dagster import AssetCheckResult, AssetCheckSeverity, asset_check

from orchestration import assets
from src.data import db

_EXPECTED_CONCEPTS = 10  # of 12 normalized concepts; some filers lack a few


def _count(sql: str) -> int:
    return int(db.read_sql(sql)["c"].iloc[0])


# --------------------------------------------------------------------------- #
# universe
# --------------------------------------------------------------------------- #
@asset_check(asset=assets.universe_table, description="is_active count in a sane band")
def universe_active_band() -> AssetCheckResult:
    n = _count("SELECT count(*) c FROM universe WHERE is_active")
    return AssetCheckResult(
        passed=2000 <= n <= 5000,
        severity=AssetCheckSeverity.WARN,
        metadata={"investable": n},
    )


@asset_check(asset=assets.sectors, description="sector coverage of investable names")
def universe_sector_coverage() -> AssetCheckResult:
    active = _count("SELECT count(*) c FROM universe WHERE is_active")
    with_sec = _count(
        "SELECT count(*) c FROM universe WHERE is_active AND gics_sector IS NOT NULL"
    )
    frac = with_sec / active if active else 0.0
    return AssetCheckResult(
        passed=frac >= 0.90,
        severity=AssetCheckSeverity.WARN,
        metadata={"coverage_pct": round(frac * 100, 1), "with_sector": with_sec,
                  "investable": active},
    )


# --------------------------------------------------------------------------- #
# prices
# --------------------------------------------------------------------------- #
@asset_check(asset=assets.prices_table, description="every stored close is present and positive")
def prices_positive() -> AssetCheckResult:
    # NaN must be matched explicitly: in Postgres NaN sorts greater than all
    # numbers, so ``close <= 0`` is FALSE for NaN — the old check was blind to it
    # while ~40% of rows were NaN grid-padding. ``close = 'NaN'`` matches (Postgres
    # treats NaN = NaN as true).
    bad = _count(
        "SELECT count(*) c FROM prices "
        "WHERE close IS NULL OR close = 'NaN' OR close <= 0"
    )
    return AssetCheckResult(
        passed=bad == 0, severity=AssetCheckSeverity.ERROR, metadata={"bad_rows": bad}
    )


@asset_check(asset=assets.prices_table, description="prices are fresh (recent month-end)")
def prices_freshness() -> AssetCheckResult:
    last = db.read_sql("SELECT MAX(date) AS c FROM prices")["c"].iloc[0]
    age = (pd.Timestamp.today().date() - last).days if last else 9999
    return AssetCheckResult(
        passed=age <= 45, severity=AssetCheckSeverity.WARN,
        metadata={"latest": str(last), "age_days": age},
    )


@asset_check(asset=assets.prices_table, description="row count not collapsed (partial load)")
def prices_row_count() -> AssetCheckResult:
    n = _count("SELECT count(*) c FROM prices")
    return AssetCheckResult(
        passed=n >= 500_000, severity=AssetCheckSeverity.ERROR, metadata={"rows": n}
    )


# --------------------------------------------------------------------------- #
# fundamental_facts
# --------------------------------------------------------------------------- #
_LOOKAHEAD_TOL_DAYS = 60  # tolerate calendar-quarter snap-forward (see below)


@asset_check(asset=assets.fundamental_facts,
             description="point-in-time: no filing dated >60d before its period_end")
def fundamentals_no_lookahead() -> AssetCheckResult:
    """Hard gate against genuine lookahead, tolerant of the snap-forward artifact.

    The true PIT guarantee is on ``filed_date`` (the panel join gates on it). A
    naive ``filed_date < period_end`` proxy over-fires because ``_snap_q`` snaps
    an off-calendar fiscal quarter-end (or a dei cover-page ``shares`` as-of-filing
    date) to the *nearest* calendar quarter-end, which can land a few weeks past
    the filing — a relabeling, not lookahead (all such cases are <=~46 days).

    Real lookahead bugs (forecast / debt-maturity facts) sit 100-2000+ days
    before their period, so a 60-day tolerance keeps the canary while ignoring the
    benign snap-forward. ``snap_forward_rows`` stays in metadata for visibility.
    """
    bad = _count(
        f"SELECT count(*) c FROM fundamental_facts "
        f"WHERE period_end - filed_date > {_LOOKAHEAD_TOL_DAYS}"
    )
    snap_forward = _count(
        "SELECT count(*) c FROM fundamental_facts WHERE filed_date < period_end"
    )
    return AssetCheckResult(
        passed=bad == 0, severity=AssetCheckSeverity.ERROR,
        metadata={"lookahead_rows": bad, "snap_forward_rows": snap_forward,
                  "tolerance_days": _LOOKAHEAD_TOL_DAYS},
    )


@asset_check(asset=assets.fundamental_facts, description="expected concept set present")
def fundamentals_concepts() -> AssetCheckResult:
    n = _count("SELECT count(DISTINCT concept) c FROM fundamental_facts")
    return AssetCheckResult(
        passed=n >= _EXPECTED_CONCEPTS, severity=AssetCheckSeverity.WARN,
        metadata={"distinct_concepts": n},
    )


@asset_check(asset=assets.fundamental_facts, description="no duplicate fact keys")
def fundamentals_unique_keys() -> AssetCheckResult:
    dups = _count(
        """SELECT count(*) c FROM (
             SELECT 1 FROM fundamental_facts
             GROUP BY ticker, concept, period_end, filed_date HAVING count(*) > 1
           ) t"""
    )
    return AssetCheckResult(
        passed=dups == 0, severity=AssetCheckSeverity.ERROR, metadata={"dup_keys": dups}
    )


# --------------------------------------------------------------------------- #
# fundamental_features (derived gold layer)
# --------------------------------------------------------------------------- #
@asset_check(asset=assets.fundamental_features,
             description="point-in-time: availability_date not >60d before period_end")
def fundamental_features_no_lookahead() -> AssetCheckResult:
    """Same PIT guard as the raw facts: availability stamps must not predate the
    reporting period (beyond the benign snap-forward tolerance)."""
    bad = _count(
        f"SELECT count(*) c FROM fundamental_features "
        f"WHERE period_end - availability_date > {_LOOKAHEAD_TOL_DAYS}"
    )
    return AssetCheckResult(
        passed=bad == 0, severity=AssetCheckSeverity.ERROR,
        metadata={"lookahead_rows": bad, "tolerance_days": _LOOKAHEAD_TOL_DAYS},
    )


@asset_check(asset=assets.fundamental_features, description="row count not collapsed (partial build)")
def fundamental_features_row_count() -> AssetCheckResult:
    n = _count("SELECT count(*) c FROM fundamental_features")
    return AssetCheckResult(
        passed=n >= 20_000, severity=AssetCheckSeverity.ERROR, metadata={"rows": n}
    )


@asset_check(asset=assets.fundamental_features, description="no duplicate (ticker, period_end)")
def fundamental_features_unique_keys() -> AssetCheckResult:
    dups = _count(
        """SELECT count(*) c FROM (
             SELECT 1 FROM fundamental_features
             GROUP BY ticker, period_end HAVING count(*) > 1
           ) t"""
    )
    return AssetCheckResult(
        passed=dups == 0, severity=AssetCheckSeverity.ERROR, metadata={"dup_keys": dups}
    )


# --------------------------------------------------------------------------- #
# ff_factors / macro
# --------------------------------------------------------------------------- #
@asset_check(asset=assets.ff_factors, description="FF factors fresh + non-null")
def ff_factors_fresh() -> AssetCheckResult:
    row = db.read_sql(
        "SELECT MAX(date) AS d, count(*) FILTER (WHERE mkt_rf IS NULL) AS nulls FROM ff_factors"
    )
    last, nulls = row["d"].iloc[0], int(row["nulls"].iloc[0])
    age = (pd.Timestamp.today().date() - last).days if last else 9999
    return AssetCheckResult(
        passed=age <= 75 and nulls == 0, severity=AssetCheckSeverity.WARN,
        metadata={"latest": str(last), "age_days": age, "null_mkt_rf": nulls},
    )


@asset_check(asset=assets.macro, description="macro fresh + not all-null")
def macro_fresh() -> AssetCheckResult:
    row = db.read_sql(
        """SELECT MAX(date) AS d,
                  count(*) FILTER (WHERE yield_curve IS NULL) AS yc_null FROM macro"""
    )
    last, yc_null = row["d"].iloc[0], int(row["yc_null"].iloc[0])
    age = (pd.Timestamp.today().date() - last).days if last else 9999
    total = _count("SELECT count(*) c FROM macro")
    return AssetCheckResult(
        passed=age <= 75 and yc_null < total, severity=AssetCheckSeverity.WARN,
        metadata={"latest": str(last), "age_days": age},
    )


ALL_CHECKS = [
    universe_active_band,
    universe_sector_coverage,
    prices_positive,
    prices_freshness,
    prices_row_count,
    fundamentals_no_lookahead,
    fundamentals_concepts,
    fundamentals_unique_keys,
    fundamental_features_no_lookahead,
    fundamental_features_row_count,
    fundamental_features_unique_keys,
    ff_factors_fresh,
    macro_fresh,
]
