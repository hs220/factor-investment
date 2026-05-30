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
@asset_check(asset=assets.prices_table, description="no non-positive close prices")
def prices_positive() -> AssetCheckResult:
    bad = _count("SELECT count(*) c FROM prices WHERE close IS NOT NULL AND close <= 0")
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
@asset_check(asset=assets.fundamental_facts,
             description="point-in-time: filed_date >= period_end (no lookahead)")
def fundamentals_no_lookahead() -> AssetCheckResult:
    bad = _count(
        "SELECT count(*) c FROM fundamental_facts WHERE filed_date < period_end"
    )
    return AssetCheckResult(
        passed=bad == 0, severity=AssetCheckSeverity.ERROR,
        metadata={"lookahead_rows": bad},
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
    ff_factors_fresh,
    macro_fresh,
]
