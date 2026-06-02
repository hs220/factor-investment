"""Dagster definitions — assets, jobs, and schedules for the data warehouse.

Load with:  dagster dev -m orchestration.definitions

Cadence:
- daily_ingest   (weekdays 6am ET): prices + fundamentals — the fast-moving
  data. Fundamentals run daily because the incremental check is cheap (poll
  submissions, fetch only new filings), so a Monday filing lands Tuesday.
- monthly_ingest (5th of month 6am ET): universe rebuild + FF factors + macro
  + sectors — slow-moving / monthly-published series.

Schedules are RUNNING by default; the daemon picks them up on deploy.
"""
from __future__ import annotations

from dagster import (
    DefaultScheduleStatus,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
)

from orchestration import assets
from orchestration.checks import ALL_CHECKS

_ALL = [
    assets.universe_table,
    assets.prices_table,
    assets.fundamental_facts,
    assets.sectors,
    assets.fundamental_features,
    assets.ff_factors,
    assets.macro,
]

daily_job = define_asset_job(
    "daily_ingest",
    selection=["prices_table", "fundamental_facts", "fundamental_features"],
)
monthly_job = define_asset_job(
    "monthly_ingest", selection=["universe_table", "ff_factors", "macro", "sectors"]
)

_TZ = "America/New_York"
defs = Definitions(
    assets=_ALL,
    asset_checks=ALL_CHECKS,
    jobs=[daily_job, monthly_job],
    schedules=[
        ScheduleDefinition(
            job=daily_job,
            cron_schedule="0 6 * * 1-5",          # weekdays 6am ET
            execution_timezone=_TZ,
            default_status=DefaultScheduleStatus.RUNNING,
        ),
        ScheduleDefinition(
            job=monthly_job,
            cron_schedule="0 6 5 * *",            # 5th of month 6am ET
            execution_timezone=_TZ,
            default_status=DefaultScheduleStatus.RUNNING,
        ),
    ],
)
