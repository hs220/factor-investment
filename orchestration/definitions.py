"""Dagster definitions — assets, jobs, and schedules for the data warehouse.

Load with:  dagster dev -m orchestration.definitions
"""
from __future__ import annotations

from dagster import Definitions, ScheduleDefinition, define_asset_job

from orchestration import assets

_ALL = [
    assets.universe_table,
    assets.prices_table,
    assets.fundamental_facts,
    assets.sectors,
    assets.ff_factors,
    assets.macro,
]

# Daily: refresh prices (and the cheap factor/macro series).
daily_job = define_asset_job(
    "daily_ingest", selection=["prices_table", "ff_factors", "macro"]
)
# Weekly: check EDGAR for new filings (incremental fundamentals + sectors).
fundamentals_job = define_asset_job(
    "fundamentals_ingest", selection=["fundamental_facts", "sectors"]
)

defs = Definitions(
    assets=_ALL,
    jobs=[daily_job, fundamentals_job],
    schedules=[
        ScheduleDefinition(job=daily_job, cron_schedule="0 6 * * 1-5"),       # weekdays 6am
        ScheduleDefinition(job=fundamentals_job, cron_schedule="0 7 * * 1"),  # Mondays 7am
    ],
)
