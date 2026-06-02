"""Pandera schemas — library-boundary data-quality contracts for ``src/data``.

These validate the **shape, dtypes, and value ranges** of DataFrames as they
leave the library functions, complementing the warehouse-level Dagster asset
checks in ``orchestration/checks.py`` (which validate the same data *after* it
lands in Postgres). Catching a violation here stops bad data at the source,
before it is ever written — the two layers are deliberately redundant.

Currently covers the EDGAR ``fundamental_facts`` long table, the point-in-time
sensitive output where a lookahead bug is most costly.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.data.fundamentals import _ALL_CONCEPTS

# Same tolerance as the warehouse PIT check (orchestration/checks.py
# ``fundamentals_no_lookahead``): a benign calendar-quarter snap-forward can land
# ``filed_date`` a few weeks *past* ``period_end`` (a relabeling, <=~46d), while a
# real lookahead bug sits 100-2000+ days early. 60d keeps the canary, ignores noise.
LOOKAHEAD_TOL_DAYS = 60

_KNOWN_CONCEPTS = sorted(_ALL_CONCEPTS)


def _no_lookahead(df: pd.DataFrame) -> pd.Series:
    """Per-row PIT guard: ``filed_date`` no earlier than ``period_end`` - 60d."""
    return (df["period_end"] - df["filed_date"]).dt.days <= LOOKAHEAD_TOL_DAYS


FUNDAMENTAL_FACTS = DataFrameSchema(
    {
        "ticker": Column(str, Check.str_matches(r"^[A-Z0-9.\-]+$"), nullable=False),
        "cik": Column(str, Check.str_matches(r"^\d{10}$"), nullable=False),
        "concept": Column(str, Check.isin(_KNOWN_CONCEPTS), nullable=False),
        "gaap_tag": Column(str, nullable=False),
        "period_end": Column("datetime64[ns]", nullable=False),
        # _snap_q always lands on a quarter-end month, so a label is expected;
        # nullable kept open so a stray off-calendar period can't hard-fail a fetch.
        "fiscal_period": Column(str, Check.isin(["Q1", "Q2", "Q3", "Q4"]), nullable=True),
        "duration_days": Column(int, Check.isin([0, 91]), nullable=False),
        "filed_date": Column("datetime64[ns]", nullable=False),
        "form": Column(str, nullable=False),
        "value": Column(float, nullable=True),
        "unit": Column(str, nullable=False),
    },
    checks=Check(
        _no_lookahead,
        name="pit_no_lookahead",
        error=f"filed_date earlier than period_end minus {LOOKAHEAD_TOL_DAYS}d (lookahead)",
    ),
    strict=True,   # reject unexpected columns — schema is the column contract
    coerce=True,   # normalise dtypes (e.g. int-valued `value`, object datetimes)
    name="fundamental_facts",
)


def validate_fundamental_facts(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Validate the long EDGAR facts table; return the (dtype-coerced) frame.

    No-op on an empty frame (the column-only placeholder a fully-skipped fetch
    returns has no rows and object-typed datetime columns to coerce). With
    ``lazy=True`` every violation is collected and surfaced in one
    ``pandera.errors.SchemaErrors`` rather than aborting on the first.
    """
    if df.empty:
        return df
    return FUNDAMENTAL_FACTS.validate(df, lazy=lazy)
