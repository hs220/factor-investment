"""Query cached parquet in place with DuckDB.

Retrieval pattern for the NAS setup: the container *writes* parquet to the
shared folder; the workstation *reads* it over an NFS mount. DuckDB queries the
parquet files directly with SQL (column + row-group pushdown — it only reads
what the query selects), so there is no database server to run.

    from src.data import query
    query.sql("SELECT date, ticker, pred FROM {predictions} WHERE date >= '2024-01-01'")
    query.load_panel(start='2020-01-01')

The ``{name}`` placeholders expand to the parquet path under DATA_PROCESSED, so
queries are portable across local disk and the NAS mount.
"""
from __future__ import annotations

import pandas as pd

from src.config import DATA_PROCESSED


def _path(name: str) -> str:
    if not name.endswith(".parquet"):
        name += ".parquet"
    return str(DATA_PROCESSED / name)


def sql(query: str, **names: str) -> pd.DataFrame:
    """Run a DuckDB SQL query. ``{artifact}`` placeholders -> quoted parquet paths.

    Example: ``sql("SELECT * FROM {panel} WHERE date >= '2024-01-01'")``.
    """
    import duckdb

    # Default placeholders for every processed artifact referenced by name.
    fmt = {k: f"'{_path(v)}'" for k, v in names.items()}
    # Auto-resolve any {token} that maps to an existing parquet file.
    import string

    for _, token, _, _ in string.Formatter().parse(query):
        if token and token not in fmt:
            fmt[token] = f"'{_path(token)}'"
    return duckdb.sql(query.format(**fmt)).df()


def load_panel(
    *, start: str | None = None, end: str | None = None, columns: str = "*"
) -> pd.DataFrame:
    """Convenience loader for the feature panel with optional date filtering."""
    where = []
    if start:
        where.append(f"date >= '{start}'")
    if end:
        where.append(f"date <= '{end}'")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return sql(f"SELECT {columns} FROM {{panel}}{clause}")
