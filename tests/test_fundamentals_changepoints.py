"""Regression tests for the restatement-aware EDGAR quarterization.

Run: python -m pytest tests/test_fundamentals_changepoints.py
(or plain `python tests/test_fundamentals_changepoints.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import fundamentals as F  # noqa: E402


def _asof(points, period_end, date):
    df = pd.DataFrame(points, columns=["period_end", "filed", "value", "form"])
    sub = df[(df.period_end == pd.Timestamp(period_end)) & (df.filed <= pd.Timestamp(date))]
    return sub.sort_values("filed").value.iloc[-1] if len(sub) else None


def test_flow_restatement_as_of():
    recs = [
        {"start": "2023-01-01", "end": "2023-03-31", "val": 100e6, "filed": "2023-05-15", "form": "10-Q"},
        {"start": "2023-01-01", "end": "2023-03-31", "val": 80e6, "filed": "2023-11-20", "form": "10-Q/A"},
    ]
    pts = F._flow_changepoints(recs)
    assert _asof(pts, "2023-03-31", "2023-07-01") == 100e6   # original
    assert _asof(pts, "2023-03-31", "2023-12-01") == 80e6    # restated


def test_flow_quarterization_incl_q4():
    recs = [
        {"start": "2023-01-01", "end": "2023-03-31", "val": 100e6, "filed": "2023-05-15", "form": "10-Q"},
        {"start": "2023-04-01", "end": "2023-06-30", "val": 120e6, "filed": "2023-08-01", "form": "10-Q"},
        {"start": "2023-01-01", "end": "2023-06-30", "val": 220e6, "filed": "2023-08-01", "form": "10-Q"},
        {"start": "2023-01-01", "end": "2023-09-30", "val": 330e6, "filed": "2023-11-01", "form": "10-Q"},
        {"start": "2023-01-01", "end": "2023-12-31", "val": 500e6, "filed": "2024-02-15", "form": "10-K"},
    ]
    q = {p[0]: p[2] for p in F._flow_changepoints(recs)}
    assert q[pd.Timestamp("2023-03-31")] == 100e6
    assert q[pd.Timestamp("2023-06-30")] == 120e6          # direct 3-month wins
    assert q[pd.Timestamp("2023-09-30")] == 330e6 - 220e6  # 9mo - 6mo
    assert q[pd.Timestamp("2023-12-31")] == 500e6 - 330e6  # annual - 9mo (Q4)


def test_instant_restatement():
    recs = [
        {"end": "2023-03-31", "val": 500e6, "filed": "2023-05-15", "form": "10-Q"},
        {"end": "2023-03-31", "val": 450e6, "filed": "2023-11-20", "form": "10-Q/A"},
    ]
    pts = F._instant_changepoints(recs)
    assert _asof(pts, "2023-03-31", "2023-07-01") == 500e6
    assert _asof(pts, "2023-03-31", "2023-12-01") == 450e6


def test_no_duplicate_changepoints_when_value_stable():
    # Same value re-reported as a comparative in a later filing -> no new point.
    recs = [
        {"end": "2023-03-31", "val": 500e6, "filed": "2023-05-15", "form": "10-Q"},
        {"end": "2023-03-31", "val": 500e6, "filed": "2024-05-15", "form": "10-Q"},
    ]
    assert len(F._instant_changepoints(recs)) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all passed")
