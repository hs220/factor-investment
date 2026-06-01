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


def test_snap_q_nearest_quarter_end():
    # Off-calendar fiscal quarter-ends snap to the NEAREST calendar quarter-end,
    # never pushed forward into a later quarter (the old containing-quarter bug).
    assert F._snap_q(pd.Timestamp("2024-04-28")) == pd.Timestamp("2024-03-31")  # HD Q1
    assert F._snap_q(pd.Timestamp("2024-07-28")) == pd.Timestamp("2024-06-30")  # HD Q2
    assert F._snap_q(pd.Timestamp("2024-10-28")) == pd.Timestamp("2024-09-30")  # HD Q3
    assert F._snap_q(pd.Timestamp("2024-01-28")) == pd.Timestamp("2023-12-31")  # HD Q4
    # On-calendar dates are unchanged.
    assert F._snap_q(pd.Timestamp("2024-03-31")) == pd.Timestamp("2024-03-31")


def test_offcalendar_filer_no_lookahead():
    # HD-style filer (quarters end ~end of Apr/Jul/Oct/Jan), each filed ~3 weeks
    # later. Snapped period_end must never exceed the filing date.
    recs = [
        {"end": "2024-04-28", "val": 500e6, "filed": "2024-05-20", "form": "10-Q"},
        {"end": "2024-07-28", "val": 510e6, "filed": "2024-08-19", "form": "10-Q"},
        {"end": "2024-10-27", "val": 520e6, "filed": "2024-11-18", "form": "10-Q"},
    ]
    pts = F._instant_changepoints(recs)
    assert pts, "expected change-points"
    assert all(period_end <= filed for period_end, filed, _v, _f in pts)


def test_future_dated_fact_dropped():
    # Debt-maturity-schedule fact: period ends years after it was filed -> drop.
    recs = [
        {"end": "2030-09-30", "val": 1e9, "filed": "2024-10-30", "form": "10-Q"},
        {"end": "2024-09-30", "val": 2e9, "filed": "2024-10-30", "form": "10-Q"},
    ]
    pts = F._instant_changepoints(recs)
    assert len(pts) == 1
    assert pts[0][0] == pd.Timestamp("2024-09-30")
    # Same guard on the flow path.
    frecs = [
        {"start": "2030-07-01", "end": "2030-09-30", "val": 1e9, "filed": "2024-10-30", "form": "10-Q"},
    ]
    assert F._flow_changepoints(frecs) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all passed")
