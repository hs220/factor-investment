"""Signal / IC — how each factor predicts forward returns (notebook-03 content)."""
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
for _p in (str(_root), str(_root / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plotly.express as px
import streamlit as st

from lib import data

st.set_page_config(page_title="Signal / IC", page_icon="📊", layout="wide")
st.title("Signal / IC analysis")
st.caption("Per-signal Information Coefficient (Spearman rank corr of the signal vs. "
           "forward return, per month) — the baseline the ML model must beat.")

if not data.ping():
    st.error("Database unreachable.")
    st.stop()

report = data.signal_report().sort_values("ic_mean", ascending=False)

left, right = st.columns([2, 3])
with left:
    st.dataframe(
        report[["ic_mean", "ic_ir", "t_stat", "hit_rate", "n_months"]].round(3),
        use_container_width=True,
    )
with right:
    fig = px.bar(report.reset_index().rename(columns={"index": "signal"}),
                 x="ic_mean", y="signal", orientation="h", title="Mean IC by signal")
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=460)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Feature coverage (% non-null, 2015+)")
cov = data.feature_coverage()
fig2 = px.bar(cov.reset_index().rename(columns={"index": "feature", 0: "coverage_pct"}),
              x="coverage_pct", y="feature", orientation="h")
fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=460,
                   xaxis_title="% non-null")
st.plotly_chart(fig2, use_container_width=True)
st.caption("Fundamentals are bounded by EDGAR filing depth (~5y); price/macro features "
           "cover more of the panel. See the panel notebook for detail.")
