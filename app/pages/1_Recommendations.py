"""Recommendations — the latest month's top names from the deployed model."""
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
for _p in (str(_root), str(_root / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plotly.express as px
import streamlit as st

from lib import data

st.set_page_config(page_title="Recommendations", page_icon="📈", layout="wide")
st.title("Recommendations")

if not data.ping():
    st.error("Database unreachable.")
    st.stop()

try:
    ranked, manifest, asof = data.recommendations()
except Exception as exc:  # noqa: BLE001
    st.warning(f"No deployed model available yet ({exc}). Run the `model_train` job.")
    st.stop()

# --- Model card ----------------------------------------------------------------
st.subheader("Model card")
c = st.columns(4)
c[0].metric("Model", f"{manifest.model_name} · {manifest.horizon}")
c[1].metric("OOS mean IC", f"{manifest.oos_metrics['ic_mean']:.3f}")
c[2].metric("OOS IC IR", f"{manifest.oos_metrics['ic_ir']:.2f}")
c[3].metric("OOS hit-rate", f"{manifest.oos_metrics['hit_rate']:.0%}")
st.caption(f"Version `{manifest.model_version}` · trained {manifest.train_start} → "
           f"{manifest.train_end} · {manifest.n_train_rows:,} rows · sha `{manifest.code_sha}`")

st.divider()

# --- Top-N ---------------------------------------------------------------------
default_n = data.n_holdings()
top_n = st.slider("Top N to show", 10, 100, value=default_n, step=5)
st.subheader(f"Top {top_n} as-of {asof.date()}  ·  {len(ranked):,} names scored")

show_feats = ["momentum_12_2", "book_to_price", "roe", "earnings_yield"]
cols = ["ticker", "gics_sector", "pred"] + [f for f in show_feats if f in ranked.columns]
top = ranked.head(top_n)

left, right = st.columns([3, 2])
with left:
    st.dataframe(
        top[cols].rename(columns={"pred": "score", "gics_sector": "sector"}),
        use_container_width=True, hide_index=True,
        column_config={"score": st.column_config.NumberColumn(format="%.4f")},
    )
with right:
    by_sector = top["gics_sector"].value_counts().rename_axis("sector").reset_index(name="n")
    fig = px.bar(by_sector, x="n", y="sector", orientation="h",
                 title=f"Top-{top_n} sector breakdown")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
    st.plotly_chart(fig, use_container_width=True)

st.caption("Scores are within-sector cross-sectional rank predictions; features shown "
           "are normalized ranks in [0,1]. Long-only selection caps/weights are applied "
           "in the portfolio stage.")
