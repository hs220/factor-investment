"""Recommendations — the latest month's top names from the deployed model."""
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
for _p in (str(_root), str(_root / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import plotly.express as px
import streamlit as st

from lib import data
from src.portfolio.holdings import parse_positions_csv, save_snapshot
from src.portfolio.rebalance import plan_rebalance

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
        width="stretch", hide_index=True,
        column_config={"score": st.column_config.NumberColumn(format="%.4f")},
    )
with right:
    by_sector = top["gics_sector"].value_counts().rename_axis("sector").reset_index(name="n")
    fig = px.bar(by_sector, x="n", y="sector", orientation="h",
                 title=f"Top-{top_n} sector breakdown")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
    st.plotly_chart(fig, width="stretch")

st.caption("Scores are within-sector cross-sectional rank predictions; features shown "
           "are normalized ranks in [0,1]. Long-only selection caps/weights are applied "
           "in the portfolio stage.")

st.divider()

# --- Rebalance actions against the current portfolio --------------------------
st.subheader("Rebalance actions")
st.caption("Compares your current Roth IRA holdings with the model's ranking using the same "
           "hold-buffer rule as the backtest: sell names that fell out of the buffer, buy the "
           "top-ranked names you don't own, resize only on large drift.")

with st.expander("Upload current positions (broker CSV export)",
                 expanded=data.holdings()[2] is None):
    up = st.file_uploader("Positions CSV (Schwab, Fidelity, … — needs Symbol + Quantity columns)",
                          type="csv")
    if up is not None:
        try:
            pos_new, cash_new = parse_positions_csv(up.getvalue())
        except ValueError as exc:
            st.error(f"Couldn't read that file: {exc}")
        else:
            st.write(f"Parsed **{len(pos_new)} positions** and **${cash_new:,.2f} cash**:")
            st.dataframe(pos_new, width="stretch", hide_index=True)
            if st.button("Save as current holdings", type="primary"):
                save_snapshot(pos_new, cash_new, pd.Timestamp.today())
                st.success("Saved.")
                st.rerun()

positions, cash, held_asof = data.holdings()
if held_asof is None:
    st.info("No holdings saved yet — upload a positions export above to get buy/sell actions.")
    st.stop()

plan, summ = plan_rebalance(positions, cash, ranked, data.latest_closes())
k = st.columns(5)
k[0].metric("Strategy sleeve", f"${summ['sleeve_value']:,.0f}")
k[1].metric("Sell", summ.get("n_sell", 0))
k[2].metric("Buy", summ.get("n_buy", 0))
k[3].metric("Est. turnover", f"{summ.get('turnover', 0):.0%}")
k[4].metric("Est. cost", f"${summ.get('est_cost', 0):,.0f}")
st.caption(f"Holdings as of {held_asof.date()} · model `{manifest.model_version}` · scores as of "
           f"{asof.date()} · target ≈ ${summ.get('target_per_name', 0):,.0f} per name · "
           f"cash after trades ≈ ${summ.get('cash_after', 0):,.0f}")

show = plan[["action", "ticker", "sector", "rank", "price", "current_shares", "trade_shares",
             "target_shares", "trade_value", "reason"]]
st.dataframe(show, width="stretch", hide_index=True, column_config={
    "price": st.column_config.NumberColumn(format="$%.2f"),
    "trade_value": st.column_config.NumberColumn("trade $", format="$%.0f"),
})
st.download_button("Download trade list (CSV)", show.to_csv(index=False).encode(),
                   file_name=f"rebalance_{asof.date()}.csv", mime="text/csv")
st.caption("Advisory only — review before placing orders. Whole shares; REVIEW rows "
           "(ETFs / names the model doesn't score) are never traded automatically.")
