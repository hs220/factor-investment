"""Performance — walk-forward OOS backtest of the long-only top-N strategy."""
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
for _p in (str(_root), str(_root / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data
from src.backtest import metrics

# Categorical slots 1-2 of the validated reference palette (identity, fixed order).
STRATEGY, BENCHMARK = "#2a78d6", "#eb6834"

st.set_page_config(page_title="Performance", page_icon="📈", layout="wide")
st.title("Performance")

if not data.ping():
    st.error("Database unreachable.")
    st.stop()

n = st.slider("Holdings (top N)", 10, 100, value=data.n_holdings(), step=5)
out = data.backtest(n_holdings=n)
if out is None:
    st.warning("No model in `model_registry` yet. Run the `model_train` job.")
    st.stop()
version, res = out
bt = res.returns
s, b = res.strategy, res.benchmark

st.caption(f"Model `{version}` · out-of-sample walk-forward scores · "
           f"{bt.index.min():%Y-%m} → {bt.index.max():%Y-%m} ({len(bt)} months) · "
           "net of commission + half-spread costs · benchmark = equal-weight universe")

# --- Headline numbers ----------------------------------------------------------
c = st.columns(5)
c[0].metric("Ann. return (net)", f"{s['ann_return']:.1%}", f"{s['ann_return'] - b['ann_return']:+.1%} vs bench")
c[1].metric("Sharpe", f"{s['sharpe']:.2f}", f"{s['sharpe'] - b['sharpe']:+.2f} vs bench")
c[2].metric("Max drawdown", f"{s['max_drawdown']:.1%}", f"bench {b['max_drawdown']:.1%}", delta_color="off")
c[3].metric("Avg turnover / mo", f"{bt['turnover'].mean():.0%}")
attr = res.attribution
c[4].metric("FF5+MOM alpha (ann.)",
            "n/a" if "error" in attr else f"{attr['alpha_annual']:.1%}",
            None if "error" in attr else f"t = {attr['alpha_tstat']:.2f}", delta_color="off")


def _line(fig, y, name, color):
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values, name=name, mode="lines", line={"color": color, "width": 2},
        hovertemplate="%{x|%Y-%m}  %{y:.2f}<extra>" + name + "</extra>"))


# --- Growth of $1 (one axis, log scale) --------------------------------------
growth = pd.DataFrame({"Strategy (net)": metrics.cumulative(bt["net"]),
                       "Benchmark": metrics.cumulative(bt["benchmark"])})
fig = go.Figure()
_line(fig, growth["Strategy (net)"], "Strategy (net)", STRATEGY)
_line(fig, growth["Benchmark"], "Benchmark", BENCHMARK)
for col, color in (("Strategy (net)", STRATEGY), ("Benchmark", BENCHMARK)):
    fig.add_annotation(x=growth.index[-1], y=growth[col].iloc[-1], text=f"{col}  ${growth[col].iloc[-1]:,.1f}",
                       showarrow=False, xanchor="left", xshift=6, font={"size": 12})
fig.update_layout(title="Growth of $1 (log scale)", yaxis_type="log", hovermode="x unified",
                  height=420, margin={"r": 170}, legend={"orientation": "h", "y": 1.08})
st.plotly_chart(fig, width="stretch")

# --- Drawdown (its own chart, same units) -------------------------------------
dd = go.Figure()
_line(dd, metrics.drawdown(bt["net"]), "Strategy (net)", STRATEGY)
_line(dd, metrics.drawdown(bt["benchmark"]), "Benchmark", BENCHMARK)
dd.update_layout(title="Drawdown", yaxis_tickformat=".0%", hovermode="x unified",
                 height=300, legend={"orientation": "h", "y": 1.12})
st.plotly_chart(dd, width="stretch")

# --- Attribution + summary table ----------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("FF5 + momentum attribution")
    if "error" in attr:
        st.info(attr["error"])
    else:
        st.dataframe(pd.DataFrame({"beta": attr["betas"], "t-stat": attr["beta_tstats"]}).round(2),
                     width="stretch")
        st.caption(f"R² {attr['r_squared']:.2f} over {attr['n_months']} months. Alpha is the "
                   "return the factors don't explain.")
with right:
    st.subheader("Summary")
    keys = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "hit_rate"]
    st.dataframe(pd.DataFrame({"strategy (net)": [s.get(k) for k in keys],
                               "benchmark": [b.get(k) for k in keys]}, index=keys).round(3),
                 width="stretch")

with st.expander("Monthly returns (table view)"):
    st.dataframe(bt[["gross", "cost", "net", "benchmark", "turnover"]].sort_index(ascending=False)
                 .style.format("{:.2%}"), width="stretch")

st.caption("Caveats: survivorship bias (universe = today's listings, delisted names absent) "
           "flatters both lines; small/illiquid names may not be tradable at the modeled cost; "
           "monthly returns outside the plausibility band are masked as price errors.")
