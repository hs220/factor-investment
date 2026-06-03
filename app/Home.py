"""Factor Investing dashboard — landing page.

Run locally:  streamlit run app/Home.py
(reads the warehouse; needs POSTGRES_PASSWORD, and FACTOR_DB_HOST if off-LAN.)
"""
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
for _p in (str(_root), str(_root / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from lib import data

st.set_page_config(page_title="Factor Investing", page_icon="📈", layout="wide")
st.title("📈 Factor Investing — Strategy B")
st.caption("Cross-sectional stock-selection ML over the Russell-3000 investable universe.")

if not data.ping():
    st.error("Database unreachable — set `POSTGRES_PASSWORD` (and `FACTOR_DB_HOST` if off-LAN).")
    st.stop()

st.markdown(
    """
This dashboard is a **read-only view over the warehouse** — the same data and model
the pipeline produces. Use the pages in the sidebar:

- **Recommendations** — the latest month's top names from the deployed model.
- **Signal / IC** — how each factor predicts forward returns.
- *(Performance / backtest — coming next.)*
"""
)

# Deployed-model snapshot
st.subheader("Deployed model")
try:
    ranked, manifest, asof = data.recommendations()
    c = st.columns(4)
    c[0].metric("Model", f"{manifest.model_name} · {manifest.horizon}")
    c[1].metric("OOS IC / IR",
                f"{manifest.oos_metrics['ic_mean']:.3f} / {manifest.oos_metrics['ic_ir']:.2f}")
    c[2].metric("Latest cross-section", str(asof.date()))
    c[3].metric("Universe", f"{len(ranked):,} names")
    st.caption(f"Version `{manifest.model_version}` · trained "
               f"{manifest.train_start} → {manifest.train_end}")
except Exception as exc:  # noqa: BLE001 — surface any artifact/DB issue to the user
    st.warning(f"No deployed model available yet ({exc}). "
               "Run the `model_train` job, then refresh.")

st.divider()
st.markdown(f"**Pipeline health & lineage:** [open the Dagster UI]({data.DAGSTER_URL}) "
            "for asset status, schedules, and data-quality checks.")
