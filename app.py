"""
Spot-Vol Correlation Tool — Web Interface

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from spotvol.models import TenorInput
from spotvol.implied_beta import compute_implied_beta

TENORS = ["1D", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y"]

st.set_page_config(page_title="Spot-Vol Beta", layout="wide")
st.title("Implied Spot-Vol Beta")

# ─────────────────────────────────────────────────────────────────────
# Top row: pair and spot
# ─────────────────────────────────────────────────────────────────────

top1, top2, _ = st.columns([1, 1, 4])
with top1:
    pair = st.text_input("Pair", value="EURUSD")
with top2:
    spot = st.number_input("Spot", value=1.1415, format="%.4f", step=0.0001)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────
# Input: paste-friendly text area
# ─────────────────────────────────────────────────────────────────────

col_input, col_results = st.columns([1, 2])

with col_input:
    st.subheader("Market Data")
    st.caption("Paste tenor, vol, RR as columns (tab or space separated). Or edit directly.")

    default_text = """1D\t13.94\t-2.67
1W\t9.03\t-1.63
2W\t8.24\t-1.47
3W\t7.88\t-1.35
1M\t7.70\t-1.26
2M\t7.46\t1.02
3M\t7.30\t-0.81
6M\t7.17\t-0.34
9M\t7.16\t-0.08
1Y\t7.12\t0.15"""

    raw_input = st.text_area("Tenor | Vol (%) | RR (%)", value=default_text, height=330)

# ─────────────────────────────────────────────────────────────────────
# Parse input
# ─────────────────────────────────────────────────────────────────────

tenor_inputs = []
parsed_rows = []

for line in raw_input.strip().split("\n"):
    line = line.strip()
    if not line:
        continue
    parts = line.replace(",", " ").replace("\t", " ").split()
    if len(parts) >= 3:
        try:
            label = parts[0].upper()
            vol = float(parts[1])
            rr = float(parts[2])
            tenor_inputs.append(TenorInput(label=label, atm_vol=vol / 100, rr_25d=rr / 100))
            parsed_rows.append({"Tenor": label, "Vol %": vol, "RR %": rr})
        except (ValueError, KeyError):
            continue

# ─────────────────────────────────────────────────────────────────────
# Compute and display
# ─────────────────────────────────────────────────────────────────────

with col_results:
    if tenor_inputs:
        results = compute_implied_beta(spot, tenor_inputs)

        # Maintain input order
        label_order = {t.label: i for i, t in enumerate(tenor_inputs)}
        results.sort(key=lambda r: label_order.get(r.label, 999))

        st.subheader("Implied Beta")

        # Results table
        rows = []
        for r in results:
            rows.append({
                "Tenor": r.label,
                "Vol %": f"{r.atm_vol*100:.2f}",
                "RR %": f"{r.rr_25d*100:+.2f}",
                "Beta": f"{r.beta:+.2f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Chart
        st.subheader("Beta Term Structure")
        chart_df = pd.DataFrame({
            "Tenor": [r.label for r in results],
            "Beta": [r.beta for r in results],
        })
        tenor_cats = [r.label for r in results]  # preserve input order
        chart_df["Tenor"] = pd.Categorical(chart_df["Tenor"], categories=tenor_cats, ordered=True)
        chart_df = chart_df.sort_values("Tenor").set_index("Tenor")
        st.bar_chart(chart_df)

    else:
        st.info("Paste market data on the left.")

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Beta = vol pts per 1% spot move | Analytical Greeks | SF=0.5 | Calendar days | Zero rates"
)
