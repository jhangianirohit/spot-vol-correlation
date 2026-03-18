"""
Spot-Vol Correlation Tool — Web Interface

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from spotvol.models import TenorInput
from spotvol.implied_beta import compute_implied_beta
from spotvol.cross_pairs import parse_vol_file, decompose_cross, CROSS_LEGS

st.set_page_config(page_title="Spot-Vol Beta", layout="wide")

tab1, tab2 = st.tabs(["Implied Beta", "Cross Decomposition"])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: Single pair implied beta (existing)
# ─────────────────────────────────────────────────────────────────────

TENORS = ["1D", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y"]

with tab1:
    st.header("Implied Spot-Vol Beta")

    top1, top2, _ = st.columns([1, 1, 4])
    with top1:
        pair = st.text_input("Pair", value="EURUSD", key="t1_pair")
    with top2:
        spot = st.number_input("Spot", value=1.1415, format="%.4f", step=0.0001, key="t1_spot")

    st.markdown("---")
    col_input, col_results = st.columns([1, 2])

    default_text = """1D\t11.88\t-2.67
1W\t9.03\t-1.63
2W\t8.24\t-1.47
3W\t7.88\t-1.35
1M\t7.70\t-1.26
2M\t7.46\t1.02
3M\t7.30\t-0.81
6M\t7.17\t-0.34
9M\t7.16\t-0.08
1Y\t7.12\t0.15"""

    with col_input:
        st.subheader("Market Data")
        st.caption("Tenor | Vol (%) | RR (%)")
        raw_input = st.text_area("Paste data", value=default_text, height=330, key="t1_input")

    tenor_inputs = []
    for line in raw_input.strip().split("\n"):
        parts = line.strip().replace(",", " ").replace("\t", " ").split()
        if len(parts) >= 3:
            try:
                tenor_inputs.append(TenorInput(parts[0].upper(), float(parts[1])/100, float(parts[2])/100))
            except (ValueError, KeyError):
                continue

    with col_results:
        if tenor_inputs:
            results = compute_implied_beta(spot, tenor_inputs)
            label_order = {t.label: i for i, t in enumerate(tenor_inputs)}
            results.sort(key=lambda r: label_order.get(r.label, 999))

            st.subheader("Implied Beta")
            df = pd.DataFrame([{
                "Tenor": r.label,
                "Vol %": f"{r.atm_vol*100:.2f}",
                "RR %": f"{r.rr_25d*100:+.2f}",
                "Beta": f"{r.beta:+.2f}",
            } for r in results])
            st.dataframe(df, use_container_width=True, hide_index=True)

            chart_df = pd.DataFrame({
                "Tenor": [r.label for r in results],
                "Beta": [r.beta for r in results],
            })
            chart_df["Tenor"] = pd.Categorical(chart_df["Tenor"],
                categories=[r.label for r in results], ordered=True)
            chart_df = chart_df.sort_values("Tenor").set_index("Tenor")
            st.bar_chart(chart_df)

    st.caption("Beta = vol pts per 1% spot move | Analytical Greeks | SF=0.5 | Calendar days")

# ─────────────────────────────────────────────────────────────────────
# TAB 2: Cross decomposition
# ─────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Cross-Pair RR Decomposition")
    st.caption(
        "Paste all vols and RRs. Format: pair  tenor  vol(%)  rr(%).  "
        "USD pairs (EURUSD, USDJPY, etc.) and crosses (EURJPY, EURCNH, etc.). "
        "RR sign for USDXXX pairs is auto-flipped."
    )

    default_cross = """# USD pairs
EURUSD\t1M\t7.53\t-1.03
EURUSD\t3M\t7.25\t-0.66
USDCNH\t1M\t3.55\t-0.25
USDCNH\t3M\t3.58\t-0.30
USDJPY\t1M\t10.20\t0.85
USDJPY\t3M\t9.80\t0.60
GBPUSD\t1M\t7.10\t-0.90
GBPUSD\t3M\t6.90\t-0.55
AUDUSD\t1M\t10.50\t-1.40
AUDUSD\t3M\t10.20\t-1.10
# Crosses
EURCNH\t1M\t6.38\t-0.65
EURCNH\t3M\t6.27\t-0.33
EURJPY\t1M\t10.80\t-0.45
EURJPY\t3M\t10.50\t-0.30
EURGBP\t1M\t5.20\t-0.25
EURGBP\t3M\t5.00\t-0.15"""

    raw_cross = st.text_area("Pair | Tenor | Vol (%) | RR (%)", value=default_cross,
                              height=400, key="t2_input")

    if raw_cross.strip():
        vol_data = parse_vol_file(raw_cross)

        if vol_data:
            # Find all crosses in the data
            crosses_in_data = [p for p in vol_data if p in CROSS_LEGS]
            # Find all tenors
            all_tenors = set()
            for p in vol_data:
                all_tenors.update(vol_data[p].keys())
            all_tenors = sorted(all_tenors, key=lambda t: {"1D":0,"1W":1,"2W":2,"3W":3,
                "1M":4,"2M":5,"3M":6,"6M":7,"9M":8,"1Y":9}.get(t, 99))

            # First show USD pair betas
            st.subheader("USD Pair Betas")
            usd_pairs = [p for p in vol_data if p not in CROSS_LEGS]
            beta_rows = []
            for pair in sorted(usd_pairs):
                for tenor in all_tenors:
                    if tenor in vol_data[pair]:
                        vol, rr = vol_data[pair][tenor]
                        if rr is not None and vol > 0:
                            r = compute_implied_beta(1.0, [TenorInput(tenor, vol, rr)])[0]
                            beta_rows.append({
                                "Pair": pair,
                                "Tenor": tenor,
                                "Vol %": f"{vol*100:.2f}",
                                "RR %": f"{rr*100:+.2f}",
                                "Beta": f"{r.beta:+.2f}",
                            })

            if beta_rows:
                st.dataframe(pd.DataFrame(beta_rows), use_container_width=True, hide_index=True)

            # Cross decomposition
            st.subheader("Cross Predicted vs Market RR")
            cross_rows = []
            for cross in crosses_in_data:
                for tenor in all_tenors:
                    result = decompose_cross(cross, tenor, vol_data)
                    if result:
                        cross_rows.append({
                            "Cross": result.cross,
                            "Tenor": result.tenor,
                            "Legs": f"{result.leg1} / {result.leg2}",
                            "ρ": f"{result.rho:.2f}",
                            "β leg1": f"{result.beta_leg1:+.2f}",
                            "β leg2": f"{result.beta_leg2:+.2f}",
                            "Vol wts": f"{result.vol_w_leg1:+.2f} / {result.vol_w_leg2:+.2f}",
                            "β cross": f"{result.beta_cross:+.2f}",
                            "Pred RR%": f"{result.predicted_rr*100:+.2f}",
                            "Mkt RR%": f"{result.market_rr*100:+.2f}" if result.market_rr is not None else "—",
                            "Rich/Cheap": f"{result.residual*100:+.2f}%" if result.residual is not None else "—",
                        })

            if cross_rows:
                df_cross = pd.DataFrame(cross_rows)
                st.dataframe(df_cross, use_container_width=True, hide_index=True)

                # Residual chart
                chart_rows = [r for r in cross_rows if r["Rich/Cheap"] != "—"]
                if chart_rows:
                    st.subheader("Residual (Market - Predicted)")
                    chart_df = pd.DataFrame(chart_rows)
                    chart_df["Residual (bps)"] = chart_df["Rich/Cheap"].str.replace("%", "").astype(float) * 100
                    chart_df["Label"] = chart_df["Cross"] + " " + chart_df["Tenor"]
                    chart_df = chart_df.set_index("Label")[["Residual (bps)"]]
                    st.bar_chart(chart_df)

            st.caption(
                "Rich/Cheap = Market RR - Predicted RR. "
                "Positive = market RR is less negative than predicted (cross RR is cheap). "
                "Negative = cross RR is rich."
            )
