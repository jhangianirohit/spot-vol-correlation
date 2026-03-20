"""
Spot-Vol Correlation Tool — Web Interface

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from spotvol.models import TenorInput
from spotvol.implied_beta import compute_implied_beta
from spotvol.cross_pairs import parse_vol_file, decompose_cross, CROSS_LEGS
from spotvol.tenors import TENOR_ORDER

# ─────────────────────────────────────────────────────────────────────
# Theme CSS — matching FX Options PnL Calculator
# ─────────────────────────────────────────────────────────────────────

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #0d1117;
    --bg-elevated: #151b23;
    --bg-input: #1c2333;
    --text-primary: #e5e5e5;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --cyan: #06b6d4;
    --cyan-dim: rgba(6, 182, 212, 0.3);
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --border: #21262d;
    --border-strong: #30363d;
}

/* Main app background */
.stApp, section[data-testid="stSidebar"] {
    background-color: var(--bg-main) !important;
}

/* All text */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Headers */
.stApp h1 {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--cyan) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
}

.stApp h2, .stApp h3 {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-main) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
    background-color: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}

/* Text areas and inputs */
.stTextArea textarea, .stTextInput input, .stNumberInput input {
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 1px var(--cyan-dim) !important;
}

/* Dataframes */
.stDataFrame, [data-testid="stDataFrame"] {
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stDataFrame"] th {
    background-color: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stDataFrame"] td {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-size: 0.8rem !important;
    font-variant-numeric: tabular-nums !important;
}

/* Captions */
.stApp .stCaption, .stApp figcaption {
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.04em;
}

/* Buttons */
.stButton button {
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.7rem !important;
    background-color: transparent !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    padding: 6px 16px !important;
}

.stButton button:hover {
    background-color: var(--cyan) !important;
    color: #000 !important;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 12px 0;
}

.metric-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-left: 3px solid var(--cyan);
    padding: 10px 14px;
    border-radius: 4px;
    flex: 1;
}

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Custom table for results */
.result-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    margin: 8px 0;
}

.result-table th {
    background: var(--bg-elevated);
    color: var(--text-secondary);
    text-transform: uppercase;
    font-size: 0.65rem;
    letter-spacing: 0.06em;
    padding: 8px 10px;
    text-align: right;
    border-bottom: 1px solid var(--border-strong);
    font-weight: 500;
}

.result-table th:first-child { text-align: left; }

.result-table td {
    padding: 6px 10px;
    text-align: right;
    border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
}

.result-table td:first-child { text-align: left; color: var(--cyan); font-weight: 500; }

.result-table tr:hover td { background: rgba(6, 182, 212, 0.05); }

.val-pos { color: var(--green) !important; }
.val-neg { color: var(--red) !important; }
.val-muted { color: var(--text-muted) !important; }

/* Divider */
.divider { border-top: 1px solid var(--border); margin: 16px 0; }

/* Horizontal rule override */
hr { border-color: var(--border) !important; }
</style>
"""

st.set_page_config(page_title="SPOT-VOL BETA", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────

st.title("Spot-Vol Correlation")

# Tenor ordering helper
TENOR_RANK = {t: i for i, t in enumerate(TENOR_ORDER)}


def sort_tenor_list(tenors):
    return sorted(tenors, key=lambda t: TENOR_RANK.get(t.upper(), 999))


def color_val(val_str):
    """Wrap a value string with color class."""
    try:
        v = float(val_str.replace("%", "").replace("+", ""))
        if v > 0.005:
            return f'<span class="val-pos">{val_str}</span>'
        elif v < -0.005:
            return f'<span class="val-neg">{val_str}</span>'
    except ValueError:
        pass
    return val_str


def html_table(headers, rows, color_cols=None):
    """Generate an HTML result table."""
    color_cols = color_cols or []
    html = '<table class="result-table"><thead><tr>'
    for h in headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead><tbody>'
    for row in rows:
        html += '<tr>'
        for i, cell in enumerate(row):
            if i in color_cols:
                html += f'<td>{color_val(str(cell))}</td>'
            else:
                html += f'<td>{cell}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


# ─────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["SINGLE PAIR BETA", "CROSS DECOMPOSITION"])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: Single pair implied beta
# ─────────────────────────────────────────────────────────────────────

with tab1:
    top1, top2, _ = st.columns([1, 1, 4])
    with top1:
        pair = st.text_input("PAIR", value="EURUSD", key="t1_pair")
    with top2:
        spot = st.number_input("SPOT", value=1.1415, format="%.4f", step=0.0001, key="t1_spot")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_input, col_results = st.columns([1, 2])

    default_text = """ON\t11.88\t-2.67
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
        st.caption("TENOR  VOL(%)  RR(%)")
        raw_input = st.text_area("data", value=default_text, height=330, key="t1_input",
                                  label_visibility="collapsed")

    tenor_inputs = []
    for line in raw_input.strip().split("\n"):
        parts = line.strip().replace(",", " ").replace("\t", " ").split()
        if len(parts) >= 3:
            try:
                tenor_inputs.append(TenorInput(parts[0].upper(), float(parts[1]) / 100, float(parts[2]) / 100))
            except (ValueError, KeyError):
                continue

    with col_results:
        if tenor_inputs:
            results = compute_implied_beta(spot, tenor_inputs)
            label_order = {t.label: i for i, t in enumerate(tenor_inputs)}
            results.sort(key=lambda r: label_order.get(r.label, 999))

            st.subheader("Implied Beta")

            headers = ["Tenor", "Vol %", "RR %", "Beta"]
            rows = []
            for r in results:
                rows.append([
                    r.label,
                    f"{r.atm_vol * 100:.2f}",
                    f"{r.rr_25d * 100:+.2f}",
                    f"{r.beta:+.2f}",
                ])
            st.markdown(html_table(headers, rows, color_cols=[2, 3]), unsafe_allow_html=True)

            chart_df = pd.DataFrame({
                "Tenor": [r.label for r in results],
                "Beta": [r.beta for r in results],
            })
            chart_df["Tenor"] = pd.Categorical(chart_df["Tenor"],
                                                categories=[r.label for r in results], ordered=True)
            chart_df = chart_df.sort_values("Tenor").set_index("Tenor")
            st.bar_chart(chart_df)

    st.caption("Beta = vol pts per 1% spot move · Analytical Greeks · SF=0.5 · Calendar days")


# ─────────────────────────────────────────────────────────────────────
# TAB 2: Cross decomposition
# ─────────────────────────────────────────────────────────────────────

with tab2:
    st.caption("Positive RR = LHS calls over puts · No sign flipping · Paste as quoted")

    default_cross = """# USD PAIRS
EURUSD\tON\t11.88\t-2.67
EURUSD\t1W\t9.03\t-1.63
EURUSD\t1M\t7.53\t-1.03
EURUSD\t3M\t7.25\t-0.66
USDCNH\tON\t5.20\t-0.80
USDCNH\t1W\t3.80\t-0.35
USDCNH\t1M\t3.55\t-0.25
USDCNH\t3M\t3.58\t-0.30
USDJPY\tON\t14.50\t0.95
USDJPY\t1W\t10.80\t0.90
USDJPY\t1M\t10.20\t0.85
USDJPY\t3M\t9.80\t0.60
GBPUSD\t1M\t7.10\t-0.90
GBPUSD\t3M\t6.90\t-0.55
AUDUSD\t1M\t10.50\t-1.40
AUDUSD\t3M\t10.20\t-1.10
# CROSSES
EURCNH\t1M\t6.38\t-0.65
EURCNH\t3M\t6.27\t-0.33
EURJPY\t1M\t10.80\t-0.45
EURJPY\t3M\t10.50\t-0.30
EURGBP\t1M\t5.20\t-0.25
EURGBP\t3M\t5.00\t-0.15"""

    raw_cross = st.text_area("data", value=default_cross, height=400, key="t2_input",
                              label_visibility="collapsed")

    if raw_cross.strip():
        vol_data = parse_vol_file(raw_cross)

        if vol_data:
            crosses_in_data = [p for p in vol_data if p in CROSS_LEGS]
            usd_pairs = [p for p in vol_data if p not in CROSS_LEGS]

            all_tenors = set()
            for p in vol_data:
                all_tenors.update(vol_data[p].keys())
            all_tenors = sort_tenor_list(list(all_tenors))

            # ── USD pair betas ──
            st.subheader("USD Pair Implied Betas")

            beta_headers = ["Pair", "Tenor", "Vol %", "RR %", "Beta", "RR/ATM"]
            beta_rows = []
            for pair in sorted(usd_pairs):
                for tenor in all_tenors:
                    if tenor in vol_data[pair]:
                        vol, rr = vol_data[pair][tenor]
                        if rr is not None and vol > 0:
                            r = compute_implied_beta(1.0, [TenorInput(tenor, vol, rr)])[0]
                            beta_rows.append([
                                pair, tenor,
                                f"{vol * 100:.2f}", f"{rr * 100:+.2f}",
                                f"{r.beta:+.2f}", f"{rr / vol:+.3f}",
                            ])

            if beta_rows:
                st.markdown(html_table(beta_headers, beta_rows, color_cols=[3, 4]),
                            unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # ── Cross decomposition ──
            st.subheader("Cross Predicted vs Market RR")

            cross_headers = ["Cross", "Tenor", "Legs", "ρ", "β cross",
                             "Pred RR%", "Mkt RR%", "Residual"]
            cross_rows = []
            all_results = []

            for cross in crosses_in_data:
                for tenor in all_tenors:
                    result = decompose_cross(cross, tenor, vol_data)
                    if result:
                        all_results.append(result)
                        residual_str = f"{result.residual * 100:+.2f}" if result.residual is not None else "—"
                        cross_rows.append([
                            result.cross, result.tenor,
                            f"{result.leg1} / {result.leg2}",
                            f"{result.rho:.2f}",
                            f"{result.beta_cross:+.2f}",
                            f"{result.predicted_rr * 100:+.2f}",
                            f"{result.market_rr * 100:+.2f}" if result.market_rr is not None else "—",
                            residual_str,
                        ])

            if cross_rows:
                st.markdown(html_table(cross_headers, cross_rows, color_cols=[4, 5, 6, 7]),
                            unsafe_allow_html=True)

                # ── Residual chart (chronological order) ──
                chart_results = [r for r in all_results if r.residual is not None]
                if chart_results:
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.subheader("Residual: Market - Predicted (bps)")

                    # Sort: by cross name, then by tenor chronologically
                    chart_results.sort(key=lambda r: (
                        r.cross, TENOR_RANK.get(r.tenor, 999)
                    ))

                    labels = [f"{r.cross} {r.tenor}" for r in chart_results]
                    residuals = [r.residual * 10000 for r in chart_results]  # in bps

                    chart_df = pd.DataFrame({
                        "Label": labels,
                        "Residual (bps)": residuals,
                    })
                    chart_df["Label"] = pd.Categorical(chart_df["Label"],
                                                        categories=labels, ordered=True)
                    chart_df = chart_df.sort_values("Label").set_index("Label")
                    st.bar_chart(chart_df)

            # ── Email report ──
            if all_results:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.subheader("Email Report")

                now = datetime.now().strftime("%Y-%m-%d %H:%M")

                # Build email HTML (light theme for email clients)
                email_lines = []
                email_lines.append(f'<div style="font-family: Consolas, Monaco, monospace; font-size: 12px; color: #1f2937; max-width: 900px;">')
                email_lines.append(f'<h2 style="font-size: 14px; color: #0d1117; border-bottom: 2px solid #06b6d4; padding-bottom: 6px;">SPOT-VOL CROSS RR DECOMPOSITION</h2>')
                email_lines.append(f'<p style="color: #6b7280; font-size: 11px;">Generated: {now}</p>')

                # USD pair betas
                email_lines.append(f'<h3 style="font-size: 12px; color: #0d1117; margin-top: 16px;">USD PAIR BETAS</h3>')
                email_lines.append('<table style="border-collapse: collapse; width: 100%; font-family: Consolas, monospace; font-size: 11px;">')
                email_lines.append('<tr style="background: #0d1117; color: #e5e5e5;">')
                for h in ["Pair", "Tenor", "Vol%", "RR%", "Beta"]:
                    email_lines.append(f'<th style="padding: 6px 8px; text-align: right; border: 1px solid #d1d5db;">{h}</th>')
                email_lines.append('</tr>')
                for row in beta_rows:
                    email_lines.append('<tr>')
                    for i, cell in enumerate(row[:5]):
                        align = "left" if i == 0 else "right"
                        try:
                            v = float(cell.replace("+", ""))
                            color = "#22c55e" if v > 0 and i >= 3 else "#ef4444" if v < 0 and i >= 3 else "#1f2937"
                        except ValueError:
                            color = "#1f2937"
                        email_lines.append(f'<td style="padding: 4px 8px; text-align: {align}; border: 1px solid #e5e7eb; color: {color};">{cell}</td>')
                    email_lines.append('</tr>')
                email_lines.append('</table>')

                # Cross results
                email_lines.append(f'<h3 style="font-size: 12px; color: #0d1117; margin-top: 16px;">CROSS RR: PREDICTED vs MARKET</h3>')
                email_lines.append('<table style="border-collapse: collapse; width: 100%; font-family: Consolas, monospace; font-size: 11px;">')
                email_lines.append('<tr style="background: #0d1117; color: #e5e5e5;">')
                for h in cross_headers:
                    email_lines.append(f'<th style="padding: 6px 8px; text-align: right; border: 1px solid #d1d5db;">{h}</th>')
                email_lines.append('</tr>')
                for row in cross_rows:
                    email_lines.append('<tr>')
                    for i, cell in enumerate(row):
                        align = "left" if i <= 2 else "right"
                        try:
                            v = float(cell.replace("+", "").replace("%", ""))
                            color = "#22c55e" if v > 0 and i >= 4 else "#ef4444" if v < 0 and i >= 4 else "#1f2937"
                        except ValueError:
                            color = "#1f2937"
                        email_lines.append(f'<td style="padding: 4px 8px; text-align: {align}; border: 1px solid #e5e7eb; color: {color};">{cell}</td>')
                    email_lines.append('</tr>')
                email_lines.append('</table>')

                email_lines.append(f'<p style="color: #9ca3af; font-size: 10px; margin-top: 12px;">Beta = vol pts per 1% spot · SF=0.5 · Calendar days · Analytical Greeks</p>')
                email_lines.append('</div>')

                email_html = "\n".join(email_lines)

                # Store in session state for copy button
                st.session_state["email_report"] = email_html

                # Use a code block as a workaround for copy
                if st.button("📋 COPY EMAIL REPORT"):
                    st.code(email_html, language="html")
                    st.success("Email HTML above — select all and copy, then paste into Outlook/Gmail as HTML.")

            st.caption("Residual = Market RR − Predicted RR · Positive = cross RR cheap · Negative = rich")
