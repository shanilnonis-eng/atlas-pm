"""
Page 1 — Universe & Data

The user selects which assets to include, the analysis window,
and triggers the data download.  Results are stored in st.session_state
so all other pages can access them without re-downloading.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import streamlit as st
import pandas as pd
import plotly.express as px

from config.settings import (
    UNIVERSE, BENCHMARK_LABEL, DEFAULT_START_DATE,
    ASSET_SHORT_NAMES, ACCENT_COLOR,
)
from data.loader import (
    cached_load_prices, cached_load_benchmark, cached_load_risk_free,
    compute_returns,
)

st.set_page_config(page_title="Universe & Data | Atlas PM", layout="wide")
st.title("Investment Universe & Data")
st.markdown("Select your asset universe and analysis window, then load historical price data.")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — asset selection and date range
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Universe Configuration")

    selected_labels = st.multiselect(
        "Select assets",
        options=list(UNIVERSE.keys()),
        default=list(UNIVERSE.keys()),
        help="Choose at least 3 assets to enable optimisation models.",
    )

    st.markdown("---")
    start_date = st.date_input("Start date", value=pd.to_datetime(DEFAULT_START_DATE))
    end_date   = st.date_input("End date",   value=pd.to_datetime("today"))

    if start_date >= end_date:
        st.error("Start date must be before end date.")

    st.markdown("---")
    load_btn = st.button("Load Data", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main panel — info and load
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Selected Universe")
    if selected_labels:
        universe_df = pd.DataFrame([
            {
                "Asset Class": label,
                "ETF Proxy":   UNIVERSE[label],
                "Short Code":  ASSET_SHORT_NAMES.get(label, ""),
            }
            for label in selected_labels
        ])
        st.dataframe(universe_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Please select at least one asset.")

with col2:
    st.subheader("Why ETF proxies?")
    st.markdown("""
ETFs are used instead of individual stocks for several institutional reasons:

- **Diversification**: each ETF represents a broad index, not single-stock risk
- **Survivorship bias mitigation**: index funds include new constituents as they enter
- **Clean pricing**: adjusted close prices account for dividends and splits
- **Liquidity**: major ETFs trade billions daily with tight bid/offer spreads
- **Data availability**: consistent daily data back to ~2005–2010

This is how **real fund-of-funds** and multi-asset managers think about asset class exposure.
    """)

# ---------------------------------------------------------------------------
# Load button logic
# ---------------------------------------------------------------------------
if load_btn:
    if len(selected_labels) < 2:
        st.error("Please select at least 2 assets.")
        st.stop()

    selected_universe = {k: UNIVERSE[k] for k in selected_labels}
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    try:
        with st.spinner("Downloading price data from Yahoo Finance…"):
            prices    = cached_load_prices(json.dumps(selected_universe), start_str, end_str)
            benchmark = cached_load_benchmark(start_str, end_str)
            rf        = cached_load_risk_free(start_str, end_str)

        # align all to common dates
        all_data = prices.join(benchmark.rename("__BENCH__"), how="inner")
        all_data = all_data.join(rf.rename("__RF__"), how="inner")
        prices_aligned    = all_data[[c for c in all_data.columns if not c.startswith("__")]]
        benchmark_aligned = all_data["__BENCH__"]
        rf_aligned        = all_data["__RF__"]

        simple_returns, log_returns = compute_returns(prices_aligned)
        bench_simple, _             = compute_returns(benchmark_aligned.to_frame())
        rf_simple, _                = compute_returns(rf_aligned.to_frame())

        bench_returns = bench_simple.iloc[:, 0]
        rf_returns    = rf_simple.iloc[:, 0]

        # store everything in session state
        st.session_state["prices"]           = prices_aligned
        st.session_state["simple_returns"]   = simple_returns
        st.session_state["log_returns"]      = log_returns
        st.session_state["bench_returns"]    = bench_returns
        st.session_state["rf_returns"]       = rf_returns
        st.session_state["selected_assets"]  = selected_labels
        st.session_state["selected_universe"]= selected_universe
        st.session_state["start_date"]       = start_str
        st.session_state["end_date"]         = end_str

        st.success(
            f"Loaded {len(prices_aligned)} trading days of data for "
            f"{len(prices_aligned.columns)} assets  "
            f"({prices_aligned.index[0].date()} → {prices_aligned.index[-1].date()})"
        )

    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Display loaded data (if available in session)
# ---------------------------------------------------------------------------
if "prices" in st.session_state:
    prices = st.session_state["prices"]
    simple_returns = st.session_state["simple_returns"]

    st.markdown("---")
    st.subheader("Price Data Overview")

    # data quality table
    quality_rows = []
    for col in prices.columns:
        series = prices[col].dropna()
        quality_rows.append({
            "Asset":         col,
            "Start":         series.index[0].date(),
            "End":           series.index[-1].date(),
            "Trading Days":  len(series),
            "Missing (%)":   f"{prices[col].isna().mean():.1%}",
            "Current Price": f"${series.iloc[-1]:.2f}",
        })

    st.dataframe(
        pd.DataFrame(quality_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Normalised Price History (rebased to 100)")
    rebased = prices / prices.iloc[0] * 100
    fig = px.line(
        rebased,
        labels={"value": "Rebased Price (100 = start)", "variable": "Asset"},
        color_discrete_sequence=[
            "#1a3a5c","#2e86ab","#e84855","#f4a261","#57cc99",
            "#9b5de5","#f72585","#4cc9f0","#b5e48c","#ffb703",
        ],
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="",
        font=dict(family="Inter, Arial, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Descriptive Statistics — Daily Returns")
    desc = simple_returns.describe().T
    desc.columns = ["Count","Mean","Std Dev","Min","25%","Median","75%","Max"]
    for col in ["Mean","Std Dev","Min","25%","Median","75%","Max"]:
        desc[col] = desc[col].map(lambda v: f"{v:.4%}")
    desc["Count"] = desc["Count"].astype(int)
    st.dataframe(desc, use_container_width=True)

    st.info(
        "Data loaded successfully. Navigate to **Portfolio Construction** to build your portfolio.",
        icon="✅"
    )
else:
    st.info(
        "Configure your universe above and click **Load Data** to begin.",
        icon="👆"
    )
