"""
Page 2 — Portfolio Construction

Builds and compares portfolios using four optimisation models.
Stores the chosen weights in session state for downstream analytics.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import MODEL_NAMES, MIN_WEIGHT, MAX_WEIGHT
from construction.optimiser import run_optimisation, efficient_frontier, compute_cov_matrix
from analytics.returns import (
    portfolio_returns, summary_statistics,
    cumulative_returns, annualised_return, annualised_volatility,
)
from ui.components.charts import (
    allocation_pie_chart, allocation_bar_chart,
    efficient_frontier_chart, cumulative_returns_chart,
)
from ui.components.metrics import (
    render_metric_row, render_weights_table, fmt_pct, fmt_ratio,
)

st.set_page_config(page_title="Portfolio Construction | Atlas PM", layout="wide")
st.title("Portfolio Construction")
st.markdown("Build and compare portfolios using institutional-grade optimisation models.")
st.markdown("---")

# ---------------------------------------------------------------------------
# Guard — require data
# ---------------------------------------------------------------------------
if "simple_returns" not in st.session_state:
    st.warning("Please load data first on the **Universe & Data** page.", icon="⚠️")
    st.stop()

returns = st.session_state["simple_returns"]
bench_returns = st.session_state["bench_returns"]
rf_returns    = st.session_state["rf_returns"]

# ---------------------------------------------------------------------------
# Sidebar — model configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model Configuration")

    model = st.selectbox(
        "Construction model",
        MODEL_NAMES,
        help="Choose the optimisation objective.",
    )

    st.markdown("---")
    st.markdown("**Constraints**")
    min_w = st.slider("Minimum weight per asset", 0.0, 0.20, MIN_WEIGHT, 0.01,
                      format="%.0f%%",
                      help="Minimum allocation to any single asset. 0% allows zero.")
    max_w = st.slider("Maximum weight per asset", 0.10, 1.00, MAX_WEIGHT, 0.05,
                      format="%.0f%%",
                      help="Maximum concentration in any single asset.")

    if model == "Maximum Sharpe":
        rf_rate = st.number_input(
            "Annual risk-free rate (%)",
            value=4.0, min_value=0.0, max_value=15.0, step=0.25,
        ) / 100
    else:
        rf_rate = 0.04

    st.markdown("---")
    optimise_btn = st.button("Optimise Portfolio", type="primary", use_container_width=True)
    compare_btn  = st.button("Compare All Models", use_container_width=True)

# ---------------------------------------------------------------------------
# Model descriptions
# ---------------------------------------------------------------------------
MODEL_DESCRIPTIONS = {
    "Equal Weight": """
**Equal Weight (1/N)** allocates identically to all selected assets.
- **Objective**: simplicity and maximum diversification by count
- **Strength**: surprisingly hard to beat out-of-sample; no estimation error
- **Limitation**: ignores all return/risk information; overweights high-vol assets by risk
    """,
    "Minimum Variance": """
**Minimum Variance** minimises portfolio volatility.
- **Objective**: minimize w'Σw subject to Σw=1
- **Strength**: no return estimate required; purely a risk model
- **Limitation**: tends to concentrate in low-volatility assets; ignores expected returns
    """,
    "Maximum Sharpe": """
**Maximum Sharpe** finds the portfolio on the efficient frontier with the highest Sharpe ratio.
- **Objective**: maximize (μ-rf) / σ
- **Strength**: theoretically optimal under mean-variance framework
- **Limitation**: highly sensitive to return estimates; estimation error leads to unstable weights
    """,
    "Risk Parity": """
**Risk Parity (Equal Risk Contribution)** equalises each asset's contribution to portfolio variance.
- **Objective**: w_i × (Σw)_i = σ²_p / N for all i
- **Strength**: true diversification at the risk level; no return estimates needed
- **Limitation**: typically overweights bonds/low-vol assets; requires leverage to compete on returns
    """,
}

col_desc, col_model = st.columns([1, 1])
with col_desc:
    st.subheader(model)
    st.markdown(MODEL_DESCRIPTIONS[model])

# ---------------------------------------------------------------------------
# Run single model optimisation
# ---------------------------------------------------------------------------
if optimise_btn or "current_weights" in st.session_state:
    if optimise_btn:
        with st.spinner(f"Running {model} optimisation…"):
            try:
                weights = run_optimisation(
                    model=model,
                    returns=returns,
                    rf_annual=rf_rate,
                    min_weight=min_w,
                    max_weight=max_w,
                )
                st.session_state["current_weights"]   = weights
                st.session_state["current_model"]     = model
                st.session_state["current_rf_rate"]   = rf_rate
            except Exception as e:
                st.error(f"Optimisation failed: {e}")
                st.stop()

    weights   = st.session_state.get("current_weights")
    model_lbl = st.session_state.get("current_model", model)

    if weights is None:
        st.info("Click **Optimise Portfolio** to run the model.")
        st.stop()

    # compute portfolio returns
    port_rets = portfolio_returns(returns, weights)
    rf_ann    = annualised_return(rf_returns)
    stats     = summary_statistics(port_rets, bench_returns, rf_returns, label=model_lbl)

    # store for downstream pages
    st.session_state["portfolio_returns"] = port_rets
    st.session_state["portfolio_stats"]   = stats
    st.session_state["current_weights"]   = weights

    st.markdown("---")
    st.subheader(f"Results: {model_lbl}")

    # key metrics row
    render_metric_row([
        {"label": "Annualised Return",    "value": fmt_pct(stats["Ann. Return"]),
         "help": "Geometric CAGR"},
        {"label": "Annualised Volatility","value": fmt_pct(stats["Ann. Volatility"]),
         "help": "Std dev × √252"},
        {"label": "Sharpe Ratio",         "value": fmt_ratio(stats["Sharpe Ratio"]),
         "help": "Excess return / vol"},
        {"label": "Sortino Ratio",        "value": fmt_ratio(stats["Sortino Ratio"]),
         "help": "Excess return / downside vol"},
        {"label": "Max Drawdown",         "value": fmt_pct(stats["Max Drawdown"]),
         "help": "Worst peak-to-trough decline"},
        {"label": "Beta (vs Benchmark)",  "value": fmt_ratio(stats.get("Beta", float("nan")), 3),
         "help": "Sensitivity to benchmark"},
    ])

    st.markdown("")
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.plotly_chart(
            allocation_pie_chart(weights, title=f"{model_lbl} — Allocation"),
            use_container_width=True,
        )

    with col_bar:
        st.subheader("Weights Table")
        render_weights_table(weights)

    st.subheader("Cumulative Performance vs Benchmark")
    bench_simple = bench_returns.reindex(port_rets.index)
    st.plotly_chart(
        cumulative_returns_chart(port_rets, bench_simple,
                                 title=f"{model_lbl} vs {st.session_state.get('benchmark_label','Benchmark')}"),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Compare all models
# ---------------------------------------------------------------------------
if compare_btn:
    st.markdown("---")
    st.subheader("Model Comparison")

    comparison_weights = {}
    comparison_stats   = []

    progress = st.progress(0)
    models_to_run = MODEL_NAMES

    for i, m in enumerate(models_to_run):
        try:
            w = run_optimisation(m, returns, rf_annual=rf_rate, min_weight=min_w, max_weight=max_w)
            comparison_weights[m] = w
            pr = portfolio_returns(returns, w)
            s  = summary_statistics(pr, bench_returns, rf_returns, label=m)
            comparison_stats.append(s)
        except Exception as e:
            st.warning(f"{m} failed: {e}")
        progress.progress((i + 1) / len(models_to_run))

    progress.empty()

    if comparison_stats:
        # summary comparison table
        comp_df = pd.DataFrame(comparison_stats).set_index("Label")
        display_cols = [
            "Ann. Return", "Ann. Volatility", "Sharpe Ratio",
            "Sortino Ratio", "Max Drawdown", "Beta",
        ]
        comp_df_display = comp_df[[c for c in display_cols if c in comp_df.columns]].copy()
        pct_cols   = ["Ann. Return", "Ann. Volatility", "Max Drawdown"]
        ratio_cols = ["Sharpe Ratio", "Sortino Ratio", "Beta"]
        for c in pct_cols:
            if c in comp_df_display:
                comp_df_display[c] = comp_df_display[c].map(fmt_pct)
        for c in ratio_cols:
            if c in comp_df_display:
                comp_df_display[c] = comp_df_display[c].map(fmt_ratio)

        st.dataframe(comp_df_display, use_container_width=True)

        # allocation comparison chart
        st.plotly_chart(
            allocation_bar_chart(comparison_weights, "Model Comparison — Asset Weights"),
            use_container_width=True,
        )

        # cumulative return comparison
        import plotly.graph_objects as go
        PALETTE = ["#1a3a5c","#2e86ab","#e84855","#f4a261"]
        fig = go.Figure()
        for i, (m, w) in enumerate(comparison_weights.items()):
            pr = portfolio_returns(returns, w)
            wealth = (1 + pr).cumprod()
            fig.add_trace(go.Scatter(
                x=wealth.index, y=wealth.values,
                name=m,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            ))
        bench_wealth = (1 + bench_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=bench_wealth.index, y=bench_wealth.values,
            name="Benchmark", line=dict(color="#999", width=1.5, dash="dash"),
        ))
        fig.update_layout(
            title="All Models — Cumulative Return",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Growth of £1",
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------
with st.expander("Efficient Frontier", expanded=False):
    st.markdown("""
The **efficient frontier** shows all minimum-variance portfolios at each target return level.
Portfolios below the frontier are suboptimal (you can get the same return with less risk).
The Maximum Sharpe portfolio sits at the tangency point with the capital market line.
    """)
    if st.button("Compute Efficient Frontier (takes ~10s)"):
        with st.spinner("Computing efficient frontier…"):
            try:
                frontier = efficient_frontier(returns, n_points=40, min_weight=min_w, max_weight=max_w)
                overlay  = {}
                if "current_weights" in st.session_state:
                    w  = st.session_state["current_weights"]
                    pr = portfolio_returns(returns, w)
                    overlay[model_lbl] = (
                        annualised_volatility(pr),
                        annualised_return(pr),
                    )
                st.plotly_chart(
                    efficient_frontier_chart(frontier, overlay),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Frontier computation failed: {e}")
