"""
Page 4 — Risk Management

Covers:
- Historical and parametric VaR / CVaR at multiple horizons
- Return distribution with VaR overlay
- Risk contribution by asset
- Correlation matrix
- Stress test scenarios
- VaR back-test (Kupiec test)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import numpy as np

from analytics.risk import (
    historical_var, historical_cvar, parametric_var,
    var_summary, component_risk_contribution, correlation_matrix,
    run_stress_test, var_backtesting,
)
from construction.optimiser import compute_cov_matrix
from ui.components.charts import (
    correlation_heatmap, risk_contribution_bar,
    stress_test_bar, var_distribution_chart,
)
from ui.components.metrics import (
    render_metric_row, fmt_pct, fmt_currency, fmt_ratio,
)
from config.settings import VAR_CONFIDENCE, ASSET_SHORT_NAMES

st.set_page_config(page_title="Risk Management | Atlas PM", layout="wide")
st.title("Risk Management")
st.markdown("---")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if "portfolio_returns" not in st.session_state:
    st.warning("Please complete **Portfolio Construction** first.", icon="⚠️")
    st.stop()

port_rets    = st.session_state["portfolio_returns"]
simple_rets  = st.session_state["simple_returns"]
weights      = st.session_state["current_weights"]
model_lbl    = st.session_state.get("current_model", "Portfolio")

# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Risk Configuration")
    confidence     = st.slider("VaR Confidence Level", 0.90, 0.99, VAR_CONFIDENCE, 0.01,
                               format="%.0f%%")
    portfolio_value = st.number_input("Portfolio Value (£)", value=1_000_000,
                                      min_value=10_000, step=100_000, format="%d")
    st.markdown("---")
    st.caption(f"Analysis for: **{model_lbl}**")

# ---------------------------------------------------------------------------
# Key risk metrics row
# ---------------------------------------------------------------------------
var_1d   = historical_var(port_rets, confidence, 1)
cvar_1d  = historical_cvar(port_rets, confidence, 1)
var_10d  = historical_var(port_rets, confidence, 10)
ann_vol  = float(port_rets.std(ddof=1) * np.sqrt(252))
max_dd   = float(((1 + port_rets).cumprod() / (1 + port_rets).cumprod().cummax() - 1).min())

st.subheader("Key Risk Metrics")
render_metric_row([
    {"label": f"1-Day VaR ({int(confidence*100)}%)",  "value": fmt_pct(var_1d),
     "help": f"1-day loss not expected to be exceeded with {confidence:.0%} probability"},
    {"label": f"1-Day CVaR ({int(confidence*100)}%)", "value": fmt_pct(cvar_1d),
     "help": "Expected loss given that VaR is breached (Expected Shortfall)"},
    {"label": f"10-Day VaR ({int(confidence*100)}%)", "value": fmt_pct(var_10d),
     "help": "10-day VaR using square-root-of-time scaling"},
    {"label": "Ann. Volatility",                      "value": fmt_pct(ann_vol),
     "help": "Annualised standard deviation"},
    {"label": "Max Drawdown",                         "value": fmt_pct(max_dd),
     "help": "Largest peak-to-trough decline"},
    {"label": f"VaR (£1M notional)",                 "value": fmt_currency(var_1d * portfolio_value),
     "help": f"1-day {confidence:.0%} VaR in £ terms"},
])

st.markdown("---")

tab_var, tab_contrib, tab_corr, tab_stress, tab_backtest = st.tabs([
    "VaR / CVaR",
    "Risk Contribution",
    "Correlations",
    "Stress Tests",
    "VaR Back-test",
])

# ---------------------------------------------------------------------------
# Tab 1: VaR / CVaR
# ---------------------------------------------------------------------------
with tab_var:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("VaR / CVaR Summary")
        var_df = var_summary(port_rets, portfolio_value, confidence)

        display = var_df.copy()
        for col in ["VaR (%)", "CVaR / ES (%)"]:
            display[col] = display[col].map(lambda v: fmt_pct(v) if not pd.isna(v) else "N/A")
        for col in ["VaR (£)", "CVaR / ES (£)"]:
            display[col] = display[col].map(lambda v: fmt_currency(v) if not pd.isna(v) else "N/A")

        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("""
**Interpretation:**
- **Historical VaR** uses the actual empirical distribution — no distributional assumption
- **Parametric VaR** assumes normally distributed returns — typically underestimates tail risk
- **CVaR (Expected Shortfall)** is the average loss in the tail beyond VaR — it is a *coherent* risk measure
- **10-day VaR** is scaled using √10 (standard regulatory approach; assumes i.i.d. returns)

The gap between historical and parametric VaR reflects the fat-tailed nature of financial returns.
        """)

    with col2:
        st.subheader("Return Distribution")
        fig = var_distribution_chart(port_rets, var_1d, cvar_1d, confidence)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info(
        "**Model limitations**: VaR assumes stationarity of the return distribution. "
        "It does not capture liquidity risk, correlation breakdown during stress, or "
        "model/operational risk. The square-root-of-time scaling for multi-day VaR "
        "underestimates risk when returns are serially correlated or fat-tailed.",
        icon="⚠️"
    )

# ---------------------------------------------------------------------------
# Tab 2: Risk Contribution
# ---------------------------------------------------------------------------
with tab_contrib:
    st.subheader("Component Risk Contribution")
    st.markdown("""
Which assets are actually driving portfolio risk?

**Component Risk Contribution** = w_i × MRC_i, where MRC is the marginal risk contribution.
It sums to total portfolio volatility and shows which assets "punch above their weight" in risk terms.
    """)

    # align weights to available returns
    w_aligned = weights.reindex(simple_rets.columns).dropna()
    w_aligned = w_aligned / w_aligned.sum()

    cov = compute_cov_matrix(simple_rets[w_aligned.index])
    risk_df = component_risk_contribution(w_aligned, cov)

    st.plotly_chart(
        risk_contribution_bar(risk_df, title="Risk vs Weight Comparison"),
        use_container_width=True,
    )

    # formatted table
    risk_display = risk_df.copy()
    risk_display["Weight"]              = risk_display["Weight"].map(fmt_pct)
    risk_display["Marginal RC"]         = risk_display["Marginal RC"].map(lambda v: f"{v:.4f}")
    risk_display["Component RC"]        = risk_display["Component RC"].map(lambda v: f"{v:.4f}")
    risk_display["% Risk Contribution"] = risk_display["% Risk Contribution"].map(fmt_pct)
    st.dataframe(risk_display, use_container_width=True)

    st.caption(
        "If % Risk Contribution >> Weight, the asset is contributing disproportionate risk. "
        "Risk parity targets equality between these two columns."
    )

# ---------------------------------------------------------------------------
# Tab 3: Correlations
# ---------------------------------------------------------------------------
with tab_corr:
    st.subheader("Asset Correlation Matrix")
    st.markdown("""
The correlation matrix shows how assets co-move. Low or negative correlations
provide diversification benefits. During market crises, many correlations spike
toward 1.0 — a phenomenon known as **correlation breakdown**, which reduces the
benefit of diversification precisely when it is most needed.
    """)

    corr = correlation_matrix(simple_rets)
    st.plotly_chart(
        correlation_heatmap(corr, title="Pairwise Pearson Correlations (Daily Returns)"),
        use_container_width=True,
    )

    # rolling correlation between portfolio and benchmark
    st.subheader("Rolling Correlation: Portfolio vs Benchmark")
    bench_rets = st.session_state["bench_returns"]
    bench_aligned = bench_rets.reindex(port_rets.index).dropna()

    window = st.slider("Rolling window", 21, 252, 63, 21, key="corr_window")
    roll_corr = port_rets.rolling(window).corr(bench_aligned)

    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        x=roll_corr.index, y=roll_corr.values,
        line=dict(color="#1a3a5c", width=2),
        fill="tozeroy",
        fillcolor="rgba(26,58,92,0.10)",
        name=f"Rolling Corr ({window}d)",
        hovertemplate="%{x|%b %Y}<br>ρ = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Correlation", yaxis_range=[-1.1, 1.1],
        font=dict(family="Inter, Arial, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Stress Tests
# ---------------------------------------------------------------------------
with tab_stress:
    st.subheader("Stress Test Scenarios")
    st.markdown("""
Stress tests apply instantaneous, historically-calibrated shocks to the current portfolio weights.
These are not forecasts — they are planning tools to understand portfolio sensitivity to specific adverse scenarios.

**Shock methodology**: shocks are applied as simultaneous one-day returns across all assets.
This ignores crisis-era correlation dynamics (correlations typically rise during stress),
so results should be treated as indicative rather than precise loss estimates.
    """)

    weights_dict = weights.to_dict()
    stress_df    = run_stress_test(weights_dict)

    st.plotly_chart(
        stress_test_bar(stress_df, title="Stress Test Results — Portfolio P&L"),
        use_container_width=True,
    )

    # detailed table
    st.subheader("Scenario Detail")
    pnl_col = stress_df["Portfolio P&L"]
    display = stress_df.copy()

    for col in display.columns:
        display[col] = display[col].map(lambda v: fmt_pct(v) if not pd.isna(v) else "-")

    st.dataframe(display, use_container_width=True)

    # worst scenario
    worst_scenario = pnl_col.idxmin()
    worst_pnl      = pnl_col.min()
    best_scenario  = pnl_col.idxmax()
    best_pnl       = pnl_col.max()

    col1, col2 = st.columns(2)
    col1.metric("Most Damaging Scenario", worst_scenario,
                delta=fmt_pct(worst_pnl), delta_color="inverse")
    col2.metric("Most Favourable Scenario", best_scenario,
                delta=fmt_pct(best_pnl))

    st.info(
        "**Limitations**: Shocks are applied as simultaneous instantaneous losses. "
        "Real crisis events unfold over days/weeks with non-linear feedback loops. "
        "Correlation increases during stress reduce the diversification benefit shown here. "
        "The scenarios are approximate historical analogues, not precise replications.",
        icon="⚠️"
    )

# ---------------------------------------------------------------------------
# Tab 5: VaR Back-test
# ---------------------------------------------------------------------------
with tab_backtest:
    st.subheader("VaR Back-test (Kupiec Test)")
    st.markdown(f"""
A VaR model should be **well-calibrated**: the actual breach rate should match the
theoretical rate.

At {confidence:.0%} confidence, we expect the VaR to be breached
**{(1-confidence):.0%} of days** = approximately **{int((1-confidence)*252)} days/year**.

If the actual breach rate is significantly higher, the model underestimates risk.
If significantly lower, the model is conservative.
    """)

    window = st.slider("Estimation window (trading days)", 126, 504, 252, 63,
                       key="backtest_window")

    with st.spinner("Running back-test…"):
        backtest_df = var_backtesting(port_rets, confidence, window)

    if len(backtest_df) > 0:
        expected_rate = 1 - confidence
        actual_rate   = backtest_df.attrs["actual_breach_rate"]
        n_breaches    = backtest_df.attrs["n_breaches"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Breach Rate", fmt_pct(expected_rate))
        col2.metric("Actual Breach Rate",   fmt_pct(actual_rate),
                    delta=fmt_pct(actual_rate - expected_rate),
                    delta_color="inverse")
        col3.metric("Total Breaches", str(n_breaches))

        calibration = "well-calibrated ✅" if abs(actual_rate - expected_rate) < 0.01 else \
                      "conservative (under-reporting risk)" if actual_rate < expected_rate else \
                      "under-estimating tail risk ⚠️"
        st.markdown(f"**Kupiec assessment:** The model appears **{calibration}**")

        # breach chart
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest_df.index, y=-backtest_df["VaR Estimate"] * 100,
            name=f"VaR ({confidence:.0%})",
            line=dict(color="#f4a261", dash="dash", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=backtest_df.index, y=backtest_df["Actual Return"] * 100,
            name="Actual Return",
            line=dict(color="#1a3a5c", width=1),
            opacity=0.6,
        ))
        breaches = backtest_df[backtest_df["Breach"]]
        fig.add_trace(go.Scatter(
            x=breaches.index, y=breaches["Actual Return"] * 100,
            mode="markers",
            marker=dict(color="#e84855", size=6, symbol="x"),
            name="VaR Breach",
        ))
        fig.update_layout(
            title=f"VaR Back-test ({window}-day rolling estimation window)",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Daily Return (%)", yaxis_ticksuffix="%",
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
