"""
Page 8 — Factor Attribution & Transaction Cost Analysis

Two sections:
1. Fama-French 3-factor attribution — decomposes portfolio alpha
2. Transaction cost & turnover analysis — costs across rebalancing frequencies
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from analytics.factors import (
    load_ff3_factors, synthetic_factors,
    run_factor_regression, regression_table, rolling_factor_betas,
)
from analytics.turnover import (
    simulate_rebalancing, turnover_comparison,
    ETF_SPREADS_BPS,
)
from ui.components.metrics import fmt_pct, fmt_ratio
from config.settings import UNIVERSE, ACCENT_COLOR

st.set_page_config(page_title="Factor Attribution | Atlas PM", layout="wide")
st.title("Factor Attribution & Transaction Costs")
st.markdown("---")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if "portfolio_returns" not in st.session_state:
    st.warning("Please complete **Portfolio Construction** first.", icon="⚠️")
    st.stop()

port_rets   = st.session_state["portfolio_returns"]
bench_rets  = st.session_state["bench_returns"]
rf_rets     = st.session_state["rf_returns"]
weights     = st.session_state["current_weights"]
model_lbl   = st.session_state.get("current_model", "Portfolio")
prices      = st.session_state["prices"]
start_str   = st.session_state.get("start_date", "2015-01-01")
end_str     = st.session_state.get("end_date", None)

tab_ff, tab_costs = st.tabs([
    "Fama-French Factor Attribution",
    "Transaction Cost Analysis",
])

# ===========================================================================
# TAB 1: Fama-French
# ===========================================================================
with tab_ff:
    st.subheader("Fama-French 3-Factor Attribution")
    st.markdown("""
    Decompose your portfolio's returns into three systematic risk factors:

    | Factor | Label | What it captures |
    |--------|-------|-----------------|
    | **Market** | Mkt-RF | Excess return of the broad equity market |
    | **Size**   | SMB (Small Minus Big) | Premium for holding small-cap stocks |
    | **Value**  | HML (High Minus Low)  | Premium for holding value (cheap) stocks |

    The regression: `R_p - R_f = α + β_mkt(Mkt-RF) + β_smb(SMB) + β_hml(HML) + ε`

    **Alpha (α)**: the return not explained by factor exposure — true skill if statistically significant.
    Low or negative alpha means performance was driven by factor tilts, not manager skill.
    """)

    with st.spinner("Fetching Fama-French factor data from Ken French Data Library…"):
        ff_factors = load_ff3_factors(start_str, end_str or "2024-12-31", frequency="monthly")

    if ff_factors is None:
        st.warning(
            "Could not fetch Fama-French data from the Ken French library. "
            "Using synthetic proxies (market factor only — SMB and HML set to zero). "
            "Install `pandas-datareader` and ensure internet access for full factor data.",
            icon="⚠️"
        )
        ff_factors = synthetic_factors(bench_rets, rf_rets, frequency="monthly")
        data_source = "Synthetic (market factor only)"
    else:
        data_source = "Ken French Data Library"

    st.caption(f"Factor data source: **{data_source}** | {len(ff_factors)} monthly observations")

    with st.spinner("Running factor regression…"):
        result = run_factor_regression(port_rets, ff_factors, frequency="monthly")
        reg_table = regression_table(result)

    # ------------------------------------------------------------------
    # Regression output
    # ------------------------------------------------------------------
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Regression Results")
        st.dataframe(reg_table, use_container_width=True, hide_index=True)

        st.markdown(f"""
        **R² = {result['r_squared']:.3f}** | Adjusted R² = {result['r_squared_adj']:.3f} | n = {result['n_observations']} months

        Significance: *** p<0.01 · ** p<0.05 · * p<0.10
        """)

    with col2:
        st.subheader("Summary Interpretation")

        alpha_ann = result["alpha_annual"]
        beta_mkt  = result["beta_mkt"]
        beta_smb  = result["beta_smb"]
        beta_hml  = result["beta_hml"]
        r2        = result["r_squared"]

        alpha_sig = "✅ Statistically significant" if result["p_alpha"] < 0.10 else "Not significant"

        st.metric("Annualised Alpha (α)", fmt_pct(alpha_ann),
                  help="Return unexplained by factors (annualised)")
        st.metric("Market Beta (β_mkt)", fmt_ratio(beta_mkt, 3),
                  help="> 1 = more market risk than benchmark")
        st.metric("Size Beta (β_smb)", fmt_ratio(beta_smb, 3),
                  help="> 0 = small-cap tilt, < 0 = large-cap tilt")
        st.metric("Value Beta (β_hml)", fmt_ratio(beta_hml, 3),
                  help="> 0 = value tilt, < 0 = growth tilt")
        st.metric("R² (explained variation)", f"{r2:.1%}",
                  help="% of return variation explained by the three factors")
        st.caption(f"Alpha significance: {alpha_sig}")

    # ------------------------------------------------------------------
    # Return decomposition chart
    # ------------------------------------------------------------------
    st.subheader("Return Decomposition")

    contribs = result["contributions"]
    total = sum(abs(v) for v in contribs.values() if isinstance(v, float))

    # monthly contributions bar
    labels = ["Alpha (monthly)", "Market", "Size (SMB)", "Value (HML)"]
    values = [contribs[l] * 100 for l in labels]
    colors = [
        "#1a3a5c" if l == "Alpha (monthly)" else
        "#2e86ab" if l == "Market" else
        "#f4a261" if l == "Size (SMB)" else "#57cc99"
        for l in labels
    ]

    fig_decomp = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        hovertemplate="%{x}<br>Avg monthly contribution: %{y:.4f}%<extra></extra>",
    ))
    fig_decomp.add_hline(y=0, line_color="grey", line_dash="dot")
    fig_decomp.update_layout(
        title="Average Monthly Return Contribution by Factor",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Avg Monthly Contribution (%)",
        font=dict(family="Inter, Arial, sans-serif"),
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

    # ------------------------------------------------------------------
    # Rolling factor betas
    # ------------------------------------------------------------------
    st.subheader("Rolling Factor Betas (24-month window)")

    window = st.slider("Rolling window (months)", 12, 36, 24, key="ff_window")

    with st.spinner("Computing rolling betas…"):
        rolling = rolling_factor_betas(port_rets, ff_factors, window=window, frequency="monthly")

    if not rolling.empty:
        fig_roll = go.Figure()
        for col, color in [("Mkt-RF β","#1a3a5c"), ("SMB β","#f4a261"), ("HML β","#57cc99"), ("Alpha","#e84855")]:
            if col in rolling.columns:
                fig_roll.add_trace(go.Scatter(
                    x=rolling.index, y=rolling[col].values,
                    name=col,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{col}: %{{y:.3f}}<extra></extra>",
                ))
        fig_roll.add_hline(y=0, line_color="grey", line_dash="dot")
        fig_roll.update_layout(
            title=f"Rolling Factor Exposures ({window}-month window)",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Beta / Alpha",
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    # Residual plot
    st.subheader("Unexplained Residuals (ε)")
    residuals = result["residuals"]
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Bar(
        x=residuals.index, y=residuals.values * 100,
        marker_color=[
            "#57cc99" if v >= 0 else "#e84855"
            for v in residuals.values
        ],
        name="Residual",
        hovertemplate="%{x|%b %Y}<br>Residual: %{y:.2f}%<extra></extra>",
    ))
    fig_resid.add_hline(y=0, line_color="grey", line_dash="dot")
    fig_resid.update_layout(
        title="Factor Model Residuals (return unexplained by Fama-French factors)",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Monthly Residual (%)", yaxis_ticksuffix="%",
        font=dict(family="Inter, Arial, sans-serif"),
    )
    st.plotly_chart(fig_resid, use_container_width=True)

    st.caption("""
    **Interpretation guide:**
    - **α > 0 and significant**: portfolio is generating returns beyond factor exposure — potential manager skill
    - **β_mkt ≈ 1**: moves in line with broad market
    - **β_smb > 0**: tilted toward smaller companies
    - **β_hml > 0**: tilted toward value (cheap) companies
    - **High R²**: performance mostly explained by systematic factors (not skill)
    - **Low R²**: significant idiosyncratic element — could be positive (alpha) or negative (noise)
    """)


# ===========================================================================
# TAB 2: Transaction Costs
# ===========================================================================
with tab_costs:
    st.subheader("Transaction Cost & Turnover Analysis")
    st.markdown("""
    Real portfolios incur costs when rebalancing. This page quantifies the drag
    from bid-offer spreads and shows how rebalancing frequency affects net returns.

    **Method**: we simulate the portfolio with periodic rebalancing, deducting
    estimated ETF bid-offer spreads on each trade. We compare gross (no-cost)
    vs net (after-cost) cumulative returns.
    """)

    # Spread table
    with st.expander("ETF Bid-Offer Spread Assumptions"):
        spread_rows = []
        selected_universe = st.session_state.get("selected_universe", UNIVERSE)
        for label, ticker in selected_universe.items():
            bps = ETF_SPREADS_BPS.get(ticker, 5.0)
            spread_rows.append({
                "Asset": label, "Ticker": ticker,
                "Est. Spread (bps)": bps,
                "Est. Spread (%)": f"{bps/100:.3f}%",
            })
        st.dataframe(pd.DataFrame(spread_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Spreads are approximate estimates as of 2024. Actual spreads vary with trade size, "
            "time of day, and market conditions. These assume retail/small institutional order size."
        )

    # Rebalancing simulation
    st.subheader("Gross vs Net Performance by Rebalancing Frequency")

    selected_universe = st.session_state.get("selected_universe", UNIVERSE)

    with st.spinner("Simulating rebalancing scenarios…"):
        try:
            comp_df = turnover_comparison(prices, weights, selected_universe)

            display = comp_df.copy()
            for col in ["Gross Ann. Return", "Net Ann. Return", "Annual Cost Drag"]:
                display[col] = display[col].map(fmt_pct)
            display["Avg Cost/Rebal (bps)"] = display["Avg Cost/Rebal (bps)"].map(lambda v: f"{v:.2f}")
            display["Avg Turnover"] = display["Avg Turnover"].map(lambda v: f"{v:.1%}")

            st.dataframe(display, use_container_width=True, hide_index=True)

            # detail chart for monthly
            monthly_result = simulate_rebalancing(prices, weights, "ME", selected_universe)

            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                x=monthly_result["gross_wealth"].index,
                y=monthly_result["gross_wealth"].values,
                name="Gross (no costs)",
                line=dict(color="#2e86ab", width=2, dash="dash"),
            ))
            fig_cost.add_trace(go.Scatter(
                x=monthly_result["net_wealth"].index,
                y=monthly_result["net_wealth"].values,
                name="Net (after costs)",
                line=dict(color=ACCENT_COLOR, width=2),
            ))
            fig_cost.update_layout(
                title="Monthly Rebalancing: Gross vs Net Cumulative Return",
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="Growth of £1",
                font=dict(family="Inter, Arial, sans-serif"),
            )
            st.plotly_chart(fig_cost, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Cost Drag (monthly rebal.)",
                        fmt_pct(monthly_result["total_cost_drag"]))
            col2.metric("Annual Cost Drag",
                        fmt_pct(monthly_result["annual_cost_drag"]))
            col3.metric("Avg Turnover per Rebalance",
                        f"{monthly_result['avg_turnover']:.1%}")

            # Rebalancing log
            if not monthly_result["rebalancing_log"].empty:
                with st.expander("Rebalancing Log (monthly)", expanded=False):
                    log = monthly_result["rebalancing_log"].copy()
                    log["Date"] = log["Date"].dt.strftime("%b %Y")
                    log["Turnover"] = log["Turnover"].map(lambda v: f"{v:.1%}")
                    log["Cost (bps)"] = log["Cost (bps)"].map(lambda v: f"{v:.2f}")
                    log["Portfolio Value"] = log["Portfolio Value"].map(lambda v: f"£{v:,.0f}")
                    st.dataframe(log, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Simulation failed: {e}")

    st.markdown("---")
    st.info("""
**What does this mean for portfolio construction?**

- Monthly rebalancing has higher turnover but keeps the portfolio close to target weights
- Annual rebalancing has lower costs but allows significant drift from targets
- The optimal frequency depends on the portfolio's drift rate (volatile assets drift faster)
- Risk parity and minimum variance portfolios typically have higher turnover than equal weight
- Transaction costs are a real drag: 10–20 bps/year is typical for this ETF universe
- For a £10M portfolio, 10 bps/year = £10,000/year in avoidable costs

**Limitations:**
Market impact (the cost of moving the market with your own trade) is not modelled.
At institutional size (>£100M), market impact can exceed bid-offer costs.
    """, icon="💡")
