"""
Page 7 — Black-Litterman Model

Interactive Black-Litterman implementation.
Users express manager views and see how they tilt the portfolio from equilibrium.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from construction.black_litterman import BlackLitterman, View
from construction.optimiser import run_optimisation
from analytics.returns import portfolio_returns, summary_statistics, annualised_return, annualised_volatility
from ui.components.charts import allocation_pie_chart, allocation_bar_chart, cumulative_returns_chart
from ui.components.metrics import render_metric_row, fmt_pct, fmt_ratio
from config.settings import TRADING_DAYS_PER_YEAR, ACCENT_COLOR

st.title("Black-Litterman Model")
st.markdown("Express your views. See how they shift the portfolio from market equilibrium.")
st.markdown("---")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if "simple_returns" not in st.session_state:
    st.warning("Please load data first on the **Universe & Data** page.", icon="⚠️")
    st.stop()

returns = st.session_state["simple_returns"]
assets  = list(returns.columns)
bench_rets = st.session_state["bench_returns"]
rf_rets    = st.session_state["rf_returns"]

# ---------------------------------------------------------------------------
# Educational explainer
# ---------------------------------------------------------------------------
with st.expander("What is Black-Litterman? (click to read)", expanded=False):
    st.markdown("""
### The problem with standard mean-variance optimisation

When you maximise the Sharpe ratio using historical return estimates, the optimiser
produces extreme, unstable portfolios — 80% in one asset, 0% in most others.
This happens because the inputs (expected returns) are very imprecise estimates,
and the optimiser magnifies that noise.

### The Black-Litterman solution (Goldman Sachs, 1990)

Instead of using raw historical returns, BL starts from a **neutral equilibrium** —
the implied returns that make the reference portfolio (e.g. equal weight) optimal.
These equilibrium returns are stable and diversified by construction.

Then it **blends** your specific manager views into that equilibrium, weighted by
how confident you are in each view:
- **High confidence** → views dominate, portfolio tilts strongly
- **Low confidence** → equilibrium dominates, portfolio stays near neutral
- **No views** → pure equilibrium = reference portfolio

### The maths (simplified)

```
Equilibrium returns:  π  = λ × Σ × w_ref      (reverse-optimise reference portfolio)
Manager views:        P × μ = Q ± uncertainty  (K views expressed as return targets)
Posterior:            μ_BL = blend(π, views)   (Bayesian update)
BL portfolio:         MaxSharpe on μ_BL        (stable, view-tilted weights)
```

The result is a portfolio that is **diversified by default** but reflects your
conviction where you have it. This is how Goldman Sachs, BlackRock, and most
large asset managers think about strategic asset allocation.
    """)

# ---------------------------------------------------------------------------
# Sidebar: model parameters and reference portfolio
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("BL Parameters")

    risk_aversion = st.slider(
        "Risk aversion (λ)",
        min_value=1.0, max_value=6.0, value=2.5, step=0.5,
        help="Higher λ = more conservative equilibrium. Typical range: 2-3.5"
    )
    tau = st.select_slider(
        "Tau (τ) — equilibrium uncertainty",
        options=[0.01, 0.025, 0.05, 0.075, 0.10],
        value=0.05,
        help="How uncertain are you about equilibrium? Smaller τ = equilibrium dominates views more."
    )

    st.markdown("---")
    st.markdown("**Reference Portfolio**")
    ref_model = st.selectbox(
        "Starting point",
        ["Equal Weight", "Minimum Variance", "Risk Parity", "Custom"],
        help="The 'neutral' portfolio that defines equilibrium. Your views tilt away from this."
    )

    if ref_model == "Custom":
        st.markdown("Set custom reference weights:")
        ref_weights_dict = {}
        for asset in assets:
            ref_weights_dict[asset] = st.number_input(
                asset, 0.0, 1.0, 1.0/len(assets), 0.01, key=f"ref_{asset}", format="%.2f"
            )
        ref_w = pd.Series(ref_weights_dict)
        ref_w = ref_w / ref_w.sum()
    else:
        with st.spinner("Computing reference weights…"):
            ref_w = run_optimisation(ref_model, returns)

    st.markdown("---")
    st.caption(f"Reference: **{ref_model}**")
    for a, w in ref_w.sort_values(ascending=False).items():
        if w > 0.01:
            st.caption(f"  {a}: {w:.1%}")

# ---------------------------------------------------------------------------
# Instantiate the model
# ---------------------------------------------------------------------------
bl = BlackLitterman(
    returns=returns,
    reference_weights=ref_w,
    risk_aversion=risk_aversion,
    tau=tau,
)

# ---------------------------------------------------------------------------
# View builder
# ---------------------------------------------------------------------------
st.subheader("Express Your Views")
st.markdown("""
Add manager views below. Each view shifts the portfolio from the equilibrium.
You can express **absolute** views (expected return for one asset) or
**relative** views (expected outperformance of one asset vs another).
""")

# session state for views list
if "bl_views" not in st.session_state:
    st.session_state["bl_views"] = []

# --- Add view form ---
with st.form("add_view_form", clear_on_submit=True):
    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 2, 1.5, 1])

    view_type = col1.selectbox("View type", ["Absolute", "Relative"],
                               help="Absolute: one asset. Relative: asset A outperforms B.")
    long_asset = col2.selectbox("Asset (long)", assets)
    short_asset = col3.selectbox("vs Asset (short)", ["None"] + assets,
                                 help="For relative views only")
    view_return = col4.number_input(
        "Expected return (%/yr)", -20.0, 50.0, 8.0, 0.5,
        help="Annualised expected return or outperformance"
    )
    confidence = col5.slider("Confidence", 0.1, 1.0, 0.7, 0.05,
                              help="0.1 = weak view, 1.0 = very high conviction")

    submitted = st.form_submit_button("Add View", type="primary")
    if submitted:
        try:
            new_view = View(
                view_type=view_type.lower(),
                long_asset=long_asset,
                short_asset=short_asset if (view_type == "Relative" and short_asset != "None") else None,
                view_return=view_return / 100,
                confidence=confidence,
            )
            st.session_state["bl_views"].append(new_view)
        except ValueError as e:
            st.error(str(e))

# --- Show and manage views ---
if st.session_state["bl_views"]:
    view_rows = []
    for i, v in enumerate(st.session_state["bl_views"]):
        if v.view_type == "absolute":
            description = f"{v.long_asset} → {v.view_return:.1%}/yr"
        else:
            description = f"{v.long_asset} outperforms {v.short_asset} by {v.view_return:.1%}/yr"
        view_rows.append({
            "#":           i + 1,
            "Type":        v.view_type.capitalize(),
            "View":        description,
            "Confidence":  f"{v.confidence:.0%}",
        })

    st.dataframe(pd.DataFrame(view_rows), use_container_width=True, hide_index=True)

    col_clear, col_run = st.columns([1, 3])
    if col_clear.button("Clear All Views"):
        st.session_state["bl_views"] = []
        st.rerun()

    run_bl = col_run.button("Run Black-Litterman", type="primary")
else:
    st.info("No views added yet. The BL portfolio will equal the reference portfolio. Add views above.", icon="💡")
    run_bl = st.button("Run Equilibrium (no views)", type="secondary")

# ---------------------------------------------------------------------------
# Run BL and display results
# ---------------------------------------------------------------------------
if run_bl or "bl_weights" in st.session_state:
    if run_bl:
        # rebuild model with current views
        bl = BlackLitterman(returns=returns, reference_weights=ref_w,
                            risk_aversion=risk_aversion, tau=tau)
        for view in st.session_state.get("bl_views", []):
            bl.add_view(view)

        with st.spinner("Running Black-Litterman optimisation…"):
            bl_weights  = bl.optimal_weights()
            eq_returns  = bl.equilibrium_returns()
            post_returns = bl.posterior_returns()
            impact_tbl  = bl.view_impact_table()

        st.session_state["bl_weights"]      = bl_weights
        st.session_state["bl_eq_returns"]   = eq_returns
        st.session_state["bl_post_returns"] = post_returns
        st.session_state["bl_impact_tbl"]   = impact_tbl
        # also store as current portfolio for downstream pages
        bl_port = portfolio_returns(returns, bl_weights)
        st.session_state["bl_portfolio_returns"] = bl_port

    bl_weights   = st.session_state["bl_weights"]
    eq_returns   = st.session_state["bl_eq_returns"]
    post_returns = st.session_state["bl_post_returns"]
    impact_tbl   = st.session_state["bl_impact_tbl"]

    st.markdown("---")
    st.subheader("Black-Litterman Results")

    # Key metrics
    bl_port = portfolio_returns(returns, bl_weights)
    bl_stats = summary_statistics(bl_port, bench_rets, rf_rets, label="Black-Litterman")
    ref_port = portfolio_returns(returns, ref_w)

    render_metric_row([
        {"label": "Ann. Return",     "value": fmt_pct(bl_stats["Ann. Return"]),
         "help": "BL portfolio annualised return"},
        {"label": "Ann. Volatility", "value": fmt_pct(bl_stats["Ann. Volatility"])},
        {"label": "Sharpe Ratio",    "value": fmt_ratio(bl_stats["Sharpe Ratio"])},
        {"label": "Max Drawdown",    "value": fmt_pct(bl_stats["Max Drawdown"])},
        {"label": "# Views",         "value": str(len(st.session_state.get("bl_views", [])))},
        {"label": "Reference",       "value": ref_model},
    ])

    tab_weights, tab_returns, tab_perf, tab_use = st.tabs([
        "Weights Comparison",
        "View Impact on Returns",
        "Performance",
        "Save to Portfolio",
    ])

    # --- Tab 1: Weights comparison ---
    with tab_weights:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                allocation_pie_chart(ref_w, title=f"Reference ({ref_model})"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                allocation_pie_chart(bl_weights, title="Black-Litterman"),
                use_container_width=True,
            )

        # weight diff bar chart
        diff = (bl_weights - ref_w.reindex(bl_weights.index).fillna(0)).sort_values()
        colors = ["#e84855" if v < 0 else "#57cc99" for v in diff.values]
        fig_diff = go.Figure(go.Bar(
            x=diff.index.tolist(), y=diff.values * 100,
            marker_color=colors,
            hovertemplate="%{x}<br>Δweight: %{y:+.2f}%<extra></extra>",
        ))
        fig_diff.add_hline(y=0, line_color="grey", line_dash="dot")
        fig_diff.update_layout(
            title="Weight Changes: BL vs Reference",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Change in Weight (%)", yaxis_ticksuffix="%",
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    # --- Tab 2: View impact on returns ---
    with tab_returns:
        st.subheader("How Views Shifted Expected Returns")
        st.markdown("""
        The table below shows how each manager view shifted the expected return
        from the market equilibrium to the BL posterior.
        **Equilibrium** is the implied return from the reference portfolio.
        **Posterior** is after incorporating your views.
        """)

        display = impact_tbl.copy()
        display["Equilibrium Return"] = display["Equilibrium Return"].map(fmt_pct)
        display["BL Posterior Return"]= display["BL Posterior Return"].map(fmt_pct)
        display["Shift"]              = display["Shift"].map(lambda v: f"{v:+.2%}")
        st.dataframe(display, use_container_width=True)

        # Arrow chart: equilibrium → posterior
        fig_ret = go.Figure()
        assets_list = impact_tbl.index.tolist()

        # equilibrium / posterior returns already annualised (Sigma is annual)
        eq_vals   = bl.equilibrium_returns().values * 100
        post_vals = bl.posterior_returns().values * 100

        fig_ret.add_trace(go.Bar(
            name="Equilibrium",
            x=assets_list, y=eq_vals,
            marker_color=ACCENT_COLOR, opacity=0.5,
        ))
        fig_ret.add_trace(go.Bar(
            name="BL Posterior",
            x=assets_list, y=post_vals,
            marker_color="#2e86ab",
        ))
        fig_ret.update_layout(
            barmode="group",
            title="Equilibrium vs BL Posterior Expected Returns (annualised)",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Expected Return (%/yr)", yaxis_ticksuffix="%",
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    # --- Tab 3: Performance ---
    with tab_perf:
        ref_stats = summary_statistics(ref_port, bench_rets, rf_rets, label=ref_model)

        render_metric_row([
            {"label": f"BL Return",          "value": fmt_pct(bl_stats["Ann. Return"])},
            {"label": f"Reference Return",   "value": fmt_pct(ref_stats["Ann. Return"])},
            {"label": "BL Sharpe",           "value": fmt_ratio(bl_stats["Sharpe Ratio"])},
            {"label": "Reference Sharpe",    "value": fmt_ratio(ref_stats["Sharpe Ratio"])},
        ])

        bench_aligned = bench_rets.reindex(bl_port.index).dropna()
        fig = go.Figure()
        for label, port, color in [
            ("Black-Litterman", bl_port, ACCENT_COLOR),
            (ref_model, ref_port, "#2e86ab"),
            ("Benchmark", bench_aligned, "#999"),
        ]:
            wealth = (1 + port).cumprod()
            fig.add_trace(go.Scatter(
                x=wealth.index, y=wealth.values,
                name=label,
                line=dict(color=color, width=2,
                          dash="dash" if label == "Benchmark" else "solid"),
            ))
        fig.update_layout(
            title="Cumulative Return: BL vs Reference vs Benchmark",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Growth of £1",
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 4: Save to portfolio ---
    with tab_use:
        st.subheader("Use Black-Litterman as Your Active Portfolio")
        st.markdown("""
        Click below to set the Black-Litterman portfolio as your current portfolio
        for the Performance, Risk, AI Commentary, and IC Report pages.
        """)
        if st.button("Set BL Portfolio as Current Portfolio", type="primary"):
            st.session_state["current_weights"]   = bl_weights
            st.session_state["current_model"]     = "Black-Litterman"
            st.session_state["portfolio_returns"] = bl_port
            st.session_state["portfolio_stats"]   = bl_stats
            st.success(
                "Black-Litterman portfolio set as current. Navigate to Performance or Risk pages.",
                icon="✅"
            )
