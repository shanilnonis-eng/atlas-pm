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
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
from analytics.garch_volatility import (
    fit_garch, fit_gjr_garch,
    get_conditional_volatility, get_garch_var,
    get_volatility_forecast, get_garch_params,
    garch_persistence, has_leverage_effect,
    MIN_OBS as _GARCH_MIN_OBS,
)

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
    confidence_pct = st.slider("VaR Confidence Level", 90, 99, int(VAR_CONFIDENCE * 100), 1,
                               format="%d%%")
    confidence = confidence_pct / 100
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

tab_var, tab_contrib, tab_corr, tab_stress, tab_backtest, tab_garch = st.tabs([
    "VaR / CVaR",
    "Risk Contribution",
    "Correlations",
    "Stress Tests",
    "VaR Back-test",
    "GARCH Volatility",
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

# ---------------------------------------------------------------------------
# Tab 6: GARCH Volatility
# ---------------------------------------------------------------------------
with tab_garch:
    st.subheader("GARCH Volatility Modelling")
    st.markdown("""
Two conditional heteroskedasticity models are fitted to the portfolio return series:

| Model | Formula | What it adds |
|-------|---------|--------------|
| **GARCH(1,1)** | σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} | Volatility clustering |
| **GJR-GARCH(1,1,1)** | + γ·ε²_{t-1}·**I**(ε_{t-1} < 0) | Leverage effect — negative shocks raise vol more than positive |

Both are estimated with **Student-t innovations** for better tail fit.
Fitted with maximum likelihood. Confidence interval uses the sidebar setting.
    """)

    # Minimum data guard
    if len(port_rets) < _GARCH_MIN_OBS:
        st.warning(
            f"GARCH requires at least {_GARCH_MIN_OBS} observations; "
            f"this portfolio has {len(port_rets)}. Load more history.",
            icon="⚠️",
        )
    else:
        if st.button("Fit GARCH Models", key="garch_fit_btn"):
            with st.spinner(
                "Fitting GARCH(1,1) and GJR-GARCH(1,1,1) via MLE — takes a few seconds…"
            ):
                try:
                    _g_res  = fit_garch(port_rets)
                    _gjr_res = fit_gjr_garch(port_rets)

                    st.session_state["_garch_cache"] = {
                        "garch_result":      _g_res,
                        "gjr_result":        _gjr_res,
                        "garch_cond_vol":    get_conditional_volatility(_g_res,   annualise=True),
                        "gjr_cond_vol":      get_conditional_volatility(_gjr_res, annualise=True),
                        "garch_persist":     garch_persistence(_g_res),
                        "gjr_persist":       garch_persistence(_gjr_res),
                        "gjr_leverage":      has_leverage_effect(_gjr_res),
                        "garch_params_df":   get_garch_params(_g_res),
                        "gjr_params_df":     get_garch_params(_gjr_res),
                        "garch_forecast":    get_volatility_forecast(_g_res,   horizon=21, annualise=True),
                        "gjr_forecast":      get_volatility_forecast(_gjr_res, horizon=21, annualise=True),
                    }
                    st.success("Models fitted successfully.")
                except Exception as _e:
                    st.error(f"GARCH fitting failed: {_e}")

        # ── Render from cache ──────────────────────────────────────────────
        if "_garch_cache" in st.session_state:
            import plotly.graph_objects as _go

            _gc = st.session_state["_garch_cache"]
            _g_cv   = _gc["garch_cond_vol"]
            _gjr_cv = _gc["gjr_cond_vol"]

            # -- Key model metrics ------------------------------------------
            st.markdown("---")
            _m1, _m2, _m3, _m4 = st.columns(4)

            _g_last_ann  = float(_g_cv.iloc[-1])
            _gjr_last_ann = float(_gjr_cv.iloc[-1])
            _gjr_sig, _gjr_pval = _gc["gjr_leverage"]

            _m1.metric(
                "GARCH Current Vol (Ann.)",
                f"{_g_last_ann:.1%}",
                help="Most recent GARCH(1,1) conditional volatility, annualised.",
            )
            _m2.metric(
                "GJR Current Vol (Ann.)",
                f"{_gjr_last_ann:.1%}",
                help="Most recent GJR-GARCH conditional volatility, annualised.",
            )
            _m3.metric(
                "GARCH Persistence (α+β)",
                f"{_gc['garch_persist']:.4f}",
                help="Closer to 1 = slower vol mean-reversion (longer memory).",
            )
            _m4.metric(
                "GJR Persistence (α+β+γ/2)",
                f"{_gc['gjr_persist']:.4f}",
                help="Persistence in GJR-GARCH; includes asymmetry contribution.",
            )

            # Leverage effect callout
            if not np.isnan(_gjr_pval):
                _lev_msg = (
                    f"✅ **Leverage effect present** — γ p-value = {_gjr_pval:.3f} "
                    f"(significant at 10%). Negative shocks raise volatility "
                    f"disproportionately in this portfolio."
                    if _gjr_sig else
                    f"ℹ️ **No significant leverage effect** — γ p-value = {_gjr_pval:.3f}. "
                    f"Positive and negative shocks have similar volatility impact."
                )
                st.info(_lev_msg)

            # -- Conditional volatility chart --------------------------------
            st.markdown("---")
            st.subheader("Conditional Volatility vs Rolling Historical Vol")

            _roll_vol = port_rets.rolling(63).std(ddof=1) * np.sqrt(252)

            _fig_vol = _go.Figure()
            _fig_vol.add_trace(_go.Scatter(
                x=_roll_vol.index, y=_roll_vol.values * 100,
                mode="lines",
                line=dict(color="#95a5a6", width=1.5, dash="dot"),
                name="Rolling Hist. Vol (63d)",
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
            ))
            _fig_vol.add_trace(_go.Scatter(
                x=_g_cv.index, y=_g_cv.values * 100,
                mode="lines",
                line=dict(color="#2e86ab", width=2),
                name="GARCH(1,1) Cond. Vol",
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
            ))
            _fig_vol.add_trace(_go.Scatter(
                x=_gjr_cv.index, y=_gjr_cv.values * 100,
                mode="lines",
                line=dict(color="#9b5de5", width=2),
                name="GJR-GARCH Cond. Vol",
                hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
            ))
            _fig_vol.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="Annualised Volatility (%)", yaxis_ticksuffix="%",
                font=dict(family="Inter, Arial, sans-serif"),
                legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1),
            )
            st.plotly_chart(_fig_vol, use_container_width=True)
            st.caption(
                "Rolling historical vol assigns equal weight to all observations in the window. "
                "GARCH conditional vol responds faster to recent market shocks."
            )

            # -- VaR comparison ---------------------------------------------
            st.markdown("---")
            st.subheader(f"VaR Comparison — {int(confidence * 100)}% Confidence")

            _hist_var_1d  = historical_var(port_rets, confidence, horizon_days=1)
            _g_var_1d     = get_garch_var(_gc["garch_result"],  confidence)
            _gjr_var_1d   = get_garch_var(_gc["gjr_result"],    confidence)

            _var_rows = [
                {"Method": "Historical (Empirical)",    "1-Day VaR (%)": _hist_var_1d,  "1-Day VaR (£)": _hist_var_1d  * portfolio_value},
                {"Method": "GARCH(1,1)",                 "1-Day VaR (%)": _g_var_1d,     "1-Day VaR (£)": _g_var_1d     * portfolio_value},
                {"Method": "GJR-GARCH(1,1,1)",           "1-Day VaR (%)": _gjr_var_1d,   "1-Day VaR (£)": _gjr_var_1d   * portfolio_value},
            ]
            _var_df = pd.DataFrame(_var_rows)
            _var_display = _var_df.copy()
            _var_display["1-Day VaR (%)"] = _var_display["1-Day VaR (%)"].map(fmt_pct)
            _var_display["1-Day VaR (£)"] = _var_display["1-Day VaR (£)"].map(fmt_currency)
            st.dataframe(_var_display, use_container_width=True, hide_index=True)

            st.caption(
                "GARCH-VaR uses the most recent conditional volatility and the "
                "Student-t innovation quantile. It is forward-looking conditional on the "
                "current volatility regime, unlike Historical VaR which weights all "
                "observations equally."
            )

            # -- Volatility forecast ----------------------------------------
            st.markdown("---")
            st.subheader("Conditional Volatility Forecast (21-day horizon)")

            _g_fcast   = _gc["garch_forecast"]
            _gjr_fcast = _gc["gjr_forecast"]

            _fig_fcast = _go.Figure()
            _fig_fcast.add_trace(_go.Scatter(
                x=list(_g_fcast.index), y=_g_fcast.values * 100,
                mode="lines+markers",
                line=dict(color="#2e86ab", width=2),
                marker=dict(size=5),
                name="GARCH(1,1)",
                hovertemplate="Day %{x}<br>Vol: %{y:.2f}%<extra></extra>",
            ))
            _fig_fcast.add_trace(_go.Scatter(
                x=list(_gjr_fcast.index), y=_gjr_fcast.values * 100,
                mode="lines+markers",
                line=dict(color="#9b5de5", width=2),
                marker=dict(size=5),
                name="GJR-GARCH(1,1,1)",
                hovertemplate="Day %{x}<br>Vol: %{y:.2f}%<extra></extra>",
            ))
            _fig_fcast.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Days Ahead", yaxis_title="Forecast Volatility (Ann. %)",
                yaxis_ticksuffix="%",
                font=dict(family="Inter, Arial, sans-serif"),
            )
            st.plotly_chart(_fig_fcast, use_container_width=True)
            st.caption(
                "Multi-step forecasts revert toward the unconditional (long-run) volatility "
                "at a rate determined by persistence (α+β). "
                "Low persistence → faster mean-reversion. "
                "Persistence near 1 → nearly flat forecast (shocks are long-lasting)."
            )

            # -- Parameter tables -------------------------------------------
            with st.expander("Model Parameters", expanded=False):
                _pc1, _pc2 = st.columns(2)
                with _pc1:
                    st.markdown("**GARCH(1,1) — Student-t MLE**")
                    _g_pdf = _gc["garch_params_df"].copy()
                    _g_pdf["Estimate"]   = _g_pdf["Estimate"].map(lambda v: f"{v:.6f}")
                    _g_pdf["Std Error"]  = _g_pdf["Std Error"].map(lambda v: f"{v:.6f}")
                    _g_pdf["t-stat"]     = _g_pdf["t-stat"].map(lambda v: f"{v:.3f}")
                    _g_pdf["p-value"]    = _g_pdf["p-value"].map(lambda v: f"{v:.4f}")
                    st.dataframe(_g_pdf.drop(columns=["Significant (10%)"]),
                                 use_container_width=True, hide_index=True)

                with _pc2:
                    st.markdown("**GJR-GARCH(1,1,1) — Student-t MLE**")
                    _gjr_pdf = _gc["gjr_params_df"].copy()
                    _gjr_pdf["Estimate"]  = _gjr_pdf["Estimate"].map(lambda v: f"{v:.6f}")
                    _gjr_pdf["Std Error"] = _gjr_pdf["Std Error"].map(lambda v: f"{v:.6f}")
                    _gjr_pdf["t-stat"]    = _gjr_pdf["t-stat"].map(lambda v: f"{v:.3f}")
                    _gjr_pdf["p-value"]   = _gjr_pdf["p-value"].map(lambda v: f"{v:.4f}")
                    st.dataframe(_gjr_pdf.drop(columns=["Significant (10%)"]),
                                 use_container_width=True, hide_index=True)

                st.markdown("""
**Parameter guide:**

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| mu        | μ      | Conditional mean of daily returns |
| omega     | ω      | Long-run variance floor (must be > 0) |
| alpha[1]  | α      | Weight on yesterday's squared shock |
| gamma[1]  | γ      | Extra weight for negative shocks (leverage effect) |
| beta[1]   | β      | Weight on yesterday's conditional variance |
| nu        | ν      | Degrees of freedom of Student-t innovations (lower = fatter tails) |

**Persistence interpretation:** α + β (GARCH) or α + β + γ/2 (GJR).
Values near 1 indicate slow decay of volatility shocks (long memory).
                """)

            # -- Explanation ------------------------------------------------
            with st.expander("How to interpret GARCH results", expanded=False):
                st.markdown(f"""
**Model purpose — conditional volatility, not constant volatility.**

Your existing Historical VaR and rolling vol assume either static or equally-weighted
volatility. GARCH models time-varying conditional volatility: the variance of tomorrow's
return is a function of recent shocks and recent variance.

---

**Volatility clustering** (both models):
> *"Large changes tend to be followed by large changes, of either sign."* — Mandelbrot (1963)

The GARCH α parameter captures how much a shock today impacts tomorrow's variance.
The β parameter captures how slowly that impact fades. α + β close to 1 means it
takes many days for a large shock to fully dissipate.

---

**The leverage effect** (GJR-GARCH only):
Markets tend to become more volatile after a price *decline* than after an equivalent-sized
rally. This is captured by the γ parameter. If γ is positive and statistically significant,
the portfolio shows the leverage effect.

Current GJR-GARCH γ p-value: **{_gjr_pval:.3f}**
{'→ Leverage effect is statistically significant at 10%.' if _gjr_sig else '→ No statistically significant leverage effect at 10%.'}

---

**GARCH-VaR vs Historical VaR:**
Historical VaR gives each day equal weight. GARCH-VaR conditions on the *current*
volatility regime. In calm periods GARCH-VaR is typically lower (tighter risk estimate);
entering a stress period it widens faster, giving earlier warning.

At **{int(confidence*100)}% confidence:**
- Historical VaR: **{fmt_pct(_hist_var_1d)}**
- GARCH(1,1) VaR: **{fmt_pct(_g_var_1d)}**
- GJR-GARCH VaR: **{fmt_pct(_gjr_var_1d)}**

---

*These results are illustrative. GARCH models assume return stationarity and do not
capture structural breaks, liquidity risk, or correlation breakdown in stress.*
                """)

            st.info(
                "**Model limitations**: GARCH assumes covariance-stationarity (persistence < 1). "
                "In extreme market regimes, the model may underestimate risk. "
                "Student-t innovations improve tail fit but the chosen degrees of freedom ν "
                "are estimated from historical data and may not reflect future tail behaviour. "
                "GARCH-VaR is conditional (point-in-time), not unconditional.",
                icon="⚠️",
            )
