"""
Page 2 — Portfolio Construction

Builds and compares portfolios using four optimisation models.
Stores the chosen weights in session state for downstream analytics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import MODEL_NAMES, MIN_WEIGHT, MAX_WEIGHT
from construction.optimiser import (
    run_optimisation, efficient_frontier, compute_cov_matrix,
    maximum_sharpe,
)
from config.settings import TRADING_DAYS_PER_YEAR as _TDAYS


def _port_risk_return(weights, returns):
    """
    Compute (ann_vol, ann_return) in the same arithmetic space as the frontier.
    Mirrors construction.optimiser.portfolio_risk_return exactly.
    """
    cov = compute_cov_matrix(returns, shrink=True)
    mu_ann = returns.mean().values * _TDAYS
    w = weights.reindex(returns.columns).fillna(0.0).values
    if w.sum() > 0:
        w = w / w.sum()
    ann_vol = float(np.sqrt(max(float(w @ cov.values @ w), 0.0)))
    ann_return = float(mu_ann @ w)
    return ann_vol, ann_return
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
from analytics.investor_utility import (
    PROFILE_NAMES        as _INV_PROFILES,
    map_profile_to_risk_aversion  as _map_ra,
    calculate_indifference_curve  as _calc_curve,
    find_utility_optimal_portfolio as _find_opt,
    calculate_mean_variance_utility as _calc_utility,
)

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

# DEBUG — remove after diagnosis
with st.expander("DEBUG: session state info", expanded=True):
    st.write(f"returns shape: {returns.shape}")
    st.write(f"returns NaN count: {int(returns.isna().sum().sum())}")
    st.write(f"returns inf count: {int((returns == float('inf')).sum().sum() + (returns == float('-inf')).sum().sum())}")
    st.write(f"bench_returns len: {len(bench_returns)}")
    st.write(f"rf_returns len: {len(rf_returns)}")
    st.write(f"returns index[0]: {returns.index[0] if len(returns) > 0 else 'EMPTY'}")
    st.write(f"returns columns: {list(returns.columns)}")

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
    min_w_pct = st.slider("Minimum weight per asset", 0, 20, int(MIN_WEIGHT * 100), 1,
                          format="%d%%",
                          help="Minimum allocation to any single asset. 0% allows zero.")
    max_w_pct = st.slider("Maximum weight per asset", 10, 100, int(MAX_WEIGHT * 100), 5,
                          format="%d%%",
                          help="Maximum concentration in any single asset.")
    min_w = min_w_pct / 100
    max_w = max_w_pct / 100

    # warn if constraints are infeasible
    n_assets = len(st.session_state.get("selected_assets", []))
    if n_assets > 0 and min_w * n_assets > 1.0:
        st.warning(
            f"⚠️ {min_w_pct}% minimum × {n_assets} assets = "
            f"{min_w_pct * n_assets}% > 100%. Reduce minimum weight.",
            icon="⚠️",
        )

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
Portfolios below the frontier are suboptimal — you can achieve the same return with less risk.
The **Maximum Sharpe** portfolio (★) is the tangency point where the Capital Market Line
(dashed amber) touches the frontier — it maximises excess return per unit of risk.
    """)

    # ── Risk preference selector (always visible once expander is open) ──
    st.markdown("---")
    st.markdown(
        "**Risk Preference Overlay** "
        "*(illustrative — model-implied preference, not financial advice)*"
    )
    _ef_col1, _ef_col2 = st.columns([1, 1])
    with _ef_col1:
        _ef_profile = st.selectbox(
            "Investor risk profile",
            _INV_PROFILES,
            index=2,       # default: Balanced (A = 4)
            key="ef_inv_profile",
            help="Sets the risk-aversion coefficient A used in U = E(r) − 0.5·A·σ².",
        )
    with _ef_col2:
        if _ef_profile == "Custom":
            _ef_ra = st.slider(
                "Risk aversion coefficient (A)",
                min_value=0.5, max_value=10.0, value=4.0, step=0.5,
                key="ef_custom_ra",
                help="A = 1 (aggressive) → A = 10 (very conservative).",
            )
        else:
            _ef_ra = _map_ra(_ef_profile)
            st.metric(
                "Risk aversion coefficient (A)", f"{_ef_ra:.1f}",
                help="U = E(r) − 0.5 × A × σ²",
            )

    if st.button("Compute Efficient Frontier (takes ~10s)"):
        with st.spinner("Computing efficient frontier and max sharpe portfolio…"):
            try:
                frontier = efficient_frontier(
                    returns, n_points=40, min_weight=min_w, max_weight=max_w,
                )
                if frontier.empty or "Ann. Volatility" not in frontier.columns:
                    st.warning(
                        "Efficient frontier could not be computed with these constraints. "
                        "Try reducing the minimum weight or increasing the maximum weight.",
                        icon="⚠️",
                    )
                else:
                    w_ms = maximum_sharpe(
                        returns, rf_annual=rf_rate,
                        min_weight=min_w, max_weight=max_w,
                    )
                    vol_ms, ret_ms = _port_risk_return(w_ms, returns)

                    # Validation: nearest frontier point should be within ~0.5 %
                    dists = np.sqrt(
                        (frontier["Ann. Volatility"].values - vol_ms) ** 2
                        + (frontier["Ann. Return"].values - ret_ms) ** 2
                    )
                    if float(dists.min()) > 0.005:
                        st.warning(
                            f"Max Sharpe portfolio is {float(dists.min()):.4f} (Euclidean) from "
                            "the nearest frontier point — inputs may be inconsistent.",
                            icon="⚠️",
                        )

                    # Equal-weight portfolio point
                    w_ew = pd.Series(
                        1.0 / len(returns.columns),
                        index=returns.columns,
                        name="Equal Weight",
                    )
                    vol_ew, ret_ew = _port_risk_return(w_ew, returns)

                    # Base overlay: Max Sharpe + Equal Weight + current model (if different)
                    base_overlay: dict[str, tuple[float, float]] = {
                        "Max Sharpe":  (vol_ms, ret_ms),
                        "Equal Weight": (vol_ew, ret_ew),
                    }
                    cur_lbl = st.session_state.get("current_model", model)
                    if (
                        "current_weights" in st.session_state
                        and cur_lbl not in ("Maximum Sharpe", "Equal Weight")
                    ):
                        w_cur = st.session_state["current_weights"]
                        vol_cur, ret_cur = _port_risk_return(w_cur, returns)
                        base_overlay[cur_lbl] = (vol_cur, ret_cur)

                    # Cache frontier data — re-rendered on every risk aversion change
                    st.session_state["_ef_cache"] = {
                        "frontier":     frontier,
                        "vol_ms":       vol_ms,
                        "ret_ms":       ret_ms,
                        "vol_ew":       vol_ew,
                        "ret_ew":       ret_ew,
                        "base_overlay": base_overlay,
                        "rf_rate":      rf_rate,
                    }
            except Exception as e:
                st.error(f"Frontier computation failed: {e}")

    # ── Render from cache (re-runs whenever risk aversion changes) ───────
    if "_ef_cache" in st.session_state:
        import plotly.graph_objects as _go

        _efc        = st.session_state["_ef_cache"]
        _ef_frontier = _efc["frontier"]
        _ef_vol_ms   = _efc["vol_ms"]
        _ef_ret_ms   = _efc["ret_ms"]
        _ef_vol_ew   = _efc["vol_ew"]
        _ef_ret_ew   = _efc["ret_ew"]
        _ef_base_ovl = dict(_efc["base_overlay"])
        _ef_rf       = _efc["rf_rate"]

        # Utility-optimal portfolio for current risk aversion
        _ef_opt = _find_opt(
            _ef_frontier["Ann. Return"].values,
            _ef_frontier["Ann. Volatility"].values,
            _ef_ra,
        )

        # Build figure with base overlay (Max Sharpe + Equal Weight + current model)
        fig = efficient_frontier_chart(_ef_frontier, _ef_base_ovl)

        # Utility-optimal marker — diamond to distinguish from star markers
        fig.add_trace(_go.Scatter(
            x=[_ef_opt["volatility"] * 100],
            y=[_ef_opt["expected_return"] * 100],
            mode="markers+text",
            marker=dict(size=14, color="#9b5de5", symbol="diamond"),
            text=["Utility-Optimal"],
            textposition="top center",
            name="Utility-Optimal Portfolio",
            hovertemplate=(
                "<b>Utility-Optimal Portfolio</b><br>"
                "Vol: %{x:.2f}%<br>"
                "Return: %{y:.2f}%<br>"
                f"A = {_ef_ra:.1f}<extra></extra>"
            ),
        ))

        # Indifference curves — optimal + 2 lower curves for visual context
        _vol_lo = max(0.001, _ef_frontier["Ann. Volatility"].min() - 0.02)
        _vol_hi = _ef_frontier["Ann. Volatility"].max() + 0.02
        _ef_vol_rng = np.linspace(_vol_lo, _vol_hi, 300)
        _opt_u = _ef_opt["utility"]
        _curve_colors = ["#9b5de5", "#b57bee", "#d4a9f7"]
        _ret_lo_clip = _ef_frontier["Ann. Return"].min() - 0.05
        _ret_hi_clip = _ef_frontier["Ann. Return"].max() + 0.05

        for _j, _u_delta in enumerate([0.0, -0.01, -0.025]):
            _u_lvl  = _opt_u + _u_delta
            _c_rets = _calc_curve(_ef_vol_rng, _u_lvl, _ef_ra)
            _mask   = (_c_rets >= _ret_lo_clip) & (_c_rets <= _ret_hi_clip)
            if _mask.sum() < 2:
                continue
            _c_name = (
                "Utility Curve (U*)" if _j == 0
                else f"Lower Utility Curve {_j}"
            )
            fig.add_trace(_go.Scatter(
                x=_ef_vol_rng[_mask] * 100,
                y=_c_rets[_mask] * 100,
                mode="lines",
                line=dict(
                    color=_curve_colors[_j],
                    width=1.8 if _j == 0 else 1.0,
                    dash="solid" if _j == 0 else "dot",
                ),
                name=_c_name,
                hovertemplate=(
                    f"U = {_u_lvl:.4f}<br>"
                    "Vol: %{x:.2f}%<br>"
                    "Return: %{y:.2f}%<extra></extra>"
                ),
            ))

        # Capital Market Line
        if _ef_vol_ms > 1e-8:
            _cml_slope = (_ef_ret_ms - _ef_rf) / _ef_vol_ms
            _cml_vols  = np.array([0.0, _ef_vol_ms * 1.5])
            _cml_rets  = _ef_rf + _cml_slope * _cml_vols
            fig.add_trace(_go.Scatter(
                x=_cml_vols * 100,
                y=_cml_rets * 100,
                mode="lines",
                line=dict(color="#f4a261", width=1.8, dash="dash"),
                name="Capital Market Line",
                hovertemplate=(
                    "CML<br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
                ),
            ))

        # Move utility curves + CML to back so frontier and points render on top
        _bg_names = {
            "Capital Market Line",
            "Utility Curve (U*)",
            "Lower Utility Curve 1",
            "Lower Utility Curve 2",
        }
        _bg = tuple(t for t in fig.data if t.name in _bg_names)
        _fg = tuple(t for t in fig.data if t.name not in _bg_names)
        fig.data = _bg + _fg

        st.plotly_chart(fig, use_container_width=True)

        # ── Explanation ───────────────────────────────────────────────────
        with st.expander("How to read the risk preference overlay", expanded=False):
            st.markdown(f"""
**Illustrative investor utility — model-implied preference only.**
*This overlay is not financial advice, regulated advice, or a suitable investment recommendation.*

---

The **utility curve** (purple) shows risk–return combinations that yield the same
mean-variance utility:

> *U = E(r) − 0.5 × A × σ²*

where *E(r)* = annualised expected return, *σ* = annualised volatility, and *A* = risk-aversion
coefficient. Higher indifference curves represent higher utility.

The **utility-optimal portfolio** (◆) is the frontier point that *maximises* utility for
the selected risk-aversion level — found by evaluating the formula at every frontier point
and selecting the highest.

**How risk aversion (A) affects the result:**
- **High A (e.g. A = 8, Very Conservative)**: the curve is steep — the investor demands much more
  return for any extra volatility. The utility-optimal portfolio sits toward the low-risk end
  of the frontier.
- **Low A (e.g. A = 1, Aggressive)**: the curve is shallow — the investor accepts more volatility
  for a smaller return premium. The utility-optimal portfolio shifts toward higher return/risk.
- **A = 4 (Balanced)** is a standard textbook reference value.

**Utility-optimal vs Maximum Sharpe:**
Maximum Sharpe maximises *excess return per unit of risk* — a ratio, independent of risk
tolerance. The utility-optimal portfolio reflects a *specific* risk tolerance level, so the two
will generally differ unless risk aversion happens to align with the tangency condition.

*Selected profile: **{_ef_profile}** — risk aversion A = {_ef_ra:.1f}*
            """)

        # ── Results panel ──────────────────────────────────────────────────
        st.markdown(
            "**Model-Implied Utility Analysis** "
            "*(illustrative — based on historical data and selected assumptions)*"
        )

        _ms_utility = float(_calc_utility(_ef_ret_ms, _ef_vol_ms, _ef_ra))
        _ew_utility = float(_calc_utility(_ef_ret_ew, _ef_vol_ew, _ef_ra))

        _p1, _p2, _p3 = st.columns(3)
        _p1.metric("Investor Profile",        _ef_profile)
        _p2.metric("Risk Aversion (A)",       f"{_ef_ra:.1f}")
        _p3.metric("Utility Score (U*)",      f"{_ef_opt['utility']:.4f}")

        _p4, _p5, _p6 = st.columns(3)
        _p4.metric("Utility-Optimal Return",     f"{_ef_opt['expected_return']:.1%}")
        _p5.metric("Utility-Optimal Volatility", f"{_ef_opt['volatility']:.1%}")
        _p6.metric(
            "Max Sharpe Return (reference)",
            f"{_ef_ret_ms:.1%}",
            delta=f"{_ef_opt['expected_return'] - _ef_ret_ms:+.1%}",
            delta_color="off",
            help="Delta shows utility-optimal return minus Max Sharpe return.",
        )

        _p7, _p8, _p9 = st.columns(3)
        _p7.metric("Max Sharpe Utility",   f"{_ms_utility:.4f}")
        _p8.metric("Equal Weight Utility", f"{_ew_utility:.4f}")
        _p9.metric(
            "Utility Gain vs Equal Weight",
            f"{_ef_opt['utility']:.4f}",
            delta=f"{_ef_opt['utility'] - _ew_utility:+.4f}",
            delta_color="normal",
            help="Positive delta means utility-optimal portfolio has higher utility than equal weight.",
        )
