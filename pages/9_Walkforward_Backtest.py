"""
Page 9 — Walk-Forward Backtesting

Out-of-sample validation of every portfolio construction model.
No look-ahead bias: weights at time t use only data up to t.

Key question answered: do these optimisers actually add value,
or do they only look good because they were fitted on the same
data they are evaluated on?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.backtest import run_walk_forward, build_summary_table, build_degradation_table
from analytics.returns import drawdown_series, annualised_return, annualised_volatility, sharpe_ratio
from ui.components.metrics import fmt_pct, fmt_ratio
from config.settings import MODEL_NAMES, TRADING_DAYS_PER_YEAR

PALETTE      = ["#1a3a5c", "#2e86ab", "#e84855", "#f4a261", "#57cc99", "#9b5de5"]
PALETTE_RGBA = ["rgba(26,58,92,0.12)", "rgba(46,134,171,0.12)", "rgba(232,72,85,0.12)",
                "rgba(244,162,97,0.12)", "rgba(87,204,153,0.12)", "rgba(155,93,229,0.12)"]
LAYOUT  = dict(
    font=dict(family="Inter, Arial, sans-serif", size=12),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1),
)

st.title("Walk-Forward Backtesting")
st.markdown(
    "Out-of-sample validation of all construction models. "
    "Weights at each rebalancing date are computed using **only past data** — "
    "no look-ahead bias."
)
st.markdown("---")

# ─── Guard ────────────────────────────────────────────────────────────────────
if "simple_returns" not in st.session_state or "prices" not in st.session_state:
    st.warning("Please load data first on the **Universe & Data** page.", icon="⚠️")
    st.stop()

prices       = st.session_state["prices"]
bench_rets   = st.session_state.get("bench_returns")
rf_rets      = st.session_state.get("rf_returns")
bench_label  = st.session_state.get("benchmark_label", "Benchmark")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Backtest Configuration")

    train_years = st.select_slider(
        "Training window",
        options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: f"{x} year{'s' if x > 1 else ''}",
        help="History used to compute weights at each rebalancing date.",
    )
    train_window = train_years * TRADING_DAYS_PER_YEAR

    rebal_freq = st.selectbox(
        "Rebalancing frequency",
        ["Monthly (~21 days)", "Quarterly (~63 days)", "Semi-Annual (~126 days)"],
        index=1,
    )
    test_window = {"Monthly (~21 days)": 21,
                   "Quarterly (~63 days)": 63,
                   "Semi-Annual (~126 days)": 126}[rebal_freq]

    rf_rate_pct = st.number_input(
        "Annual risk-free rate (%)", value=4.0, min_value=0.0, max_value=15.0, step=0.25
    )
    rf_annual = rf_rate_pct / 100

    st.markdown("---")
    st.markdown("**Models to test**")
    selected_models = []
    for m in MODEL_NAMES:
        if st.checkbox(m, value=True, key=f"bt_model_{m}"):
            selected_models.append(m)

    st.markdown("---")
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)
    st.caption(
        f"Data available: {len(prices)} days  \n"
        f"Training: {train_window} days  \n"
        f"Est. OOS periods: ~{max(0, (len(prices) - train_window) // test_window)}"
    )

if not selected_models:
    st.info("Select at least one model in the sidebar.")
    st.stop()

# ─── Cache key ────────────────────────────────────────────────────────────────
# Cache results in session state so re-runs with same params are instant.
cache_key = (
    tuple(sorted(prices.columns)),
    train_window,
    test_window,
    rf_annual,
    tuple(sorted(selected_models)),
)
cached = st.session_state.get("backtest_cache_key")
results = st.session_state.get("backtest_results")

needs_run = run_btn or (results is None) or (cached != cache_key)

if needs_run:
    if not run_btn and results is not None:
        st.info("Parameters changed — click **Run Backtest** to update.", icon="ℹ️")
        st.stop()

    n_periods_est = max(0, (len(prices) - train_window) // test_window)
    if n_periods_est < 4:
        st.error(
            f"Only ~{n_periods_est} out-of-sample periods available. "
            "Reduce the training window or select a longer date range.",
            icon="❌",
        )
        st.stop()

    with st.spinner(f"Running walk-forward backtest across {n_periods_est} periods per model…"):
        try:
            results = run_walk_forward(
                prices=prices,
                models=selected_models,
                train_window=train_window,
                test_window=test_window,
                rf_annual=rf_annual,
                min_weight=0.0,
                max_weight=0.40,
                shrink=True,
            )
            st.session_state["backtest_results"]   = results
            st.session_state["backtest_cache_key"] = cache_key
        except ValueError as e:
            st.error(str(e), icon="❌")
            st.stop()

# ─── Summary metrics ─────────────────────────────────────────────────────────
st.subheader("Out-of-Sample Performance Summary")

first_result = next(iter(results.values()))
oos_start = first_result.oos_returns.index[0].strftime("%d %b %Y")
oos_end   = first_result.oos_returns.index[-1].strftime("%d %b %Y")
n_periods = first_result.n_periods

st.caption(
    f"OOS period: **{oos_start}** → **{oos_end}**  |  "
    f"Rebalancing events: **{n_periods}** per model  |  "
    f"Training window: **{train_years} year{'s' if train_years > 1 else ''}** "
    f"({train_window} trading days)"
)

summary_df = build_summary_table(results, bench_rets, rf_annual)

if not summary_df.empty:
    display = summary_df.copy()
    for col in ["OOS Ann. Return", "OOS Ann. Vol", "OOS Max DD", "Avg Turnover"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_pct)
    for col in ["OOS Sharpe", "OOS Sortino"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_ratio)
    if "N Periods" in display.columns:
        display["N Periods"] = display["N Periods"].fillna("N/A").astype(str)

    st.dataframe(display, use_container_width=True)

st.markdown("---")

# ─── Cumulative wealth ────────────────────────────────────────────────────────
st.subheader("Cumulative Wealth (Out-of-Sample)")
st.caption(
    "All series start at £1.00 at the first OOS date. "
    "The training period is not shown — results here are purely out-of-sample."
)

fig_wealth = go.Figure()

for i, (model, result) in enumerate(results.items()):
    oos_r  = result.oos_returns
    wealth = (1 + oos_r).cumprod()
    fig_wealth.add_trace(go.Scatter(
        x=wealth.index, y=wealth.values,
        name=model,
        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
    ))

if bench_rets is not None:
    oos_idx   = first_result.oos_returns.index
    bench_oos = bench_rets.reindex(oos_idx).dropna()
    if not bench_oos.empty:
        bench_wealth = (1 + bench_oos).cumprod()
        fig_wealth.add_trace(go.Scatter(
            x=bench_wealth.index, y=bench_wealth.values,
            name=bench_label,
            line=dict(color="#aaa", width=1.5, dash="dash"),
        ))

fig_wealth.update_layout(
    title="Growth of £1 — Out-of-Sample",
    yaxis_title="Portfolio Value (£)",
    **LAYOUT,
)
st.plotly_chart(fig_wealth, use_container_width=True)

# ─── Drawdown ────────────────────────────────────────────────────────────────
st.subheader("Out-of-Sample Drawdown")

fig_dd = go.Figure()
for i, (model, result) in enumerate(results.items()):
    dd = drawdown_series(result.oos_returns) * 100
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name=model,
        fill="tozeroy",
        line=dict(color=PALETTE[i % len(PALETTE)], width=1),
        fillcolor=PALETTE_RGBA[i % len(PALETTE_RGBA)],
    ))

fig_dd.update_layout(
    title="Drawdown (%)",
    yaxis_title="Drawdown (%)",
    yaxis=dict(ticksuffix="%"),
    **LAYOUT,
)
st.plotly_chart(fig_dd, use_container_width=True)

# ─── Rolling Sharpe ───────────────────────────────────────────────────────────
st.subheader("Rolling 1-Year Out-of-Sample Sharpe")
st.caption("Trailing 252-day Sharpe ratio (arithmetic). Periods shorter than 252 days are excluded.")

fig_rs = go.Figure()
rf_daily = rf_annual / TRADING_DAYS_PER_YEAR

for i, (model, result) in enumerate(results.items()):
    r = result.oos_returns
    roll_excess = (r - rf_daily).rolling(252)
    roll_sharpe = (roll_excess.mean() * 252) / (r.rolling(252).std(ddof=1) * np.sqrt(252))
    roll_sharpe = roll_sharpe.dropna()
    if roll_sharpe.empty:
        continue
    fig_rs.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values,
        name=model,
        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
    ))

fig_rs.add_hline(y=0, line_dash="dot", line_color="#ccc")
fig_rs.update_layout(
    title="Rolling 12-Month Sharpe Ratio",
    yaxis_title="Sharpe Ratio",
    **LAYOUT,
)
st.plotly_chart(fig_rs, use_container_width=True)

# ─── IS vs OOS degradation ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("In-Sample vs Out-of-Sample Degradation")
st.markdown(
    """
The table below compares each model's Sharpe ratio **on its own training data** (in-sample)
versus what it actually delivered in the unseen test period (out-of-sample).

A large **Degradation** column means the model benefited substantially from fitting on
the same data it is evaluated on — i.e. estimation error is high. Equal Weight should
show the lowest degradation because it uses zero estimated parameters.

> DeMiguel et al. (2009) showed that no optimised model consistently beats 1/N
> out-of-sample across 7 datasets. The backtest above shows whether that holds here.
    """
)

deg_df = build_degradation_table(results, rf_annual)
if not deg_df.empty:
    deg_display = deg_df.copy()
    for col in deg_display.columns:
        deg_display[col] = deg_display[col].map(fmt_ratio)
    st.dataframe(deg_display, use_container_width=True)

# ─── Weight evolution ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Weight Evolution Over Time")
st.caption("How the optimal allocation changed at each rebalancing date.")

model_for_weights = st.selectbox(
    "Select model",
    list(results.keys()),
    key="wt_evolution_model",
)
wt_hist = results[model_for_weights].weights_history

if not wt_hist.empty:
    fig_wt = go.Figure()
    assets = list(wt_hist.columns)
    for j, asset in enumerate(assets):
        fig_wt.add_trace(go.Bar(
            x=wt_hist.index,
            y=wt_hist[asset].values * 100,
            name=asset,
            marker_color=PALETTE[j % len(PALETTE)],
        ))
    fig_wt.update_layout(
        barmode="stack",
        title=f"{model_for_weights} — Portfolio Weights at Each Rebalancing Date",
        yaxis_title="Weight (%)",
        yaxis=dict(ticksuffix="%"),
        **LAYOUT,
    )
    st.plotly_chart(fig_wt, use_container_width=True)

# ─── Turnover table ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Turnover & Transaction Costs")
st.caption(
    "Higher turnover means higher implementation costs. "
    "Equal Weight has near-zero turnover (weights are fixed). "
    "MaxSharpe typically has the highest turnover due to weight instability."
)

turnover_rows = []
for model, result in results.items():
    to_series = result.turnover_series.iloc[1:]  # skip first (initial investment)
    if to_series.empty:
        continue
    approx_cost_bps = float(to_series.mean()) * 5 * 10000  # ~5bps per 1% traded, one-way
    turnover_rows.append({
        "Model":              model,
        "Avg One-Way Turnover": fmt_pct(float(to_series.mean())),
        "Max Turnover":         fmt_pct(float(to_series.max())),
        "Min Turnover":         fmt_pct(float(to_series.min())),
        "Est. Cost/Rebal (bps)": f"{approx_cost_bps:.1f}",
    })

if turnover_rows:
    st.dataframe(pd.DataFrame(turnover_rows).set_index("Model"), use_container_width=True)

# ─── Methodology note ─────────────────────────────────────────────────────────
with st.expander("Methodology & Limitations", expanded=False):
    st.markdown(f"""
**Walk-Forward Protocol**
- Training window: {train_years} year{'s' if train_years > 1 else ''} ({train_window} trading days)
- Test window (rebalancing frequency): {rebal_freq}
- Weights computed at each rebalancing date using only prior data — no look-ahead bias
- Within each test window, weights are held constant (buy-and-hold approximation)

**What this test does and does not show**
- ✅ Correctly measures out-of-sample performance under realistic rebalancing
- ✅ Separates in-sample fitting from out-of-sample delivery
- ✅ Accounts for the fact that estimation error is the main enemy of portfolio optimisation
- ⚠️ Transaction costs are estimated (not exact) — see the Turnover page for detailed cost modelling
- ⚠️ The training window is fixed. In practice, managers adjust model parameters dynamically
- ⚠️ Regime changes (e.g. 2022 rate shock) can cause any model to underperform without it being a model failure

**Key reference**
DeMiguel, Garlappi & Uppal (2009), *"Optimal versus Naive Diversification"*,
Review of Financial Studies 22(5): 1915–1953.
Found that the 1/N rule outperformed 14 optimised models across 7 datasets on
out-of-sample Sharpe, certainty-equivalent return, and turnover.
    """)
