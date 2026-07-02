"""
Page 3 — Performance Analytics

Full suite of return and risk-adjusted performance analytics:
- Summary statistics vs benchmark
- Cumulative returns
- Drawdown analysis
- Rolling metrics
- Monthly returns heatmap
- Beta / alpha decomposition
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd

from analytics.returns import (
    summary_statistics, cumulative_returns, drawdown_series,
    rolling_volatility, rolling_sharpe, monthly_returns_table,
    drawdown_duration,
)
from ui.components.charts import (
    cumulative_returns_chart, drawdown_chart,
    rolling_volatility_chart, rolling_sharpe_chart,
    monthly_returns_heatmap,
)
from ui.components.metrics import (
    render_metric_row, render_summary_table, fmt_pct, fmt_ratio,
)
from config.settings import ROLLING_WINDOW, BENCHMARK_LABEL

st.set_page_config(page_title="Performance Analytics | Atlas PM", layout="wide")
st.title("Performance Analytics")
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
model_lbl   = st.session_state.get("current_model", "Portfolio")
stats       = st.session_state.get("portfolio_stats", {})

period_str = (
    f"{port_rets.index[0].strftime('%d %b %Y')} — "
    f"{port_rets.index[-1].strftime('%d %b %Y')}"
)
bench_rets_aligned = bench_rets.reindex(port_rets.index).dropna()

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
st.subheader(f"Portfolio: {model_lbl} | Period: {period_str}")

render_metric_row([
    {"label": "Total Return",         "value": fmt_pct(stats.get("Total Return", float("nan"))),
     "help": "Geometric total return over the full period"},
    {"label": "Ann. Return (CAGR)",   "value": fmt_pct(stats.get("Ann. Return", float("nan"))),
     "help": "Compound annual growth rate"},
    {"label": "Ann. Volatility",      "value": fmt_pct(stats.get("Ann. Volatility", float("nan"))),
     "help": "Annualised standard deviation of daily returns"},
    {"label": "Sharpe Ratio",         "value": fmt_ratio(stats.get("Sharpe Ratio", float("nan"))),
     "help": "(Ann. excess return) / Ann. volatility"},
    {"label": "Max Drawdown",         "value": fmt_pct(stats.get("Max Drawdown", float("nan"))),
     "help": "Largest peak-to-trough decline"},
    {"label": "Sortino Ratio",        "value": fmt_ratio(stats.get("Sortino Ratio", float("nan"))),
     "help": "(Ann. excess return) / Downside deviation"},
])

st.markdown("---")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_perf, tab_dd, tab_roll, tab_monthly, tab_stats = st.tabs([
    "Cumulative Returns",
    "Drawdown Analysis",
    "Rolling Metrics",
    "Monthly Returns",
    "Full Statistics",
])

# --- Tab 1: Cumulative returns ---
with tab_perf:
    st.subheader("Portfolio vs Benchmark — Cumulative Return")
    fig = cumulative_returns_chart(port_rets, bench_rets_aligned, title="")
    st.plotly_chart(fig, use_container_width=True)

    # relative performance
    port_wealth  = (1 + port_rets).cumprod()
    bench_wealth = (1 + bench_rets_aligned).cumprod()
    relative     = port_wealth / bench_wealth

    import plotly.graph_objects as go
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=relative.index, y=relative.values,
        fill="tozeroy",
        fillcolor="rgba(26,58,92,0.12)",
        line=dict(color="#1a3a5c", width=2),
        name="Relative Wealth",
        hovertemplate="%{x|%b %Y}<br>Relative: %{y:.3f}x<extra></extra>",
    ))
    fig2.add_hline(y=1, line_dash="dot", line_color="grey")
    fig2.update_layout(
        title="Relative Performance vs Benchmark (> 1.0 = outperformance)",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        yaxis_title="Relative Wealth",
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: Drawdown ---
with tab_dd:
    dd_series = drawdown_series(port_rets)
    st.subheader("Drawdown History")
    st.plotly_chart(drawdown_chart(dd_series), use_container_width=True)

    st.subheader("Drawdown Episodes")
    dd_table = drawdown_duration(port_rets)
    if not dd_table.empty:
        dd_display = dd_table.copy()
        dd_display["Depth"] = dd_display["Depth"].map(fmt_pct)
        for col in ["Start","Trough","Recovery"]:
            dd_display[col] = dd_display[col].apply(
                lambda d: d.strftime("%d %b %Y") if pd.notna(d) else "Ongoing"
            )
        dd_display = dd_display.sort_values("Depth")
        st.dataframe(dd_display, use_container_width=True, hide_index=True)
    else:
        st.info("No drawdown episodes identified.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Maximum Drawdown",
                  fmt_pct(stats.get("Max Drawdown", float("nan"))),
                  help="Worst peak-to-trough loss")
    with col2:
        if not dd_table.empty:
            worst = dd_table.loc[dd_table["Depth"].idxmin()]
            dur = worst.get("Duration (days)", "N/A")
            st.metric("Longest Drawdown Duration", f"{dur:,.0f} days" if dur else "N/A")

# --- Tab 3: Rolling metrics ---
with tab_roll:
    window = st.slider(
        "Rolling window (trading days)",
        min_value=21, max_value=252, value=ROLLING_WINDOW, step=21,
        help="63 ≈ 3 months, 126 ≈ 6 months, 252 ≈ 1 year",
    )

    roll_vol   = rolling_volatility(port_rets, window)
    roll_sharpe = rolling_sharpe(port_rets, rf_rets.reindex(port_rets.index), window)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(rolling_volatility_chart(roll_vol), use_container_width=True)
    with col2:
        st.plotly_chart(rolling_sharpe_chart(roll_sharpe), use_container_width=True)

    # rolling beta
    import numpy as np
    roll_cov  = port_rets.rolling(window).cov(bench_rets_aligned)
    roll_bvar = bench_rets_aligned.rolling(window).var()
    roll_beta = roll_cov / roll_bvar

    fig_beta = go.Figure()
    fig_beta.add_trace(go.Scatter(
        x=roll_beta.index, y=roll_beta.values,
        line=dict(color="#2e86ab", width=2),
        name=f"Rolling Beta ({window}d)",
        hovertemplate="%{x|%b %Y}<br>Beta: %{y:.2f}<extra></extra>",
    ))
    fig_beta.add_hline(y=1, line_dash="dot", line_color="grey",
                       annotation_text="β = 1 (benchmark)")
    fig_beta.update_layout(
        title=f"Rolling Beta vs Benchmark ({window}d window)",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        yaxis_title="Beta",
    )
    st.plotly_chart(fig_beta, use_container_width=True)

# --- Tab 4: Monthly returns ---
with tab_monthly:
    monthly_tbl = monthly_returns_table(port_rets)
    st.plotly_chart(
        monthly_returns_heatmap(monthly_tbl,
                                title=f"{model_lbl} — Monthly Returns Heatmap"),
        use_container_width=True,
    )

    # annual return bar
    annual_rets = monthly_tbl["Annual"].dropna()
    colors = ["#57cc99" if v >= 0 else "#e84855" for v in annual_rets.values]
    fig_ann = go.Figure(go.Bar(
        x=annual_rets.index.tolist(),
        y=annual_rets.values * 100,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig_ann.add_hline(y=0, line_color="grey", line_dash="dot")
    fig_ann.update_layout(
        title="Annual Returns by Year",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Return (%)", yaxis_ticksuffix="%",
        font=dict(family="Inter, Arial, sans-serif"),
    )
    st.plotly_chart(fig_ann, use_container_width=True)

# --- Tab 5: Full statistics ---
with tab_stats:
    col_port, col_bench = st.columns(2)

    with col_port:
        st.subheader(f"{model_lbl} Statistics")
        render_summary_table(stats, label=model_lbl)

    with col_bench:
        st.subheader(f"Benchmark ({BENCHMARK_LABEL})")
        bench_stats = summary_statistics(
            bench_rets_aligned,
            rf_returns=rf_rets.reindex(bench_rets_aligned.index),
            label="Benchmark",
        )
        render_summary_table(bench_stats, label="Benchmark")

    st.markdown("---")
    st.caption("""
**Statistical notes:** Sharpe and Sortino ratios are annualised. Skewness < 0 indicates
left-tail risk (more negative outliers). Excess kurtosis > 0 indicates fat tails
(more frequent extreme returns than a normal distribution). These properties are
typical of equity returns and mean Gaussian VaR will underestimate tail risk.
    """)
