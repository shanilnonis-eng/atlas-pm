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
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
from analytics.active_share import (
    calculate_active_share, calculate_tracking_error,
    active_weight_breakdown, build_benchmark_weights,
    active_share_classification, te_classification, quadrant_label,
    TE_THRESHOLD_LOW, TE_THRESHOLD_HIGH,
    AS_MODERATE, AS_GENUINE,
)

st.title("Performance Analytics")
st.markdown("---")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if "portfolio_returns" not in st.session_state:
    st.warning("Please complete **Portfolio Construction** first.", icon="⚠️")
    st.stop()

port_rets      = st.session_state["portfolio_returns"]
bench_rets     = st.session_state["bench_returns"]
rf_rets        = st.session_state["rf_returns"]
model_lbl      = st.session_state.get("current_model", "Portfolio")
stats          = st.session_state.get("portfolio_stats", {})
current_weights = st.session_state.get("current_weights")

# Pre-compute Active Share and Tracking Error for headline + tab
bench_rets_aligned_full = bench_rets.reindex(port_rets.index).dropna()
_te = calculate_tracking_error(port_rets, bench_rets_aligned_full)
if current_weights is not None:
    _bench_w  = build_benchmark_weights(current_weights)
    _as       = calculate_active_share(current_weights, _bench_w)
    _as_class = active_share_classification(_as)
else:
    _as = float("nan")
    _bench_w  = None
    _as_class = {"label": "N/A", "description": "", "color": "#999"}

period_str = (
    f"{port_rets.index[0].strftime('%d %b %Y')} — "
    f"{port_rets.index[-1].strftime('%d %b %Y')}"
)
bench_rets_aligned = bench_rets_aligned_full   # alias used throughout the page

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

# Second row — benchmark-relative metrics
render_metric_row([
    {"label": "Active Share",
     "value": fmt_pct(_as) if _as == _as else "N/A",
     "help": "0.5 × Σ|w_portfolio - w_benchmark|. 0% = identical, 100% = no overlap."},
    {"label": "Tracking Error (Ann.)",
     "value": fmt_pct(_te),
     "help": "Annualised std dev of (portfolio − benchmark) daily returns."},
    {"label": "Active Classification",
     "value": _as_class["label"],
     "help": "Based on Cremers & Petajisto (2009). See the Active Share tab for detail."},
    {"label": "Information Ratio",
     "value": fmt_ratio(stats.get("Information Ratio", float("nan"))),
     "help": "Annualised active return ÷ tracking error. Measures alpha generation per unit of active risk."},
])

st.markdown("---")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_perf, tab_dd, tab_roll, tab_monthly, tab_stats, tab_active = st.tabs([
    "Cumulative Returns",
    "Drawdown Analysis",
    "Rolling Metrics",
    "Monthly Returns",
    "Full Statistics",
    "Active Share",
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

# ---------------------------------------------------------------------------
# Tab 6: Active Share
# ---------------------------------------------------------------------------
with tab_active:
    import numpy as np
    import plotly.graph_objects as go

    st.subheader("Active Share & Tracking Error")
    st.markdown(f"""
**Active Share** measures how different the portfolio's weights are from the benchmark
(**{BENCHMARK_LABEL}**). **Tracking Error** measures how different the *returns* are.
Together they characterise the nature of active management.

> *Active Share = 0.5 × Σ |w_portfolio − w_benchmark|*
> Range: 0 % (identical to benchmark) → 100 % (no overlap at all)

Reference: Cremers & Petajisto (2009) *'How Active Is Your Fund Manager?'*
    """)

    if current_weights is None:
        st.warning(
            "Portfolio weights not available. Please run **Portfolio Construction** first.",
            icon="⚠️",
        )
    else:
        # ── Headline numbers ──────────────────────────────────────────────
        _q_lbl = quadrant_label(_as, _te)
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric(
            "Active Share",
            fmt_pct(_as),
            help="0% = perfect index fund.  100% = completely different from benchmark.",
        )
        _m2.metric(
            "Tracking Error (Ann.)",
            fmt_pct(_te),
            help="Annualised std dev of daily active returns (portfolio − benchmark).",
        )
        _m3.metric(
            "Classification",
            _as_class["label"],
            help="Cremers & Petajisto (2009) classification.",
        )
        _m4.metric(
            "Manager Quadrant",
            _q_lbl,
            help="2×2 framework: High/Low Active Share × High/Low Tracking Error.",
        )

        # ── Active weight breakdown chart ─────────────────────────────────
        st.markdown("---")
        st.subheader("Active Positions vs Benchmark")
        st.caption(
            "Active Weight = Portfolio Weight − Benchmark Weight. "
            "Green = overweight vs benchmark.  Red = underweight."
        )

        _breakdown = active_weight_breakdown(current_weights, _bench_w)
        _colors = [
            "#57cc99" if v >= 0 else "#e84855"
            for v in _breakdown["Active Weight"]
        ]

        _fig_aw = go.Figure()
        _fig_aw.add_trace(go.Bar(
            y=_breakdown["Asset"],
            x=_breakdown["Active Weight"] * 100,
            orientation="h",
            marker_color=_colors,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Active Weight: %{x:.2f}%<br>"
                "Portfolio: " + _breakdown["Portfolio Weight"].map(lambda v: f"{v:.1%}") + "<br>"
                "Benchmark: " + _breakdown["Benchmark Weight"].map(lambda v: f"{v:.1%}") +
                "<extra></extra>"
            ),
            name="Active Weight",
        ))
        _fig_aw.add_vline(x=0, line_color="grey", line_dash="dot")
        _fig_aw.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Active Weight (pp)", xaxis_ticksuffix="%",
            font=dict(family="Inter, Arial, sans-serif"),
            height=max(250, 45 * len(_breakdown)),
            margin=dict(l=200, r=20, t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(_fig_aw, use_container_width=True)

        # ── Weight comparison table ───────────────────────────────────────
        with st.expander("Weight detail", expanded=False):
            _bd_display = _breakdown.copy()
            for _col in ["Portfolio Weight", "Benchmark Weight", "Active Weight"]:
                _bd_display[_col] = _bd_display[_col].map(fmt_pct)
            st.dataframe(_bd_display, use_container_width=True, hide_index=True)

        # ── 2×2 Quadrant chart ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Active Management Quadrant (Cremers & Petajisto 2009)")

        _te_pct = _te * 100
        _as_pct = _as * 100
        _te_lo  = TE_THRESHOLD_LOW  * 100
        _te_hi  = TE_THRESHOLD_HIGH * 100
        _as_lo  = AS_MODERATE * 100

        _fig_q = go.Figure()

        # Background quadrants
        _quadrant_defs = [
            (0,     _te_lo, _as_lo, 100, "rgba(232,72,85,0.10)",   "Closet Indexer"),
            (_te_lo, 25,   _as_lo, 100, "rgba(244,162,97,0.10)",   "Selective Active"),
            (0,     _te_lo, 0,     _as_lo, "rgba(244,162,97,0.10)", "Diversified Factor Bets"),
            (_te_lo, 25,   0,     _as_lo, "rgba(87,204,153,0.15)", "Active Allocator"),
        ]
        for _x0, _x1, _y0, _y1, _col, _lbl in _quadrant_defs:
            _fig_q.add_shape(
                type="rect", x0=_x0, x1=_x1, y0=_y0, y1=_y1,
                fillcolor=_col, line_width=0, layer="below",
            )
            _fig_q.add_annotation(
                x=(_x0 + _x1) / 2, y=(_y0 + _y1) / 2,
                text=_lbl, showarrow=False,
                font=dict(size=11, color="#555"),
                opacity=0.7,
            )

        # Threshold lines
        _fig_q.add_vline(x=_te_lo, line_dash="dash", line_color="#aaa", line_width=1)
        _fig_q.add_hline(y=_as_lo, line_dash="dash", line_color="#aaa", line_width=1)

        # Current portfolio
        _fig_q.add_trace(go.Scatter(
            x=[_te_pct], y=[_as_pct],
            mode="markers+text",
            marker=dict(size=16, color=_as_class["color"], symbol="diamond",
                        line=dict(color="white", width=2)),
            text=[model_lbl],
            textposition="top right",
            name=model_lbl,
            hovertemplate=(
                f"<b>{model_lbl}</b><br>"
                f"Active Share: {_as_pct:.1f}%<br>"
                f"Tracking Error: {_te_pct:.1f}%<br>"
                f"Quadrant: {_q_lbl}<extra></extra>"
            ),
        ))

        _fig_q.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Tracking Error (Ann. %)", range=[0, 25],
                       ticksuffix="%", showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(title="Active Share (%)", range=[0, 100],
                       ticksuffix="%", showgrid=True, gridcolor="#f0f0f0"),
            font=dict(family="Inter, Arial, sans-serif"),
            showlegend=False,
            height=420,
        )
        st.plotly_chart(_fig_q, use_container_width=True)

        # ── Interpretation ────────────────────────────────────────────────
        with st.expander("How to interpret Active Share", expanded=False):
            st.markdown(f"""
**Active Share — what it measures:**
Active Share quantifies the percentage of the portfolio that differs from the benchmark.
It is purely weight-based — it does not depend on return realisation.

**Tracking Error — what it adds:**
Tracking Error measures the *return* difference. A portfolio can have high Active Share
(very different weights) but low Tracking Error (if the active positions happen to move
similarly to the benchmark). Cremers & Petajisto found that only **high AS + high TE**
reliably predicts outperformance — the combination of genuine active bets that also
manifest in returns.

**The 2×2 framework:**

| | Low Tracking Error | High Tracking Error |
|---|---|---|
| **High Active Share** | Diversified factor bets | Active Allocator / Stock Picker |
| **Low Active Share** | Closet Indexer | (not sustainable) |

**For multi-asset portfolios:**
Active Share vs a single-equity benchmark (SPY) is expected to be higher than for
equity-only funds, because bonds, gold, and commodities are not in SPY by definition.
A multi-asset portfolio with Active Share of 70–90% is not exceptional — it simply
reflects broad diversification.

**This portfolio: {_as_class['label']}**
{_as_class['description']}

Active Share: **{fmt_pct(_as)}** · Tracking Error: **{fmt_pct(_te)}** · Quadrant: **{_q_lbl}**

---
*Active Share is a descriptive metric. High Active Share does not guarantee outperformance.
It is a necessary but not sufficient condition for alpha generation.*
            """)

        st.info(
            "**Benchmark context:** Active Share is computed relative to "
            f"**{BENCHMARK_LABEL}** (100 % single-asset benchmark). "
            "For equity-only mandates, the relevant benchmark would be "
            "a cap-weighted equity index with multiple constituents, which "
            "would produce different Active Share values.",
            icon="ℹ️",
        )
