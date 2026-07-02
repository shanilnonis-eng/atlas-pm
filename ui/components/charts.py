"""
Reusable Plotly chart components for Atlas PM.

All charts use a consistent institutional dark-navy/white colour palette.
Each function returns a plotly Figure object so the caller decides how to
render it (st.plotly_chart, export, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.settings import (
    POSITIVE_COLOR,
    NEGATIVE_COLOR,
    NEUTRAL_COLOR,
    ACCENT_COLOR,
)

# Institutional colour palette
PALETTE = [
    "#1a3a5c",   # dark navy
    "#2e86ab",   # steel blue
    "#e84855",   # red
    "#f4a261",   # amber
    "#57cc99",   # green
    "#9b5de5",   # purple
    "#f72585",   # pink
    "#4cc9f0",   # light blue
    "#b5e48c",   # lime
    "#ffb703",   # gold
]

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, Arial, sans-serif", size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#e0e0e0",
        borderwidth=1,
    ),
)


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=ACCENT_COLOR)),
                      **LAYOUT_DEFAULTS)
    return fig


# ---------------------------------------------------------------------------
# Cumulative returns chart
# ---------------------------------------------------------------------------

def cumulative_returns_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    title: str = "Cumulative Portfolio Return",
) -> go.Figure:
    """Wealth index chart starting at 1.0 (100%)."""
    fig = go.Figure()

    wealth = (1 + portfolio_returns).cumprod()
    fig.add_trace(go.Scatter(
        x=wealth.index, y=wealth.values,
        name="Portfolio",
        line=dict(color=ACCENT_COLOR, width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:.3f}x<extra></extra>",
    ))

    if benchmark_returns is not None:
        aligned_bench = benchmark_returns.reindex(portfolio_returns.index).dropna()
        bench_wealth  = (1 + aligned_bench).cumprod()
        fig.add_trace(go.Scatter(
            x=bench_wealth.index, y=bench_wealth.values,
            name="Benchmark",
            line=dict(color=NEUTRAL_COLOR, width=1.5, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>%{y:.3f}x<extra></extra>",
        ))

    fig.update_yaxes(title_text="Growth of £1", tickformat=".2f")
    fig.update_xaxes(title_text="")
    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------------------------

def drawdown_chart(
    drawdown: pd.Series,
    title: str = "Portfolio Drawdown",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        fill="tozeroy",
        fillcolor="rgba(232, 72, 85, 0.2)",
        line=dict(color=NEGATIVE_COLOR, width=1.5),
        name="Drawdown",
        hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Rolling metrics chart
# ---------------------------------------------------------------------------

def rolling_volatility_chart(
    rolling_vol: pd.Series,
    title: str = "Rolling Annualised Volatility",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol.values * 100,
        line=dict(color="#2e86ab", width=2),
        name="Rolling Vol",
        hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_yaxes(title_text="Annualised Volatility (%)", ticksuffix="%")
    return _apply_defaults(fig, title)


def rolling_sharpe_chart(
    rolling_sharpe: pd.Series,
    title: str = "Rolling Sharpe Ratio",
) -> go.Figure:
    fig = go.Figure()
    # colour above/below zero
    pos = rolling_sharpe.copy()
    neg = rolling_sharpe.copy()
    pos[pos < 0] = 0
    neg[neg > 0] = 0

    fig.add_trace(go.Scatter(
        x=pos.index, y=pos.values,
        fill="tozeroy",
        fillcolor="rgba(46,134,171,0.25)",
        line=dict(color="#2e86ab", width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=neg.index, y=neg.values,
        fill="tozeroy",
        fillcolor="rgba(232,72,85,0.25)",
        line=dict(color=NEGATIVE_COLOR, width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        line=dict(color=ACCENT_COLOR, width=2),
        name="Rolling Sharpe",
        hovertemplate="%{x|%b %Y}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_yaxes(title_text="Sharpe Ratio")
    return _apply_defaults(fig, title)


# ---------------------------------------------------------------------------
# Allocation charts
# ---------------------------------------------------------------------------

def allocation_pie_chart(
    weights: pd.Series | dict,
    title: str = "Portfolio Allocation",
) -> go.Figure:
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    weights = weights[weights > 0.001]  # hide near-zero allocations

    fig = go.Figure(go.Pie(
        labels=weights.index.tolist(),
        values=weights.values.tolist(),
        hole=0.45,
        marker_colors=PALETTE[:len(weights)],
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:.2%}<extra></extra>",
    ))
    fig.update_layout(showlegend=True, **LAYOUT_DEFAULTS)
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=ACCENT_COLOR)))
    return fig


def allocation_bar_chart(
    weights_dict: dict[str, pd.Series],
    title: str = "Model Comparison — Weights",
) -> go.Figure:
    """Side-by-side bar chart comparing multiple model weights."""
    fig = go.Figure()
    for i, (model_name, weights) in enumerate(weights_dict.items()):
        fig.add_trace(go.Bar(
            name=model_name,
            x=weights.index.tolist(),
            y=weights.values.tolist(),
            marker_color=PALETTE[i % len(PALETTE)],
            hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
        ))
    fig.update_layout(barmode="group", **LAYOUT_DEFAULTS)
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=ACCENT_COLOR)))
    fig.update_yaxes(title_text="Weight", tickformat=".0%")
    return fig


# ---------------------------------------------------------------------------
# Correlation matrix heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
) -> go.Figure:
    labels = corr_matrix.columns.tolist()
    z = corr_matrix.values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[
            [0.0,  "#d62728"],
            [0.5,  "#ffffff"],
            [1.0,  "#1a3a5c"],
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="ρ", tickformat=".1f"),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=ACCENT_COLOR)))
    return fig


# ---------------------------------------------------------------------------
# Monthly returns heatmap
# ---------------------------------------------------------------------------

def monthly_returns_heatmap(
    monthly_table: pd.DataFrame,
    title: str = "Monthly Returns Heatmap",
) -> go.Figure:
    # exclude 'Annual' column for the colour scale
    cols = [c for c in monthly_table.columns if c != "Annual"]
    z = monthly_table[cols].values * 100  # convert to %

    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=cols,
        y=monthly_table.index.tolist(),
        colorscale=[
            [0.0, "#d62728"],
            [0.5, "#ffffff"],
            [1.0, "#1a3a5c"],
        ],
        zmid=0,
        text=text,
        texttemplate="%{text}",
        colorbar=dict(title="%", tickformat=".0f"),
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=ACCENT_COLOR)),
        height=max(300, 40 * len(monthly_table)),
    )
    return fig


# ---------------------------------------------------------------------------
# Risk contribution chart
# ---------------------------------------------------------------------------

def risk_contribution_bar(
    risk_df: pd.DataFrame,
    title: str = "Risk Contribution by Asset",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=risk_df.index.tolist(),
        y=risk_df["% Risk Contribution"].values * 100,
        marker_color=PALETTE[:len(risk_df)],
        hovertemplate="%{x}<br>Risk contribution: %{y:.1f}%<extra></extra>",
        name="Risk Contribution",
    ))
    # overlay the weight for comparison
    if "Weight" in risk_df.columns:
        fig.add_trace(go.Scatter(
            x=risk_df.index.tolist(),
            y=risk_df["Weight"].values * 100,
            mode="markers",
            marker=dict(symbol="diamond", size=10, color=ACCENT_COLOR),
            name="Weight",
            hovertemplate="%{x}<br>Weight: %{y:.1f}%<extra></extra>",
        ))
    fig.update_yaxes(title_text="Contribution (%)", ticksuffix="%")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=ACCENT_COLOR)))
    return fig


# ---------------------------------------------------------------------------
# Stress test chart
# ---------------------------------------------------------------------------

def stress_test_bar(
    stress_df: pd.DataFrame,
    title: str = "Stress Test Results — Portfolio P&L",
) -> go.Figure:
    pnl = stress_df["Portfolio P&L"].sort_values()
    colors = [NEGATIVE_COLOR if v < 0 else POSITIVE_COLOR for v in pnl.values]

    fig = go.Figure(go.Bar(
        x=pnl.values * 100,
        y=pnl.index.tolist(),
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>P&L: %{x:.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="grey", line_dash="dot")
    fig.update_xaxes(title_text="Portfolio P&L (%)", ticksuffix="%")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=ACCENT_COLOR)),
        height=max(300, 60 * len(pnl)),
    )
    return fig


# ---------------------------------------------------------------------------
# VaR returns distribution chart
# ---------------------------------------------------------------------------

def var_distribution_chart(
    returns: pd.Series,
    var_pct: float,
    cvar_pct: float,
    confidence: float = 0.95,
    title: str = "Return Distribution & VaR",
) -> go.Figure:
    fig = go.Figure()

    vals = returns.values * 100
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=80,
        name="Daily Returns",
        marker_color=ACCENT_COLOR,
        opacity=0.7,
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))

    fig.add_vline(
        x=-var_pct * 100,
        line_color=NEGATIVE_COLOR, line_dash="dash",
        annotation_text=f"VaR ({int(confidence*100)}%) = {var_pct:.2%}",
        annotation_position="top right",
        annotation_font_color=NEGATIVE_COLOR,
    )
    fig.add_vline(
        x=-cvar_pct * 100,
        line_color="#8b0000", line_dash="dot",
        annotation_text=f"CVaR = {cvar_pct:.2%}",
        annotation_position="bottom right",
        annotation_font_color="#8b0000",
    )

    fig.update_xaxes(title_text="Daily Return (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Frequency")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=ACCENT_COLOR)))
    return fig


# ---------------------------------------------------------------------------
# Efficient frontier chart
# ---------------------------------------------------------------------------

def efficient_frontier_chart(
    frontier: pd.DataFrame,
    portfolio_points: dict[str, tuple[float, float]] | None = None,
    title: str = "Efficient Frontier",
    rf_annual: float | None = None,
    max_sharpe_point: tuple[float, float] | None = None,
) -> go.Figure:
    """
    Plot the efficient frontier with optional overlay of portfolio points and CML.

    portfolio_points  : dict of {label: (ann_vol, ann_return)} — fractions, not percent.
    rf_annual         : annual risk-free rate (fraction); required to draw the CML.
    max_sharpe_point  : (ann_vol, ann_return) of the Max Sharpe portfolio — required for CML.
                        Both rf_annual and max_sharpe_point must be provided to draw the CML.
    """
    fig = go.Figure()

    # frontier line
    fig.add_trace(go.Scatter(
        x=frontier["Ann. Volatility"] * 100,
        y=frontier["Ann. Return"] * 100,
        mode="lines",
        line=dict(color="#2e86ab", width=2.5),
        name="Efficient Frontier",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    # Capital Market Line — drawn before overlay points so stars appear on top
    if rf_annual is not None and max_sharpe_point is not None:
        vol_ms, ret_ms = max_sharpe_point
        if vol_ms > 1e-8:
            slope = (ret_ms - rf_annual) / vol_ms
            cml_vols = np.array([0.0, vol_ms * 1.5])
            cml_rets = rf_annual + slope * cml_vols
            fig.add_trace(go.Scatter(
                x=cml_vols * 100,
                y=cml_rets * 100,
                mode="lines",
                line=dict(color="#f4a261", width=1.8, dash="dash"),
                name="Capital Market Line",
                hovertemplate="CML<br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
            ))

    # portfolio overlay points
    if portfolio_points:
        for i, (label, (vol, ret)) in enumerate(portfolio_points.items()):
            fig.add_trace(go.Scatter(
                x=[vol * 100], y=[ret * 100],
                mode="markers+text",
                marker=dict(size=12, color=PALETTE[i % len(PALETTE)], symbol="star"),
                text=[label],
                textposition="top center",
                name=label,
                hovertemplate=f"<b>{label}</b><br>Vol: {vol:.2%}<br>Return: {ret:.2%}<extra></extra>",
            ))

    fig.update_xaxes(title_text="Annualised Volatility (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Annualised Return (%)", ticksuffix="%")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=ACCENT_COLOR)))
    return fig
