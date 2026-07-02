"""
Metric display components for Atlas PM.

Generates formatted metric cards, summary tables, and helper formatting
functions used across all Streamlit pages.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string, e.g. 0.1234 → '12.34%'."""
    if value != value:  # NaN check
        return "N/A"
    return f"{value:.{decimals}%}"


def fmt_ratio(value: float, decimals: int = 2) -> str:
    """Format a ratio, e.g. 1.23 → '1.23'."""
    if value != value:
        return "N/A"
    if abs(value) == float("inf"):
        return "∞"
    return f"{value:.{decimals}f}"


def fmt_currency(value: float, symbol: str = "£") -> str:
    """Format as currency, e.g. 1234567.0 → '£1,234,567'."""
    if value != value:
        return "N/A"
    return f"{symbol}{value:,.0f}"


def colour_value(value: float, invert: bool = False) -> str:
    """Return HTML-coloured span for positive (green) or negative (red) value."""
    positive = value > 0 if not invert else value < 0
    colour = POSITIVE_COLOR if positive else NEGATIVE_COLOR if value < 0 else NEUTRAL_COLOR
    return f'<span style="color:{colour}">{value:+.2%}</span>'


# ---------------------------------------------------------------------------
# Metric card row (Streamlit columns)
# ---------------------------------------------------------------------------

def render_metric_row(metrics: list[dict]) -> None:
    """
    Render a horizontal row of metric cards.

    Each dict in metrics should have:
        label   : str
        value   : str (pre-formatted)
        delta   : str | None  (optional change indicator)
        help    : str | None  (tooltip)
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                help=m.get("help"),
            )


def render_summary_table(stats: dict, label: str = "Portfolio") -> None:
    """
    Render a clean two-column table of key statistics.

    Groups metrics by category for readability.
    """
    GROUPS = {
        "Returns": [
            ("Total Return",     "Total Return",     fmt_pct),
            ("Ann. Return",      "Ann. Return",       fmt_pct),
        ],
        "Risk": [
            ("Ann. Volatility",  "Ann. Volatility",   fmt_pct),
            ("Max Drawdown",     "Max Drawdown",       fmt_pct),
        ],
        "Risk-Adjusted": [
            ("Sharpe Ratio",     "Sharpe",             fmt_ratio),
            ("Sortino Ratio",    "Sortino",            fmt_ratio),
            ("Calmar Ratio",     "Calmar",             fmt_ratio),
        ],
        "Distribution": [
            ("Skewness",         "Skewness",           lambda v: fmt_ratio(v, 3)),
            ("Kurtosis (excess)","Excess Kurtosis",    lambda v: fmt_ratio(v, 3)),
            ("% Positive Days",  "% Positive Days",    fmt_pct),
        ],
        "Market": [
            ("Beta",             "Beta (vs Benchmark)", lambda v: fmt_ratio(v, 3)),
            ("Alpha (ann.)",     "Jensen's Alpha (ann.)",fmt_pct),
            ("Information Ratio","Information Ratio",  fmt_ratio),
        ],
    }

    rows = []
    for group, items in GROUPS.items():
        for stat_key, display_name, formatter in items:
            if stat_key in stats:
                v = stats[stat_key]
                rows.append({
                    "Category": group,
                    "Metric":   display_name,
                    "Value":    formatter(v) if not (v != v) else "N/A",
                })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="small"),
                "Metric":   st.column_config.TextColumn("Metric"),
                "Value":    st.column_config.TextColumn("Value", width="small"),
            },
        )


def render_weights_table(weights: pd.Series) -> None:
    """Render a styled weights table with a progress bar per row."""
    df = pd.DataFrame({
        "Asset":  weights.index.tolist(),
        "Weight": weights.values.tolist(),
    })
    df = df[df["Weight"] > 0.001].sort_values("Weight", ascending=False)
    df["Weight %"] = df["Weight"].apply(fmt_pct)

    st.dataframe(
        df[["Asset", "Weight"]].rename(columns={"Weight": "Allocation"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Asset": st.column_config.TextColumn("Asset"),
            "Allocation": st.column_config.ProgressColumn(
                "Allocation",
                min_value=0,
                max_value=1,
                format="%.1%",
            ),
        },
    )


def render_governance_notice() -> None:
    """Render the standard governance / disclaimer notice."""
    st.info(
        """
**Model Governance & Limitations**

- **Data source**: Yahoo Finance via yfinance (free data; may contain errors or gaps)
- **Survivorship bias**: ETF proxies mitigate but do not eliminate this concern
- **Look-ahead bias**: all analytics use only data available at the time of calculation
- **Return estimation**: historical mean returns are poor predictors of future returns
- **Transaction costs**: not modelled — real portfolios incur bid/offer spreads, commissions, and market impact
- **Liquidity**: ETFs assumed fully liquid; real execution may differ
- **Rebalancing**: assumes frictionless rebalancing at month-end prices
- **AI commentary**: generated by Claude (LLM) — may contain errors; requires human review
- **Not investment advice**: Atlas PM is a portfolio analytics tool for educational and research purposes only
        """,
        icon="⚠️",
    )
