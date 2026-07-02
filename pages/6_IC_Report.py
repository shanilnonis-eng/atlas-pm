"""
Page 6 — Investment Committee Report

Assembles a full monthly IC report combining all analytics into a
structured, printable document with AI-generated narrative sections.
Also includes the Governance & Model Risk section.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from analytics.returns import (
    summary_statistics, monthly_returns_table,
    drawdown_duration,
)
from reporting.pdf_export import generate_pdf_report, is_available as pdf_available
from analytics.risk import (
    historical_var, historical_cvar, run_stress_test,
    component_risk_contribution,
)
from construction.optimiser import compute_cov_matrix
from ai.commentary import generate_ic_report_narrative
from ui.components.charts import (
    cumulative_returns_chart, drawdown_chart,
    allocation_pie_chart, stress_test_bar,
    correlation_heatmap, monthly_returns_heatmap,
    risk_contribution_bar,
)
from ui.components.metrics import (
    render_metric_row, render_governance_notice,
    fmt_pct, fmt_ratio, fmt_currency,
)
from config.settings import (
    VAR_CONFIDENCE, BENCHMARK_LABEL, ASSET_SHORT_NAMES,
    APP_TITLE, APP_SUBTITLE,
)


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
stats       = st.session_state.get("portfolio_stats", {})
simple_rets = st.session_state["simple_returns"]

period_str = (
    f"{port_rets.index[0].strftime('%d %b %Y')} — "
    f"{port_rets.index[-1].strftime('%d %b %Y')}"
)
report_date = datetime.today().strftime("%d %B %Y")

# precompute
var_1d  = historical_var(port_rets, VAR_CONFIDENCE, 1)
cvar_1d = historical_cvar(port_rets, VAR_CONFIDENCE, 1)
ann_vol = float(port_rets.std(ddof=1) * np.sqrt(252))
max_dd  = float(((1 + port_rets).cumprod() / (1 + port_rets).cumprod().cummax() - 1).min())

stress_df = run_stress_test(weights.to_dict())
w_aligned = weights.reindex(simple_rets.columns).dropna()
w_aligned = w_aligned / w_aligned.sum()
cov       = compute_cov_matrix(simple_rets[w_aligned.index])
risk_df   = component_risk_contribution(w_aligned, cov)

top_stress = [
    {"name": s, "pnl": stress_df.loc[s, "Portfolio P&L"]}
    for s in stress_df.index
]

# ---------------------------------------------------------------------------
# Report header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1a3a5c, #2e86ab);
    padding: 2.5rem 2rem;
    border-radius: 8px;
    color: white;
    margin-bottom: 1.5rem;
">
    <h1 style="color:white; margin:0; font-size:2rem;">{APP_TITLE}</h1>
    <h3 style="color:rgba(255,255,255,0.8); margin:0.3rem 0 0 0;">Investment Committee Report</h3>
    <p style="color:rgba(255,255,255,0.6); margin:0.5rem 0 0 0;">
        Report Date: {report_date} &nbsp;|&nbsp;
        Portfolio: {model_lbl} &nbsp;|&nbsp;
        Period: {period_str}
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: Executive summary metrics
# ---------------------------------------------------------------------------
st.markdown("## 1. Executive Summary")

render_metric_row([
    {"label": "Total Return",      "value": fmt_pct(stats.get("Total Return",0)),
     "delta": None},
    {"label": "Ann. Return",       "value": fmt_pct(stats.get("Ann. Return",0))},
    {"label": "Ann. Volatility",   "value": fmt_pct(ann_vol)},
    {"label": "Sharpe Ratio",      "value": fmt_ratio(stats.get("Sharpe Ratio",0))},
    {"label": "Max Drawdown",      "value": fmt_pct(max_dd)},
    {"label": f"1-Day VaR ({int(VAR_CONFIDENCE*100)}%)", "value": fmt_pct(var_1d)},
])

st.markdown("---")

# ---------------------------------------------------------------------------
# AI narrative (optional)
# ---------------------------------------------------------------------------
with st.expander("Generate AI Narrative (requires API key)", expanded=False):
    if st.button("Generate Full IC Report Narrative", type="primary"):
        with st.spinner("Claude is drafting the IC report narrative…"):
            narrative = generate_ic_report_narrative(
                period=period_str,
                model_name=model_lbl,
                stats=stats,
                weights=weights.to_dict(),
                top_stress_scenarios=top_stress,
            )
        st.markdown(narrative)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2: Portfolio allocation
# ---------------------------------------------------------------------------
st.markdown("## 2. Portfolio Allocation")
col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(
        allocation_pie_chart(weights, title="Current Allocation"),
        use_container_width=True,
    )

with col2:
    alloc_df = pd.DataFrame({
        "Asset": weights.index.tolist(),
        "Weight": weights.values.tolist(),
    }).sort_values("Weight", ascending=False)
    alloc_df["Weight %"] = alloc_df["Weight"].map(fmt_pct)

    top3 = alloc_df.head(3)
    st.metric("Top Holding", top3.iloc[0]["Asset"], top3.iloc[0]["Weight %"])
    st.metric("2nd Holding", top3.iloc[1]["Asset"], top3.iloc[1]["Weight %"])
    st.metric("3rd Holding", top3.iloc[2]["Asset"], top3.iloc[2]["Weight %"])

    n_meaningful = (weights > 0.02).sum()
    st.metric("Effective Assets (>2%)", str(n_meaningful))

    herfindahl = float((weights ** 2).sum())
    st.metric("Concentration (HHI)", f"{herfindahl:.4f}",
              help="Herfindahl-Hirschman Index: 0=perfect diversification, 1=full concentration")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Performance
# ---------------------------------------------------------------------------
st.markdown("## 3. Performance Analysis")

bench_aligned = bench_rets.reindex(port_rets.index).dropna()
st.plotly_chart(
    cumulative_returns_chart(port_rets, bench_aligned,
                             title=f"{model_lbl} vs {BENCHMARK_LABEL}"),
    use_container_width=True,
)

# statistics comparison table
st.subheader("Performance Statistics")
bench_stats = summary_statistics(
    bench_aligned,
    rf_returns=rf_rets.reindex(bench_aligned.index),
    label=BENCHMARK_LABEL,
)

compare_metrics = [
    "Total Return", "Ann. Return", "Ann. Volatility",
    "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
    "Beta", "Alpha (ann.)", "Information Ratio",
]

rows = []
for m in compare_metrics:
    port_val  = stats.get(m, float("nan"))
    bench_val = bench_stats.get(m, float("nan"))
    if m in ["Total Return","Ann. Return","Ann. Volatility","Max Drawdown","Alpha (ann.)"]:
        fmt = fmt_pct
    else:
        fmt = fmt_ratio
    rows.append({
        "Metric": m,
        model_lbl: fmt(port_val),
        BENCHMARK_LABEL: fmt(bench_val) if not (bench_val != bench_val) else "N/A",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# monthly heatmap
st.subheader("Monthly Returns Heatmap")
monthly_tbl = monthly_returns_table(port_rets)
st.plotly_chart(
    monthly_returns_heatmap(monthly_tbl),
    use_container_width=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4: Drawdown analysis
# ---------------------------------------------------------------------------
st.markdown("## 4. Drawdown Analysis")

from analytics.returns import drawdown_series, drawdown_duration
dd_series = drawdown_series(port_rets)
st.plotly_chart(drawdown_chart(dd_series, title="Portfolio Drawdown History"),
                use_container_width=True)

dd_table = drawdown_duration(port_rets)
if not dd_table.empty:
    dd_display = dd_table.copy()
    dd_display["Depth"] = dd_display["Depth"].map(fmt_pct)
    for col in ["Start","Trough","Recovery"]:
        dd_display[col] = dd_display[col].apply(
            lambda d: d.strftime("%d %b %Y") if pd.notna(d) else "Ongoing"
        )
    top_dd = dd_display.sort_values("Depth").head(5)
    st.subheader("Top 5 Drawdown Episodes")
    st.dataframe(top_dd, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 5: Risk
# ---------------------------------------------------------------------------
st.markdown("## 5. Risk Management")

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"VaR 1D ({int(VAR_CONFIDENCE*100)}%)", fmt_pct(var_1d))
col2.metric(f"CVaR 1D ({int(VAR_CONFIDENCE*100)}%)", fmt_pct(cvar_1d))
col3.metric("Ann. Volatility", fmt_pct(ann_vol))
col4.metric("Max Drawdown", fmt_pct(max_dd))

st.subheader("Risk Contribution by Asset")
st.plotly_chart(
    risk_contribution_bar(risk_df, title="Risk vs Weight"),
    use_container_width=True,
)

st.subheader("Stress Test Scenarios")
st.plotly_chart(
    stress_test_bar(stress_df, title="Stress Test Results"),
    use_container_width=True,
)

from analytics.risk import correlation_matrix
corr = correlation_matrix(simple_rets)
st.subheader("Asset Correlation Matrix")
st.plotly_chart(
    correlation_heatmap(corr, title="Pairwise Correlations"),
    use_container_width=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 6: Governance & Model Risk
# ---------------------------------------------------------------------------
st.markdown("## 6. Governance & Model Risk")

render_governance_notice()

st.markdown("""
### Model Assumptions

| Assumption | Detail |
|-----------|--------|
| **Return distribution** | Historical simulation assumes past returns represent the future distribution. Non-stationary in practice. |
| **Transaction costs** | Zero transaction costs assumed. Real portfolios incur spread, commission, and market impact. |
| **Rebalancing** | Monthly rebalancing at month-end closing prices. Real execution differs. |
| **Liquidity** | All ETFs assumed fully liquid. In reality, size and market conditions affect execution. |
| **Covariance stability** | Optimisers assume the covariance matrix is stable. Correlations are known to break down in crises. |
| **VaR scaling** | Multi-day VaR uses √T scaling, which assumes i.i.d. returns. Not valid when returns are autocorrelated. |
| **Return estimation** | Maximum Sharpe uses historical mean returns, which have very poor predictive power. |

### Known Biases

| Bias | Mitigation Applied |
|------|-------------------|
| **Survivorship bias** | ETF proxies reduce (not eliminate) this. Defunct funds are not in the analysis. |
| **Look-ahead bias** | Analytics use only data available at each calculation date. Rolling metrics use trailing windows. |
| **Overfitting** | In-sample optimised portfolios will appear better than out-of-sample performance. |
| **Data vendor risk** | Yahoo Finance data may contain errors or gaps. No independent data verification performed. |

### AI-Specific Controls

- All AI commentary is grounded in provided portfolio data only
- The AI is explicitly instructed not to predict future returns or invent figures
- All AI output is labelled as AI-generated and subject to human review
- AI commentary is informational only and does not constitute investment advice

### Required Human Oversight

This system is designed to support, not replace, human investment judgement. Before any
decision based on Atlas PM analysis:
- Verify data quality and completeness independently
- Review all model assumptions for appropriateness
- Apply qualitative judgement about market conditions not captured by historical data
- Consult a qualified investment professional for any live investment decision
""")

st.markdown("---")

# ---------------------------------------------------------------------------
# PDF Export
# ---------------------------------------------------------------------------
st.subheader("Export Report")

if pdf_available():
    ai_commentary_for_pdf = None  # optionally capture from session if generated

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF…"):
            try:
                bench_stats_for_pdf = summary_statistics(
                    bench_aligned,
                    rf_returns=rf_rets.reindex(bench_aligned.index),
                    label=BENCHMARK_LABEL,
                )
                pdf_bytes = generate_pdf_report(
                    model_name=model_lbl,
                    period=period_str,
                    stats=stats,
                    weights=weights,
                    bench_stats=bench_stats_for_pdf,
                    stress_df=stress_df,
                    risk_df=risk_df,
                    var_1d=var_1d,
                    cvar_1d=cvar_1d,
                    ann_vol=ann_vol,
                    max_dd=max_dd,
                    confidence=VAR_CONFIDENCE,
                    portfolio_value=1_000_000,
                    ai_commentary=ai_commentary_for_pdf,
                )
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=f"atlas_pm_report_{report_date.replace(' ','_')}.pdf",
                    mime="application/pdf",
                )
                st.success("PDF ready. Click Download PDF above.", icon="✅")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
else:
    st.info(
        "PDF export requires `fpdf2`. Install with: `pip install fpdf2`",
        icon="📄"
    )

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:#999; font-size:0.8rem; padding:1rem;">
        {APP_TITLE} | {APP_SUBTITLE} | Report generated {report_date}<br>
        <strong>NOT INVESTMENT ADVICE</strong> — For educational and analytical purposes only.<br>
        Powered by Python · Streamlit · yfinance · Anthropic Claude
    </div>
    """,
    unsafe_allow_html=True,
)
