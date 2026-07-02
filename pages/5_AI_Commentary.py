"""
Page 5 — AI Commentary

AI-powered investment committee-style commentary using Claude.
Includes:
- Performance narrative
- Risk exposure summary
- Allocation rationale
- Bull / Base / Bear scenario analysis
- Q&A against portfolio data
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np

from ai.commentary import (
    generate_performance_commentary,
    generate_risk_commentary,
    generate_allocation_commentary,
    generate_bull_base_bear,
    answer_question,
)
from analytics.risk import (
    historical_var, historical_cvar, run_stress_test,
    component_risk_contribution,
)
from construction.optimiser import compute_cov_matrix
from config.settings import VAR_CONFIDENCE, ASSET_SHORT_NAMES, BENCHMARK_LABEL

st.title("AI Investment Commentary")
st.markdown("Claude-powered analysis grounded in your portfolio's actual data.")
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
stats       = st.session_state.get("portfolio_stats", {})
simple_rets = st.session_state["simple_returns"]

period_str = (
    f"{port_rets.index[0].strftime('%d %b %Y')} — "
    f"{port_rets.index[-1].strftime('%d %b %Y')}"
)

# precompute risk numbers for AI prompts
var_1d  = historical_var(port_rets, VAR_CONFIDENCE, 1)
cvar_1d = historical_cvar(port_rets, VAR_CONFIDENCE, 1)
ann_vol = float(port_rets.std(ddof=1) * np.sqrt(252))
max_dd  = float(((1 + port_rets).cumprod() / (1 + port_rets).cumprod().cummax() - 1).min())

# stress results (top 3 worst)
stress_df = run_stress_test(weights.to_dict())
top_stress = [
    {"name": scenario, "pnl": row["Portfolio P&L"]}
    for scenario, row in stress_df.iterrows()
]

# risk contributions
w_aligned = weights.reindex(simple_rets.columns).dropna()
w_aligned = w_aligned / w_aligned.sum()
cov = compute_cov_matrix(simple_rets[w_aligned.index])
risk_df = component_risk_contribution(w_aligned, cov)
risk_pct = risk_df["% Risk Contribution"].to_dict()

# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    st.warning(
        "**ANTHROPIC_API_KEY not set.** "
        "To enable AI commentary, add your key to `.env` or set it as an environment variable.\n\n"
        "Get your key at [console.anthropic.com](https://console.anthropic.com)",
        icon="🔑"
    )
    st.info(
        "You can still use all other pages without the API key. "
        "AI commentary is the only feature that requires it.",
        icon="ℹ️"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_perf, tab_risk, tab_alloc, tab_bbb, tab_qa = st.tabs([
    "Performance Commentary",
    "Risk Commentary",
    "Allocation Rationale",
    "Bull / Base / Bear",
    "Ask a Question",
])

# ---------------------------------------------------------------------------
# Tab 1: Performance
# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader("Investment Committee — Performance Commentary")
    st.markdown(f"**Portfolio:** {model_lbl} | **Period:** {period_str}")

    if st.button("Generate Performance Commentary", type="primary"):
        with st.spinner("Claude is analysing your portfolio performance…"):
            commentary = generate_performance_commentary(
                stats=stats,
                benchmark_label=BENCHMARK_LABEL,
                period=period_str,
                model_name=model_lbl,
            )
        st.markdown("---")
        st.markdown(commentary)
    else:
        st.info(
            "Click **Generate Performance Commentary** to produce an IC-style performance narrative "
            "based on your portfolio's actual statistics.",
            icon="📊"
        )

# ---------------------------------------------------------------------------
# Tab 2: Risk
# ---------------------------------------------------------------------------
with tab_risk:
    st.subheader("Investment Committee — Risk Commentary")

    if st.button("Generate Risk Commentary", type="primary"):
        with st.spinner("Claude is analysing your risk exposures…"):
            commentary = generate_risk_commentary(
                var_pct=var_1d,
                cvar_pct=cvar_1d,
                ann_vol=ann_vol,
                max_dd=max_dd,
                stress_results={s["name"]: s["pnl"] for s in top_stress},
                risk_contributions=risk_pct,
                confidence=VAR_CONFIDENCE,
            )
        st.markdown("---")
        st.markdown(commentary)
    else:
        st.info(
            "Generates a risk-focused commentary covering VaR, CVaR, stress scenarios, "
            "and risk concentration.",
            icon="⚠️"
        )

# ---------------------------------------------------------------------------
# Tab 3: Allocation rationale
# ---------------------------------------------------------------------------
with tab_alloc:
    st.subheader("Allocation Rationale")

    rf_display = st.session_state.get("current_rf_rate", 0.04)

    if st.button("Generate Allocation Commentary", type="primary"):
        with st.spinner("Claude is explaining the allocation decisions…"):
            commentary = generate_allocation_commentary(
                weights=weights.to_dict(),
                model_name=model_lbl,
                rf_rate=rf_display,
            )
        st.markdown("---")
        st.markdown(commentary)
    else:
        st.info(
            "Explains what the construction model is trying to achieve and how the resulting "
            "weights reflect that objective, including honest discussion of limitations.",
            icon="🔬"
        )

# ---------------------------------------------------------------------------
# Tab 4: Bull / Base / Bear
# ---------------------------------------------------------------------------
with tab_bbb:
    st.subheader("Scenario Analysis: Bull / Base / Bear")
    st.markdown("""
Three qualitative scenarios based on the portfolio's actual characteristics.
These are **not return forecasts** — they are analytical frameworks for
understanding how this portfolio is positioned across different macro environments.
    """)

    if st.button("Generate Scenario Analysis", type="primary"):
        with st.spinner("Claude is building scenario frameworks…"):
            commentary = generate_bull_base_bear(
                portfolio_name=model_lbl,
                ann_return=stats.get("Ann. Return", 0),
                ann_vol=ann_vol,
                max_dd=max_dd,
                weights=weights.to_dict(),
            )
        st.markdown("---")
        st.markdown(commentary)
    else:
        st.info(
            "Produces qualitative Bull / Base / Bear scenario analysis grounded in "
            "the portfolio's actual allocation and historical risk profile.",
            icon="🐂"
        )

# ---------------------------------------------------------------------------
# Tab 5: Q&A
# ---------------------------------------------------------------------------
with tab_qa:
    st.subheader("Portfolio Q&A")
    st.markdown("""
Ask any question about your portfolio. Claude will answer using **only the data from your analysis**.
It will not invent numbers or make predictions.

**Example questions:**
- "What is driving the maximum drawdown of this portfolio?"
- "Why does the risk parity model allocate so little to equities?"
- "How does the Sharpe ratio compare to the benchmark?"
- "What is the biggest risk concentration in this portfolio?"
    """)

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the worst stress scenario for this portfolio?",
    )

    if st.button("Ask Claude", type="primary", key="qa_submit"):
        if not question:
            st.warning("Please enter a question.")
        else:
            portfolio_context = {
                "model": model_lbl,
                "period": period_str,
                "weights": weights.to_dict(),
                "statistics": {k: v for k, v in stats.items() if k != "Label"},
                "risk": {
                    "1d_var_95pct": var_1d,
                    "1d_cvar_95pct": cvar_1d,
                    "annualised_vol": ann_vol,
                    "max_drawdown": max_dd,
                },
                "stress_tests": {s["name"]: s["pnl"] for s in top_stress},
                "risk_contributions_pct": risk_pct,
                "benchmark": BENCHMARK_LABEL,
            }

            with st.spinner("Claude is thinking…"):
                answer = answer_question(question, portfolio_context)

            st.markdown("---")
            st.markdown(f"**Q: {question}**")
            st.markdown(answer)

# ---------------------------------------------------------------------------
# Disclaimer footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ All AI commentary is generated by Claude (Anthropic) using historical portfolio data. "
    "It does not constitute investment advice, financial advice, or a recommendation to buy or sell "
    "any security. Always consult a qualified financial adviser before making investment decisions."
)
