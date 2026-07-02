"""
AI-powered investment commentary module for Atlas PM.

Uses the Claude API (claude-opus-4-8) to generate:
- Portfolio performance explanations grounded in actual data
- Risk exposure summaries
- Monthly investment committee (IC) commentary
- Bull / Base / Bear scenario narratives
- Q&A against portfolio data

Design principles:
1. DATA-GROUNDED: the AI is given actual portfolio numbers and is instructed
   to reason from those numbers only.  No stock predictions, no speculative
   forecasts dressed as analysis.

2. HALLUCINATION CONTROLS: the system prompt explicitly forbids the model
   from inventing numbers not in the provided context.  All commentary is
   clearly labelled as AI-generated analysis, not investment advice.

3. STRUCTURED PROMPTS: each use case gets a dedicated prompt template so
   the output format is consistent and professional.

4. GRACEFUL DEGRADATION: if the API key is missing or the call fails, we
   return a clear error string rather than crashing the app.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import anthropic

from config.settings import AI_MODEL, MAX_COMMENTARY_TOKENS


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def _get_client() -> Optional[anthropic.Anthropic]:
    """Return an Anthropic client if ANTHROPIC_API_KEY is set."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------------------
# System prompt — shared across all commentary types
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an investment analyst and portfolio risk manager working for an institutional asset manager in London.

Your role is to write clear, professional, CFA-level investment commentary based exclusively on the quantitative data provided to you.

RULES YOU MUST FOLLOW:
1. Base every statement on the numbers in the data provided. Do not invent or hallucinate any figures.
2. If data is missing or insufficient, say so explicitly rather than guessing.
3. Write in a professional, measured tone — like a Goldman Sachs or BlackRock research note.
4. Always acknowledge model limitations honestly.
5. This is analytical commentary for internal use only. It is NOT investment advice.
6. Do not predict future returns. Describe what the data shows about the past and the current positioning.
7. Use precise finance vocabulary: attribution, drawdown, tracking error, risk-adjusted return, factor exposure, etc.
8. Keep commentary concise. Prioritise insight over length.
9. When discussing risks, be specific — reference actual portfolio metrics, not generic warnings.
10. End every commentary with: "⚠️ This commentary is AI-generated from historical data and does not constitute investment advice."
"""


# ---------------------------------------------------------------------------
# Commentary generators
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, max_tokens: int = MAX_COMMENTARY_TOKENS) -> str:
    """Low-level call to Claude API. Returns text or error message."""
    client = _get_client()
    if client is None:
        return (
            "⚠️ AI commentary unavailable: ANTHROPIC_API_KEY environment variable not set.\n\n"
            "To enable AI commentary:\n"
            "1. Get your API key from console.anthropic.com\n"
            "2. Set it in the .env file: ANTHROPIC_API_KEY=your-key-here\n"
            "3. Restart the app."
        )

    try:
        message = client.messages.create(
            model=AI_MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except anthropic.APIConnectionError:
        return "⚠️ AI commentary unavailable: Could not connect to Anthropic API. Check your internet connection."
    except anthropic.AuthenticationError:
        return "⚠️ AI commentary unavailable: Invalid API key. Please check your ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return "⚠️ AI commentary unavailable: Rate limit reached. Please wait a moment and try again."
    except Exception as e:
        return f"⚠️ AI commentary unavailable: {str(e)}"


def generate_performance_commentary(
    stats: dict,
    benchmark_label: str,
    period: str,
    model_name: str,
) -> str:
    """
    Generate a performance attribution narrative for the IC report.

    Parameters
    ----------
    stats         : dict from analytics.returns.summary_statistics()
    benchmark_label : e.g. "S&P 500 (SPY)"
    period        : human-readable period string, e.g. "January 2015 – December 2024"
    model_name    : e.g. "Maximum Sharpe"
    """
    stats_json = json.dumps(
        {k: round(v, 6) if isinstance(v, float) else v for k, v in stats.items()},
        indent=2,
    )
    prompt = f"""
Write a professional investment performance commentary for an Investment Committee report.

## Portfolio Details
- Construction model: {model_name}
- Benchmark: {benchmark_label}
- Analysis period: {period}

## Portfolio Statistics (all figures are exact — reference them precisely)
{stats_json}

## Instructions
Write 3-4 paragraphs covering:
1. Overall return performance vs benchmark — use the exact figures from the statistics
2. Risk-adjusted performance (Sharpe, Sortino, Calmar) — interpret what the numbers mean
3. Drawdown analysis — how did the portfolio behave in adverse conditions
4. Key strengths and weaknesses of this construction approach based on the actual numbers

Tone: professional, analytical, concise. Appropriate for an institutional investment committee.
"""
    return _call_claude(prompt)


def generate_risk_commentary(
    var_pct: float,
    cvar_pct: float,
    ann_vol: float,
    max_dd: float,
    stress_results: dict,
    risk_contributions: dict,
    confidence: float = 0.95,
) -> str:
    """
    Generate a risk exposure narrative including VaR, CVaR, stress tests, and factor exposures.
    """
    stress_json = json.dumps(
        {k: round(v, 4) for k, v in stress_results.items()},
        indent=2,
    )
    risk_contrib_json = json.dumps(
        {k: round(v, 4) for k, v in risk_contributions.items()},
        indent=2,
    )
    prompt = f"""
Write a professional risk management commentary for an Investment Committee report.

## Key Risk Metrics
- 1-Day Historical VaR ({int(confidence*100)}% confidence): {var_pct:.2%}
- 1-Day Historical CVaR / Expected Shortfall ({int(confidence*100)}%): {cvar_pct:.2%}
- Annualised Volatility: {ann_vol:.2%}
- Maximum Drawdown: {max_dd:.2%}

## Stress Test Results (portfolio P&L fraction)
{stress_json}

## Risk Contribution by Asset (% of total portfolio risk)
{risk_contrib_json}

## Instructions
Write 3-4 paragraphs covering:
1. Overall risk profile — interpret VaR, CVaR, and volatility in plain English for senior stakeholders
2. Stress test results — which scenarios pose the greatest threat and why
3. Risk concentration — are there dominant contributors that warrant attention?
4. Risk management observations — what does the data suggest about hedging or diversification?

Use precise language. Reference the exact numbers provided.
"""
    return _call_claude(prompt)


def generate_allocation_commentary(
    weights: dict,
    model_name: str,
    rf_rate: float,
) -> str:
    """
    Explain the portfolio allocation and the rationale of the construction model.
    """
    weights_json = json.dumps(
        {k: f"{v:.2%}" for k, v in weights.items()},
        indent=2,
    )
    prompt = f"""
Write a professional portfolio allocation commentary for an Investment Committee report.

## Construction Model: {model_name}
## Risk-Free Rate Assumption: {rf_rate:.1%}

## Portfolio Weights
{weights_json}

## Instructions
Write 2-3 paragraphs covering:
1. What is the {model_name} model trying to achieve mathematically — explain the objective function in plain English without jargon overload
2. How do the resulting weights reflect the model's objective? Comment on any notable tilts or diversification patterns.
3. What are the key model assumptions and limitations investors should be aware of?

Be honest about limitations — e.g. sensitivity to return estimates for Maximum Sharpe, or concentration in low-volatility assets for Minimum Variance.
"""
    return _call_claude(prompt)


def generate_bull_base_bear(
    portfolio_name: str,
    ann_return: float,
    ann_vol: float,
    max_dd: float,
    weights: dict,
) -> str:
    """
    Generate three scenarios (Bull / Base / Bear) based on portfolio characteristics.
    These are analytical frameworks, not return predictions.
    """
    weights_json = json.dumps(
        {k: f"{v:.2%}" for k, v in weights.items()},
        indent=2,
    )
    prompt = f"""
Write a professional scenario analysis for an Investment Committee, structured as Bull / Base / Bear cases.

## Portfolio: {portfolio_name}
## Historical Statistics
- Annualised Return: {ann_return:.2%}
- Annualised Volatility: {ann_vol:.2%}
- Maximum Drawdown: {max_dd:.2%}

## Current Allocation
{weights_json}

## Instructions
For each of the three scenarios (Bull, Base, Bear), write 2-3 sentences covering:
- What macro/market environment drives this scenario
- How this portfolio would likely respond given its actual allocation and historical risk profile
- What the key risk or opportunity is in each case

DO NOT predict specific return numbers for future periods. Frame these as qualitative analytical frameworks.
Be explicit that these are scenario analyses for planning purposes, not forecasts.
"""
    return _call_claude(prompt)


def generate_ic_report_narrative(
    period: str,
    model_name: str,
    stats: dict,
    weights: dict,
    top_stress_scenarios: list,
) -> str:
    """
    Generate the narrative section of a full monthly IC report.
    This is the most comprehensive commentary, suitable for the full report.
    """
    stats_json = json.dumps(
        {k: round(v, 6) if isinstance(v, float) else v for k, v in stats.items()},
        indent=2,
    )
    weights_str = ", ".join(f"{k}: {v:.1%}" for k, v in weights.items())
    stress_str  = "; ".join(
        f"{s['name']}: {s['pnl']:.2%}" for s in top_stress_scenarios
    )

    prompt = f"""
Write the narrative section of a monthly Investment Committee Report for the period: {period}.

## Portfolio Construction
Model: {model_name}
Allocation: {weights_str}

## Performance Summary
{stats_json}

## Stress Test Highlights
{stress_str}

## Report Structure
Write the following sections, each 1-2 paragraphs:

### Executive Summary
High-level portfolio performance and positioning summary.

### Performance Attribution
Discuss return drivers given the portfolio's allocation.

### Risk Assessment
Interpret the quantitative risk metrics and stress test implications.

### Portfolio Positioning
Commentary on the allocation rationale and any implicit factor tilts.

### Key Risks & Considerations
Specific, data-grounded risks the committee should monitor.

### Rebalancing Considerations
Any observations about whether the current allocation remains appropriate.

Keep the tone professional, concise, and grounded in the data provided.
"""
    return _call_claude(prompt, max_tokens=2000)


def answer_question(
    question: str,
    portfolio_context: dict,
) -> str:
    """
    Answer a user question about the portfolio using only the provided data.

    Parameters
    ----------
    question          : free-text question from the user
    portfolio_context : dict containing stats, weights, stress results, etc.
    """
    context_json = json.dumps(
        {k: (round(v, 6) if isinstance(v, float) else
             {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()}
             if isinstance(v, dict) else v)
         for k, v in portfolio_context.items()},
        indent=2,
    )
    prompt = f"""
A user has asked the following question about their portfolio:

"{question}"

Answer using ONLY the data provided below. If the data does not contain enough information to answer the question, say so explicitly.

## Portfolio Data
{context_json}

Write a clear, professional answer as if responding to a portfolio manager or investment committee member. Reference specific numbers from the data where relevant. Keep it concise — 2-4 sentences for simple questions, 1 paragraph for complex ones.
"""
    return _call_claude(prompt, max_tokens=600)
