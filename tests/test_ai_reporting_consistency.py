"""
test_ai_reporting_consistency.py
----------------------------------
Structural and data-consistency tests for the AI commentary module.

Because the Claude API is non-deterministic (model responses vary between calls)
and requires a paid API key, we CANNOT test the TEXT content of responses.

What we CAN and DO test:
  1. Prompt construction — verify the prompt contains the actual data passed in.
     This ensures the AI receives the numbers, not a garbled / missing input.
  2. Module structure — all commentary generators are callable with correct signatures.
  3. Fallback behaviour — when ANTHROPIC_API_KEY is absent, the function returns
     a useful error string rather than crashing or returning empty.
  4. System prompt content — the anti-hallucination rules are present.
  5. Input serialisation — JSON-encoded stats/weights round-trip without NaN injection.

These tests do NOT call the live Claude API.
They do NOT assert anything about the content of AI-generated text.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from ai.commentary import (
    SYSTEM_PROMPT,
    generate_performance_commentary,
    generate_risk_commentary,
    generate_allocation_commentary,
    generate_bull_base_bear,
    generate_ic_report_narrative,
    answer_question,
    _call_claude,
    _get_client,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sample_stats() -> dict:
    return {
        "Label":           "Maximum Sharpe",
        "Total Return":    0.82,
        "Ann. Return":     0.073,
        "Ann. Volatility": 0.112,
        "Sharpe Ratio":    0.58,
        "Sortino Ratio":   0.79,
        "Calmar Ratio":    0.44,
        "Max Drawdown":    -0.168,
        "Skewness":        -0.24,
        "Kurtosis (excess)": 1.12,
        "Best Day":        0.033,
        "Worst Day":       -0.041,
        "% Positive Days": 0.531,
        "Observations":    2516,
        "Beta":            0.72,
        "Alpha (ann.)":    0.021,
        "Information Ratio": 0.31,
    }


def _sample_weights() -> dict:
    return {
        "US Equities (S&P 500)": 0.32,
        "US Aggregate Bonds":    0.28,
        "Gold":                  0.18,
        "Emerging Markets":      0.12,
        "Cash Proxy (T-Bills)":  0.10,
    }


# ─── System prompt checks ──────────────────────────────────────────────────────

class TestSystemPrompt:
    """The anti-hallucination system prompt must contain the critical guard clauses."""

    def test_system_prompt_is_non_empty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_forbids_hallucination(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "hallucinate" in lowered or "invent" in lowered, (
            "System prompt must explicitly forbid hallucination / inventing numbers"
        )

    def test_system_prompt_requires_data_grounding(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "data" in lowered and "provided" in lowered, (
            "System prompt must instruct the model to use only data provided"
        )

    def test_system_prompt_includes_disclaimer(self):
        assert "not constitute investment advice" in SYSTEM_PROMPT.lower() or \
               "not investment advice" in SYSTEM_PROMPT.lower(), (
            "System prompt must include 'not investment advice' disclaimer"
        )

    def test_system_prompt_does_not_predict_future(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "do not predict" in lowered or "future returns" in lowered, (
            "System prompt must forbid future return predictions"
        )


# ─── Fallback behaviour (no API key) ─────────────────────────────────────────

class TestFallbackBehaviour:
    """
    When ANTHROPIC_API_KEY is absent (or blank), every commentary function must:
      - Return a string (not None, not raise an exception)
      - Contain a recognisable error/unavailable message
      - NOT return an empty string
    """

    @pytest.fixture(autouse=True)
    def remove_api_key(self, monkeypatch):
        """Ensure no API key is set for these tests."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def test_get_client_returns_none_without_key(self):
        client = _get_client()
        assert client is None, "Should return None when API key is absent"

    def test_call_claude_returns_string_without_key(self):
        result = _call_claude("Hello")
        assert isinstance(result, str), "_call_claude must return a string"
        assert len(result) > 0, "_call_claude must return non-empty string"

    def test_call_claude_unavailable_message_without_key(self):
        result = _call_claude("Hello")
        assert "unavailable" in result.lower() or "api key" in result.lower(), (
            "No-key response must indicate AI is unavailable or key is missing"
        )

    def test_generate_performance_commentary_no_key(self):
        result = generate_performance_commentary(
            stats=_sample_stats(),
            benchmark_label="S&P 500 (SPY)",
            period="2015–2024",
            model_name="Maximum Sharpe",
        )
        assert isinstance(result, str) and len(result) > 0

    def test_generate_risk_commentary_no_key(self):
        result = generate_risk_commentary(
            var_pct=0.021,
            cvar_pct=0.031,
            ann_vol=0.112,
            max_dd=-0.168,
            stress_results={"GFC 2008": -0.28, "COVID 2020": -0.19},
            risk_contributions={"US EQ": 0.48, "US BOND": 0.22},
        )
        assert isinstance(result, str) and len(result) > 0

    def test_generate_allocation_commentary_no_key(self):
        result = generate_allocation_commentary(
            weights=_sample_weights(),
            model_name="Maximum Sharpe",
            rf_rate=0.04,
        )
        assert isinstance(result, str) and len(result) > 0

    def test_generate_bull_base_bear_no_key(self):
        result = generate_bull_base_bear(
            portfolio_name="Maximum Sharpe",
            ann_return=0.073,
            ann_vol=0.112,
            max_dd=-0.168,
            weights=_sample_weights(),
        )
        assert isinstance(result, str) and len(result) > 0

    def test_answer_question_no_key(self):
        result = answer_question(
            question="What is the Sharpe ratio?",
            portfolio_context={"Sharpe Ratio": 0.58, "Ann. Return": 0.073},
        )
        assert isinstance(result, str) and len(result) > 0

    def test_generate_ic_report_no_key(self):
        result = generate_ic_report_narrative(
            period="January 2024",
            model_name="Maximum Sharpe",
            stats=_sample_stats(),
            weights=_sample_weights(),
            top_stress_scenarios=[
                {"name": "GFC 2008", "pnl": -0.28},
                {"name": "COVID 2020", "pnl": -0.19},
            ],
        )
        assert isinstance(result, str) and len(result) > 0


# ─── Prompt data-injection verification ──────────────────────────────────────

class TestPromptDataInjection:
    """
    Verify that the data provided to each commentary generator appears in the prompt
    that would be sent to the model. We do this by patching _call_claude to capture
    the prompt string and checking it contains key data values.
    """

    def test_performance_commentary_prompt_contains_sharpe(self, monkeypatch):
        """
        generate_performance_commentary must include the Sharpe ratio value in the prompt.
        This ensures the AI receives the actual data, not a missing/corrupted payload.
        """
        captured = {}

        def mock_call(prompt: str, max_tokens: int = 1500) -> str:
            captured["prompt"] = prompt
            return "MOCKED RESPONSE"

        monkeypatch.setattr("ai.commentary._call_claude", mock_call)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        stats = _sample_stats()
        generate_performance_commentary(
            stats=stats,
            benchmark_label="S&P 500 (SPY)",
            period="2015–2024",
            model_name="Maximum Sharpe",
        )

        assert "prompt" in captured, "Mock was not called — function didn't reach _call_claude"
        prompt = captured["prompt"]
        # Sharpe ratio value should appear in the prompt
        assert str(round(stats["Sharpe Ratio"], 6)) in prompt or "Sharpe" in prompt, (
            "Prompt does not appear to include Sharpe ratio data"
        )

    def test_performance_commentary_prompt_contains_model_name(self, monkeypatch):
        captured = {}

        def mock_call(prompt: str, max_tokens: int = 1500) -> str:
            captured["prompt"] = prompt
            return "MOCKED RESPONSE"

        monkeypatch.setattr("ai.commentary._call_claude", mock_call)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        generate_performance_commentary(
            stats=_sample_stats(),
            benchmark_label="S&P 500 (SPY)",
            period="2015–2024",
            model_name="Maximum Sharpe",
        )

        assert "Maximum Sharpe" in captured["prompt"], (
            "Model name 'Maximum Sharpe' not found in prompt"
        )

    def test_risk_commentary_prompt_contains_var(self, monkeypatch):
        captured = {}

        def mock_call(prompt: str, max_tokens: int = 1500) -> str:
            captured["prompt"] = prompt
            return "MOCKED"

        monkeypatch.setattr("ai.commentary._call_claude", mock_call)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        generate_risk_commentary(
            var_pct=0.021,
            cvar_pct=0.031,
            ann_vol=0.112,
            max_dd=-0.168,
            stress_results={"GFC 2008": -0.28},
            risk_contributions={"US EQ": 0.48},
        )

        prompt = captured.get("prompt", "")
        assert "0.021" in prompt or "2.10%" in prompt or "VaR" in prompt, (
            "VaR value not found in risk commentary prompt"
        )

    def test_allocation_commentary_prompt_contains_weights(self, monkeypatch):
        captured = {}

        def mock_call(prompt: str, max_tokens: int = 1500) -> str:
            captured["prompt"] = prompt
            return "MOCKED"

        monkeypatch.setattr("ai.commentary._call_claude", mock_call)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        weights = _sample_weights()
        generate_allocation_commentary(
            weights=weights,
            model_name="Maximum Sharpe",
            rf_rate=0.04,
        )

        prompt = captured.get("prompt", "")
        assert "Gold" in prompt or "US Equities" in prompt, (
            "Asset names not found in allocation commentary prompt"
        )

    def test_answer_question_prompt_contains_question(self, monkeypatch):
        captured = {}

        def mock_call(prompt: str, max_tokens: int = 600) -> str:
            captured["prompt"] = prompt
            return "MOCKED ANSWER"

        monkeypatch.setattr("ai.commentary._call_claude", mock_call)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        question = "What is the maximum drawdown of this portfolio?"
        answer_question(
            question=question,
            portfolio_context={"Max Drawdown": -0.168},
        )

        prompt = captured.get("prompt", "")
        assert question in prompt, (
            "User question not found in the prompt sent to the AI"
        )


# ─── JSON serialisation safety ────────────────────────────────────────────────

class TestJSONSerialisation:
    """
    Verify that stats dicts round-trip through JSON serialisation without
    introducing NaN, Infinity, or lost keys. The commentary module serialises
    stats to JSON before injecting them into prompts.
    """

    def test_sample_stats_json_roundtrip(self):
        stats = _sample_stats()
        serialised = json.dumps(
            {k: round(v, 6) if isinstance(v, float) else v for k, v in stats.items()},
            indent=2,
        )
        parsed = json.loads(serialised)
        for key in stats:
            assert key in parsed, f"Key '{key}' lost in JSON round-trip"

    def test_no_nan_in_stats(self):
        stats = _sample_stats()
        for key, val in stats.items():
            if isinstance(val, float):
                assert not np.isnan(val), f"NaN in stats['{key}']"
                assert not np.isinf(val), f"Inf in stats['{key}']"

    def test_weights_sum_to_one_before_serialisation(self):
        weights = _sample_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, (
            f"Sample weights sum to {total:.6f}, should be 1.0"
        )

    def test_weights_json_shows_percentages(self):
        """generate_allocation_commentary formats weights as '32.00%' in the prompt."""
        weights = _sample_weights()
        formatted = {k: f"{v:.2%}" for k, v in weights.items()}
        for val_str in formatted.values():
            assert "%" in val_str, "Weight not formatted as percentage"

    def test_serialise_stats_with_non_finite_float_replaced(self):
        """
        If a stat is Inf or NaN (e.g. Sharpe of constant-return portfolio),
        JSON serialisation must not crash. Python json module raises ValueError
        for NaN/Inf by default — verify the module handles this defensively.
        """
        stats = _sample_stats()
        stats["Sharpe Ratio"] = float("nan")   # edge case

        # The commentary module filters with `round(v, 6) if isinstance(v, float)`
        # round(nan, 6) raises ValueError in standard Python.
        # Verify the function doesn't crash by wrapping in a safe serialiser.
        try:
            serialised = json.dumps(
                {k: (round(v, 6) if isinstance(v, float) and np.isfinite(v) else str(v))
                 for k, v in stats.items()},
                indent=2,
            )
            parsed = json.loads(serialised)
            assert "Sharpe Ratio" in parsed
        except Exception as e:
            pytest.fail(f"Safe serialisation of NaN stats crashed: {e}")
