"""
Tests for analytics.investor_utility — mean-variance utility functions.

Test coverage:
 1.  Utility increases when expected return increases (volatility fixed)
 2.  Utility decreases when volatility increases (return fixed)
 3.  Higher risk aversion penalises volatility more
 4.  Indifference curve returns increase as volatility increases
 5.  Higher risk aversion creates a steeper indifference curve
 6.  Utility-optimal portfolio is correctly selected from synthetic frontier data
 7.  Utility-optimal portfolio lies on one of the efficient frontier points
 8.  Custom risk aversion values work
 9.  Invalid risk aversion values (≤0) raise ValueError
10.  Chart data includes the utility-optimal point and indifference curve
11.  Formula correctness — cross-check utility and indifference curve
12.  Investor profile → risk-aversion mapping
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analytics.investor_utility import (
    calculate_mean_variance_utility,
    calculate_indifference_curve,
    find_utility_optimal_portfolio,
    map_profile_to_risk_aversion,
    PROFILE_RISK_AVERSION,
    PROFILE_NAMES,
)


# ---------------------------------------------------------------------------
# Shared synthetic frontier
# ---------------------------------------------------------------------------

def make_synthetic_frontier(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Linearly increasing return with linearly increasing vol (simple test frontier)."""
    vols = np.linspace(0.05, 0.25, n)
    rets = np.linspace(0.04, 0.14, n)   # monotonically increasing
    return rets, vols


RETS, VOLS = make_synthetic_frontier()


# ---------------------------------------------------------------------------
# 1. Utility increases with expected return (volatility fixed)
# ---------------------------------------------------------------------------

def test_utility_increases_with_expected_return():
    vol = 0.15
    ra  = 4.0
    u_low  = calculate_mean_variance_utility(0.06, vol, ra)
    u_high = calculate_mean_variance_utility(0.10, vol, ra)
    assert float(u_high) > float(u_low), (
        "Utility must increase when expected return increases, holding vol constant"
    )


# ---------------------------------------------------------------------------
# 2. Utility decreases when volatility increases (return fixed)
# ---------------------------------------------------------------------------

def test_utility_decreases_with_volatility():
    ret = 0.08
    ra  = 4.0
    u_low_vol  = calculate_mean_variance_utility(ret, 0.05, ra)
    u_high_vol = calculate_mean_variance_utility(ret, 0.25, ra)
    assert float(u_low_vol) > float(u_high_vol), (
        "Utility must decrease when volatility increases, holding expected return constant"
    )


# ---------------------------------------------------------------------------
# 3. Higher risk aversion penalises volatility more
# ---------------------------------------------------------------------------

def test_higher_risk_aversion_penalises_volatility_more():
    ret = 0.10
    vol = 0.20
    u_low_ra  = calculate_mean_variance_utility(ret, vol, risk_aversion=2.0)
    u_high_ra = calculate_mean_variance_utility(ret, vol, risk_aversion=8.0)
    assert float(u_low_ra) > float(u_high_ra), (
        "Higher risk aversion must produce lower utility for the same (return, vol) pair"
    )
    # Confirm the penalty term 0.5*A*sigma^2 is strictly larger for higher A
    penalty_low  = 0.5 * 2.0 * vol ** 2
    penalty_high = 0.5 * 8.0 * vol ** 2
    assert penalty_high > penalty_low


# ---------------------------------------------------------------------------
# 4. Indifference curve is strictly upward sloping
# ---------------------------------------------------------------------------

def test_indifference_curve_is_upward_sloping():
    vols = np.linspace(0.05, 0.30, 50)
    rets = calculate_indifference_curve(vols, utility_level=0.02, risk_aversion=4.0)
    diffs = np.diff(rets)
    assert (diffs > 0).all(), (
        "Indifference curve must be strictly upward sloping (returns increase with vol)"
    )


# ---------------------------------------------------------------------------
# 5. Higher risk aversion creates a steeper curve
# ---------------------------------------------------------------------------

def test_higher_risk_aversion_steeper_curve():
    vols = np.linspace(0.05, 0.30, 50)
    u_level = 0.02
    rets_low_ra  = calculate_indifference_curve(vols, u_level, risk_aversion=2.0)
    rets_high_ra = calculate_indifference_curve(vols, u_level, risk_aversion=8.0)
    # Slope = d(E(r))/d(sigma) = A * sigma; larger A → steeper at every point
    mid = len(vols) // 2
    slope_low  = (rets_low_ra[mid + 1]  - rets_low_ra[mid])  / (vols[mid + 1] - vols[mid])
    slope_high = (rets_high_ra[mid + 1] - rets_high_ra[mid]) / (vols[mid + 1] - vols[mid])
    assert slope_high > slope_low, (
        "Higher risk aversion must produce a steeper indifference curve"
    )


# ---------------------------------------------------------------------------
# 6. Utility-optimal portfolio is correctly selected
# ---------------------------------------------------------------------------

def test_utility_optimal_correctly_selected():
    """The function must return the index with the maximum utility."""
    rets = np.array([0.05, 0.07, 0.09, 0.08, 0.06])
    vols = np.array([0.10, 0.12, 0.20, 0.16, 0.11])
    ra   = 4.0

    utilities = calculate_mean_variance_utility(rets, vols, ra)
    expected_idx = int(np.argmax(utilities))
    result = find_utility_optimal_portfolio(rets, vols, ra)

    assert result["index"] == expected_idx, (
        f"Expected index {expected_idx} but got {result['index']}"
    )
    assert result["expected_return"] == rets[expected_idx]
    assert result["volatility"]      == vols[expected_idx]
    np.testing.assert_allclose(result["utility"], float(utilities[expected_idx]), rtol=1e-10)


# ---------------------------------------------------------------------------
# 7. Utility-optimal point lies on one of the frontier points (not interpolated)
# ---------------------------------------------------------------------------

def test_utility_optimal_lies_on_frontier():
    """The optimal portfolio must be one of the discrete frontier points."""
    rets, vols = make_synthetic_frontier(n=20)
    result = find_utility_optimal_portfolio(rets, vols, risk_aversion=4.0)
    idx = result["index"]

    assert 0 <= idx < len(rets), "Index out of bounds"
    assert result["expected_return"] == rets[idx], (
        "Returned expected_return must match frontier[index]"
    )
    assert result["volatility"] == vols[idx], (
        "Returned volatility must match frontier[index]"
    )


# ---------------------------------------------------------------------------
# 8. Custom risk aversion values work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ra", [0.5, 1.0, 3.5, 7.0, 10.0])
def test_custom_risk_aversion_values(ra):
    """Any positive risk aversion value must produce finite, valid results."""
    u = calculate_mean_variance_utility(0.08, 0.15, ra)
    assert np.isfinite(float(u)), f"Utility is not finite for ra={ra}"

    curve = calculate_indifference_curve(np.linspace(0.05, 0.25, 10), 0.03, ra)
    assert curve.shape == (10,)
    assert np.isfinite(curve).all()

    result = find_utility_optimal_portfolio(RETS, VOLS, ra)
    assert 0 <= result["index"] < len(RETS)
    assert np.isfinite(result["utility"])


# ---------------------------------------------------------------------------
# 9. Invalid risk aversion values are handled safely
# ---------------------------------------------------------------------------

def test_invalid_risk_aversion_utility_zero():
    with pytest.raises(ValueError, match="positive"):
        calculate_mean_variance_utility(0.08, 0.15, risk_aversion=0.0)


def test_invalid_risk_aversion_utility_negative():
    with pytest.raises(ValueError, match="positive"):
        calculate_mean_variance_utility(0.08, 0.15, risk_aversion=-1.0)


def test_invalid_risk_aversion_curve():
    with pytest.raises(ValueError, match="positive"):
        calculate_indifference_curve(np.linspace(0.05, 0.25, 10), 0.03, risk_aversion=-2.0)


def test_invalid_risk_aversion_optimal_zero():
    with pytest.raises(ValueError, match="positive"):
        find_utility_optimal_portfolio(RETS, VOLS, risk_aversion=0.0)


def test_invalid_risk_aversion_optimal_negative():
    with pytest.raises(ValueError, match="positive"):
        find_utility_optimal_portfolio(RETS, VOLS, risk_aversion=-0.5)


def test_empty_frontier_raises():
    with pytest.raises(ValueError, match="non-empty"):
        find_utility_optimal_portfolio(np.array([]), np.array([]), risk_aversion=4.0)


def test_mismatched_frontier_shapes_raises():
    with pytest.raises(ValueError, match="same shape"):
        find_utility_optimal_portfolio(
            np.array([0.05, 0.08]),
            np.array([0.10, 0.15, 0.20]),
            risk_aversion=4.0,
        )


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="Unknown profile"):
        map_profile_to_risk_aversion("Ultra-Aggressive")


# ---------------------------------------------------------------------------
# 10. Chart data includes the utility-optimal point and indifference curve
# ---------------------------------------------------------------------------

def test_chart_data_includes_utility_optimal_and_curve():
    """
    Simulate the chart-building logic used on the Portfolio Construction page
    and confirm the utility-optimal point and at least one indifference curve
    are added as named traces.
    """
    import plotly.graph_objects as go
    from construction.optimiser import efficient_frontier
    from ui.components.charts import efficient_frontier_chart

    # Build a small synthetic returns dataset
    rng = np.random.default_rng(42)
    n_days, n_assets = 400, 3
    rets_df = pd.DataFrame(
        rng.normal(0.0003, 0.01, (n_days, n_assets)),
        columns=["A", "B", "C"],
        index=pd.date_range("2020-01-01", periods=n_days, freq="B"),
    )

    frontier_df = efficient_frontier(
        rets_df, n_points=20, min_weight=0.0, max_weight=1.0
    )
    assert not frontier_df.empty, "Frontier is empty — cannot proceed with chart test"

    ra = 4.0
    opt = find_utility_optimal_portfolio(
        frontier_df["Ann. Return"].values,
        frontier_df["Ann. Volatility"].values,
        ra,
    )

    # Utility-optimal must be within frontier bounds
    assert (
        frontier_df["Ann. Return"].min() - 1e-6
        <= opt["expected_return"]
        <= frontier_df["Ann. Return"].max() + 1e-6
    ), "Utility-optimal return is outside the frontier range"
    assert (
        frontier_df["Ann. Volatility"].min() - 1e-6
        <= opt["volatility"]
        <= frontier_df["Ann. Volatility"].max() + 1e-6
    ), "Utility-optimal volatility is outside the frontier range"

    # Build figure and add utility-optimal marker
    fig = efficient_frontier_chart(frontier_df, {})
    fig.add_trace(go.Scatter(
        x=[opt["volatility"] * 100],
        y=[opt["expected_return"] * 100],
        mode="markers+text",
        marker=dict(size=14, color="#9b5de5", symbol="diamond"),
        name="Utility-Optimal Portfolio",
    ))

    # Add one indifference curve
    vol_range = np.linspace(
        frontier_df["Ann. Volatility"].min(),
        frontier_df["Ann. Volatility"].max(),
        100,
    )
    curve_rets = calculate_indifference_curve(vol_range, opt["utility"], ra)
    fig.add_trace(go.Scatter(
        x=vol_range * 100,
        y=curve_rets * 100,
        mode="lines",
        name="Utility Curve (U*)",
    ))

    trace_names = [t.name for t in fig.data]
    assert "Utility-Optimal Portfolio" in trace_names, (
        "Utility-optimal point must appear as a named trace"
    )
    assert "Utility Curve (U*)" in trace_names, (
        "Utility indifference curve must appear as a named trace"
    )


# ---------------------------------------------------------------------------
# 11. Formula correctness
# ---------------------------------------------------------------------------

def test_utility_formula_correctness():
    """Exact check: U = E(r) - 0.5 * A * sigma^2."""
    ret, vol, ra = 0.10, 0.15, 4.0
    expected = ret - 0.5 * ra * vol ** 2
    actual = float(calculate_mean_variance_utility(ret, vol, ra))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_indifference_curve_formula_correctness():
    """Exact check: E(r) = U + 0.5 * A * sigma^2."""
    vols = np.array([0.10, 0.15, 0.20])
    u_level, ra = 0.03, 4.0
    expected = u_level + 0.5 * ra * vols ** 2
    actual = calculate_indifference_curve(vols, u_level, ra)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_utility_vectorised_matches_scalar():
    """Vectorised call must match loop of scalar calls."""
    rets = np.array([0.05, 0.08, 0.12])
    vols = np.array([0.10, 0.15, 0.20])
    ra = 3.0

    u_vec = calculate_mean_variance_utility(rets, vols, ra)
    u_scalars = [
        float(calculate_mean_variance_utility(r, v, ra))
        for r, v in zip(rets, vols)
    ]
    np.testing.assert_allclose(u_vec, u_scalars, rtol=1e-12)


# ---------------------------------------------------------------------------
# 12. Investor profile → risk-aversion mapping
# ---------------------------------------------------------------------------

def test_all_profiles_map_correctly():
    """All named profiles must map to their canonical risk-aversion values."""
    expected = {
        "Very Conservative": 8.0,
        "Conservative":      6.0,
        "Balanced":          4.0,
        "Growth":            2.0,
        "Aggressive":        1.0,
    }
    for profile, ra in expected.items():
        assert map_profile_to_risk_aversion(profile) == ra, (
            f"Profile {profile!r} should map to A={ra}"
        )


def test_profile_names_includes_custom():
    assert "Custom" in PROFILE_NAMES
    assert len(PROFILE_NAMES) == len(PROFILE_RISK_AVERSION) + 1


def test_utility_ordering_across_profiles():
    """For fixed (return, vol), Very Conservative must have lower utility than Aggressive."""
    ret, vol = 0.10, 0.20
    u_conservative = float(
        calculate_mean_variance_utility(ret, vol, PROFILE_RISK_AVERSION["Very Conservative"])
    )
    u_aggressive = float(
        calculate_mean_variance_utility(ret, vol, PROFILE_RISK_AVERSION["Aggressive"])
    )
    assert u_aggressive > u_conservative, (
        "An aggressive investor assigns higher utility to a risky portfolio "
        "than a very conservative investor"
    )
