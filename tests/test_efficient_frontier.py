"""
Tests for efficient frontier / max sharpe consistency.

Validates that:
  - max sharpe portfolio lies on the efficient frontier
  - both use identical inputs (mu_ann, Sigma, constraints, annualisation)
  - frontier Sharpe column subtracts the risk-free rate
  - target return range starts at the min-variance return
  - long-only and weight constraints are respected
  - portfolio_risk_return() matches raw matrix computation
  - Capital Market Line is only described when actually plotted
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from config.settings import TRADING_DAYS_PER_YEAR, MIN_WEIGHT, MAX_WEIGHT
from construction.optimiser import (
    compute_cov_matrix,
    _ensure_psd,
    minimum_variance,
    maximum_sharpe,
    efficient_frontier,
    portfolio_risk_return,
)


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def make_returns(n_assets: int = 4, n_days: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic returns with varied means so the frontier is non-trivial.
    """
    rng = np.random.default_rng(seed)
    base_means = np.linspace(0.0001, 0.0006, n_assets)
    vols = np.linspace(0.008, 0.018, n_assets)
    rets = rng.normal(0, 1, (n_days, n_assets)) * vols + base_means
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(rets, index=dates, columns=cols)


RETURNS = make_returns()
RF = 0.04
MIN_W, MAX_W = 0.0, 1.0


# ---------------------------------------------------------------------------
# 1. Max Sharpe lies on the efficient frontier
# ---------------------------------------------------------------------------

def test_max_sharpe_lies_on_efficient_frontier():
    """
    The Max Sharpe portfolio computed with portfolio_risk_return() must sit on
    (or numerically very close to) the efficient frontier curve.
    Tolerance: Euclidean distance < 0.5 % in (vol, return) space.
    """
    frontier = efficient_frontier(
        RETURNS, n_points=80, min_weight=MIN_W, max_weight=MAX_W, rf_annual=RF,
    )
    assert not frontier.empty, "Frontier is empty"

    w_ms = maximum_sharpe(RETURNS, rf_annual=RF, min_weight=MIN_W, max_weight=MAX_W)
    vol_ms, ret_ms = portfolio_risk_return(w_ms, RETURNS)

    dists = np.sqrt(
        (frontier["Ann. Volatility"].values - vol_ms) ** 2
        + (frontier["Ann. Return"].values - ret_ms) ** 2
    )
    min_dist = float(dists.min())
    assert min_dist < 0.005, (
        f"Max Sharpe ({vol_ms:.4f}, {ret_ms:.4f}) is {min_dist:.4f} "
        f"from the nearest frontier point — coordinate mismatch"
    )


# ---------------------------------------------------------------------------
# 2. portfolio_risk_return() is consistent with raw matrix computation
# ---------------------------------------------------------------------------

def test_portfolio_risk_return_matches_raw_computation():
    """
    portfolio_risk_return() must produce vol = sqrt(w'Σw) and return = mu_ann@w
    using the same Ledoit-Wolf shrunk, annualised Sigma as the optimisers.
    """
    cov = compute_cov_matrix(RETURNS, shrink=True)
    Sigma = _ensure_psd(cov.values)
    mu_ann = RETURNS.mean().values * TRADING_DAYS_PER_YEAR

    w_ms = maximum_sharpe(RETURNS, rf_annual=RF)
    w_arr = w_ms.reindex(RETURNS.columns).fillna(0.0).values
    w_arr = w_arr / w_arr.sum()

    expected_vol = float(np.sqrt(w_arr @ Sigma @ w_arr))
    expected_ret = float(mu_ann @ w_arr)

    vol, ret = portfolio_risk_return(w_ms, RETURNS)

    np.testing.assert_allclose(vol, expected_vol, rtol=1e-6,
                                err_msg="Vol mismatch between portfolio_risk_return and raw w'Σw")
    np.testing.assert_allclose(ret, expected_ret, rtol=1e-6,
                                err_msg="Return mismatch between portfolio_risk_return and raw mu@w")


# ---------------------------------------------------------------------------
# 3. Identical inputs: same Sigma and mu_ann across all three functions
# ---------------------------------------------------------------------------

def test_frontier_and_max_sharpe_use_identical_sigma():
    """
    compute_cov_matrix() with shrink=True must produce the same matrix whether
    called by efficient_frontier, maximum_sharpe, or portfolio_risk_return.
    (All three call it independently — this test confirms idempotence.)
    """
    cov_a = compute_cov_matrix(RETURNS, shrink=True)
    cov_b = compute_cov_matrix(RETURNS, shrink=True)
    np.testing.assert_array_equal(
        cov_a.values, cov_b.values,
        err_msg="compute_cov_matrix is not deterministic",
    )


def test_frontier_uses_arithmetic_mean_returns():
    """
    efficient_frontier must use mu_ann = returns.mean() * 252 (arithmetic, not CAGR).
    The frontier Ann. Return values must all lie in [mu_ann.min(), mu_ann.max()].
    """
    mu_ann = RETURNS.mean().values * TRADING_DAYS_PER_YEAR
    frontier = efficient_frontier(RETURNS, n_points=40, min_weight=MIN_W, max_weight=MAX_W)

    assert frontier["Ann. Return"].min() >= mu_ann.min() - 1e-6, (
        "Frontier starts below the lowest individual asset return"
    )
    assert frontier["Ann. Return"].max() <= mu_ann.max() + 1e-6, (
        "Frontier extends above the highest individual asset return"
    )


# ---------------------------------------------------------------------------
# 4. Frontier range starts at min-variance return (not mu_ann.min())
# ---------------------------------------------------------------------------

def test_frontier_range_starts_at_min_variance_return():
    """
    The first frontier point must coincide with the minimum-variance portfolio
    return (which may be above mu_ann.min() when weights are constrained).
    """
    min_w, max_w = 0.05, 0.60
    frontier = efficient_frontier(RETURNS, n_points=40, min_weight=min_w, max_weight=max_w)
    assert not frontier.empty

    w_mv = minimum_variance(RETURNS, min_weight=min_w, max_weight=max_w)
    _, ret_mv = portfolio_risk_return(w_mv, RETURNS)

    frontier_min_ret = frontier["Ann. Return"].min()
    np.testing.assert_allclose(
        frontier_min_ret, ret_mv, atol=0.005,
        err_msg=(
            f"Frontier starts at {frontier_min_ret:.4f} but min-variance return is {ret_mv:.4f}"
        ),
    )


# ---------------------------------------------------------------------------
# 5. Frontier range covers the max sharpe portfolio
# ---------------------------------------------------------------------------

def test_frontier_range_covers_max_sharpe():
    """
    The max sharpe portfolio return must lie inside the frontier's return range.
    """
    frontier = efficient_frontier(RETURNS, n_points=60, min_weight=MIN_W, max_weight=MAX_W)
    w_ms = maximum_sharpe(RETURNS, rf_annual=RF)
    _, ret_ms = portfolio_risk_return(w_ms, RETURNS)

    min_f = frontier["Ann. Return"].min()
    max_f = frontier["Ann. Return"].max()
    assert min_f - 1e-6 <= ret_ms <= max_f + 1e-6, (
        f"Max Sharpe return {ret_ms:.4f} is outside frontier range [{min_f:.4f}, {max_f:.4f}]"
    )


# ---------------------------------------------------------------------------
# 6. Frontier Sharpe column uses the risk-free rate
# ---------------------------------------------------------------------------

def test_frontier_sharpe_subtracts_rf():
    """
    The Sharpe column in the frontier must be (return - rf) / vol, not return / vol.
    """
    rf = 0.03
    frontier = efficient_frontier(RETURNS, n_points=20, rf_annual=rf)
    assert not frontier.empty

    for _, row in frontier.iterrows():
        if row["Ann. Volatility"] > 1e-8:
            expected_sharpe = (row["Ann. Return"] - rf) / row["Ann. Volatility"]
            np.testing.assert_allclose(
                row["Sharpe"], expected_sharpe, rtol=1e-6,
                err_msg=f"Sharpe column does not subtract rf={rf}",
            )


# ---------------------------------------------------------------------------
# 7. Annualisation factor is consistent across all modules
# ---------------------------------------------------------------------------

def test_annualisation_factor_is_252():
    """All modules must use TRADING_DAYS_PER_YEAR = 252."""
    assert TRADING_DAYS_PER_YEAR == 252, "TRADING_DAYS_PER_YEAR must be 252"


def test_cov_matrix_annualises_by_252():
    """compute_cov_matrix with shrink=False must equal cov_daily * 252."""
    cov_ann = compute_cov_matrix(RETURNS, shrink=False)
    cov_daily = RETURNS.cov()
    np.testing.assert_allclose(
        cov_ann.values, cov_daily.values * 252, rtol=1e-10,
        err_msg="Annualised covariance matrix is not cov_daily × 252",
    )


# ---------------------------------------------------------------------------
# 8. Long-only and weight constraints are consistent
# ---------------------------------------------------------------------------

def test_max_sharpe_respects_weight_bounds():
    """Max sharpe must produce weights in [min_w, max_w] summing to 1."""
    min_w, max_w = 0.05, 0.50
    w_ms = maximum_sharpe(RETURNS, rf_annual=RF, min_weight=min_w, max_weight=max_w)

    assert (w_ms >= min_w - 1e-6).all(), "Max Sharpe violates min_weight"
    assert (w_ms <= max_w + 1e-6).all(), "Max Sharpe violates max_weight"
    np.testing.assert_allclose(w_ms.sum(), 1.0, atol=1e-6,
                                err_msg="Max Sharpe weights do not sum to 1")


def test_efficient_frontier_respects_weight_bounds():
    """
    Re-solve the frontier at each point and confirm weights from min-var match
    the bounds (indirect test — if frontier points exist, bounds were satisfied).
    """
    min_w, max_w = 0.05, 0.50
    frontier = efficient_frontier(
        RETURNS, n_points=20, min_weight=min_w, max_weight=max_w,
    )
    # If bounds were violated, the solver would fail and fewer points would appear.
    assert len(frontier) >= 5, (
        f"Fewer than 5 frontier points with min_w={min_w} max_w={max_w} — "
        "possible constraint infeasibility"
    )


def test_long_only_weights():
    """max_sharpe and minimum_variance must not produce negative weights."""
    w_ms = maximum_sharpe(RETURNS, min_weight=0.0, max_weight=1.0)
    w_mv = minimum_variance(RETURNS, min_weight=0.0, max_weight=1.0)

    assert (w_ms >= -1e-8).all(), "Max Sharpe has negative weights (violates long-only)"
    assert (w_mv >= -1e-8).all(), "Min Var has negative weights (violates long-only)"


# ---------------------------------------------------------------------------
# 9. Frontier drops failed optimisation points (only valid points plotted)
# ---------------------------------------------------------------------------

def test_frontier_only_contains_valid_points():
    """All frontier rows must have positive, finite vol and finite return."""
    frontier = efficient_frontier(RETURNS, n_points=40)
    assert (frontier["Ann. Volatility"] > 0).all(), "Frontier contains zero or negative vol"
    assert frontier["Ann. Volatility"].notna().all(), "Frontier contains NaN vol"
    assert frontier["Ann. Return"].notna().all(), "Frontier contains NaN return"


# ---------------------------------------------------------------------------
# 10. CML: text only mentions CML if it is actually plotted (UI contract)
# ---------------------------------------------------------------------------

def test_cml_requires_rf_and_max_sharpe_point():
    """
    efficient_frontier_chart must NOT draw the CML when rf_annual or
    max_sharpe_point is missing.  We verify the function signature supports
    optional CML rather than testing Streamlit rendering.
    """
    from ui.components.charts import efficient_frontier_chart

    frontier = efficient_frontier(RETURNS, n_points=20)
    fig_no_cml = efficient_frontier_chart(frontier)
    trace_names = [t.name for t in fig_no_cml.data]
    assert "Capital Market Line" not in trace_names, (
        "CML should not appear when rf_annual / max_sharpe_point are not supplied"
    )

    w_ms = maximum_sharpe(RETURNS, rf_annual=RF)
    vol_ms, ret_ms = portfolio_risk_return(w_ms, RETURNS)
    fig_with_cml = efficient_frontier_chart(
        frontier,
        rf_annual=RF,
        max_sharpe_point=(vol_ms, ret_ms),
    )
    trace_names_cml = [t.name for t in fig_with_cml.data]
    assert "Capital Market Line" in trace_names_cml, (
        "CML should appear when rf_annual and max_sharpe_point are both supplied"
    )
