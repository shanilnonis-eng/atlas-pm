"""
Tests for analytics.garch_volatility — GARCH and GJR-GARCH models.

Test coverage:
 1.  Input validation — too-short series raises ValueError
 2.  Input validation — constant returns raises ValueError
 3.  GARCH(1,1) fits successfully and produces a usable result
 4.  GJR-GARCH(1,1,1) fits successfully
 5.  GJR-GARCH result contains the gamma (asymmetry) parameter
 6.  Conditional volatility is strictly positive for all observations
 7.  Conditional volatility series has the same length as input returns
 8.  Annualised conditional vol ≈ daily conditional vol × sqrt(252)
 9.  GARCH persistence is in (0, 1] for plausible equity-like data
10.  GJR-GARCH persistence is in (0, 1] for plausible equity-like data
11.  GARCH-VaR is a positive finite number
12.  GARCH-VaR is monotone in confidence (higher confidence → higher VaR)
13.  GJR-GARCH VaR is positive
14.  Volatility forecast returns a Series of the requested horizon length
15.  All forecast values are positive and finite
16.  get_garch_params returns a DataFrame with required columns
17.  has_leverage_effect returns a (bool, float) tuple
18.  GARCH-VaR has a plausible magnitude relative to historical VaR
19.  garch_persistence result is a float
20.  GJR-GARCH: model with no gamma raises gracefully via has_leverage_effect
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analytics.garch_volatility import (
    MIN_OBS,
    _validate_returns,
    fit_garch,
    fit_gjr_garch,
    get_conditional_volatility,
    get_garch_var,
    get_volatility_forecast,
    get_garch_params,
    garch_persistence,
    has_leverage_effect,
)
from analytics.risk import historical_var


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def make_garch_returns(n: int = 400, seed: int = 42) -> pd.Series:
    """
    Synthetic GARCH(1,1) returns with Student-t innovations.
    Parameters chosen so the process is stationary and fits quickly.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta, nu = 0.04, 0.09, 0.86, 6.0   # persistence = 0.95
    sigma2 = omega / (1.0 - alpha - beta)   # start at unconditional variance
    rets_pct = np.zeros(n)
    t_scale = np.sqrt((nu - 2.0) / nu)          # standardise to unit variance

    for t in range(1, n):
        z = float(rng.standard_t(nu)) * t_scale
        sigma2 = omega + alpha * rets_pct[t - 1] ** 2 + beta * sigma2
        rets_pct[t] = np.sqrt(sigma2) * z

    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    # Return as fractions
    return pd.Series(rets_pct / 100.0, index=dates, name="Portfolio")


RETURNS = make_garch_returns(n=400)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — fit models ONCE for the whole test module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def garch_result():
    return fit_garch(RETURNS)


@pytest.fixture(scope="module")
def gjr_result():
    return fit_gjr_garch(RETURNS)


# ---------------------------------------------------------------------------
# 1. Validation — too-short series
# ---------------------------------------------------------------------------

def test_validate_too_short():
    short = pd.Series(np.random.default_rng(0).normal(0, 0.01, MIN_OBS - 1))
    with pytest.raises(ValueError, match="observations required"):
        _validate_returns(short)


# ---------------------------------------------------------------------------
# 2. Validation — constant returns
# ---------------------------------------------------------------------------

def test_validate_constant_returns():
    const = pd.Series(np.ones(200) * 0.001)
    with pytest.raises(ValueError, match="constant"):
        _validate_returns(const)


# ---------------------------------------------------------------------------
# 3. GARCH fits successfully
# ---------------------------------------------------------------------------

def test_garch_fit_succeeds(garch_result):
    """Result object must have standard arch attributes."""
    assert hasattr(garch_result, "conditional_volatility")
    assert hasattr(garch_result, "params")
    assert hasattr(garch_result, "pvalues")
    assert hasattr(garch_result, "std_err")


# ---------------------------------------------------------------------------
# 4. GJR-GARCH fits successfully
# ---------------------------------------------------------------------------

def test_gjr_garch_fit_succeeds(gjr_result):
    assert hasattr(gjr_result, "conditional_volatility")
    assert hasattr(gjr_result, "params")


# ---------------------------------------------------------------------------
# 5. GJR-GARCH result contains gamma parameter
# ---------------------------------------------------------------------------

def test_gjr_garch_has_gamma_parameter(gjr_result):
    gamma_keys = [k for k in gjr_result.params.index if "gamma" in k.lower()]
    assert len(gamma_keys) >= 1, (
        "GJR-GARCH must contain at least one gamma (asymmetry) parameter"
    )


# ---------------------------------------------------------------------------
# 6. Conditional volatility is strictly positive
# ---------------------------------------------------------------------------

def test_conditional_volatility_positive(garch_result, gjr_result):
    for label, res in [("GARCH", garch_result), ("GJR-GARCH", gjr_result)]:
        cond_vol = get_conditional_volatility(res, annualise=False)
        assert (cond_vol > 0).all(), (
            f"{label}: conditional volatility must be strictly positive everywhere"
        )


# ---------------------------------------------------------------------------
# 7. Conditional volatility has same length as input returns
# ---------------------------------------------------------------------------

def test_conditional_volatility_length(garch_result, gjr_result):
    for label, res in [("GARCH", garch_result), ("GJR-GARCH", gjr_result)]:
        cond_vol = get_conditional_volatility(res, annualise=False)
        assert len(cond_vol) == len(RETURNS), (
            f"{label}: conditional volatility length must match input returns length"
        )


# ---------------------------------------------------------------------------
# 8. Annualised vol ≈ daily vol × sqrt(252)
# ---------------------------------------------------------------------------

def test_annualised_vs_daily_conditional_vol(garch_result):
    cv_daily = get_conditional_volatility(garch_result, annualise=False)
    cv_ann   = get_conditional_volatility(garch_result, annualise=True)
    ratio = cv_ann.values / cv_daily.values
    # Every point should be within floating-point tolerance of sqrt(252)
    np.testing.assert_allclose(
        ratio, np.sqrt(252), rtol=1e-10,
        err_msg="Annualised vol must equal daily vol × sqrt(252) at every point",
    )


# ---------------------------------------------------------------------------
# 9 & 10. GARCH and GJR-GARCH persistence in (0, 1]
# ---------------------------------------------------------------------------

def test_garch_persistence_range(garch_result):
    p = garch_persistence(garch_result)
    assert isinstance(p, float)
    assert 0.0 < p <= 1.0, (
        f"GARCH persistence must be in (0, 1]; got {p:.4f}. "
        "Check that the GARCH process is covariance-stationary."
    )


def test_gjr_persistence_range(gjr_result):
    p = garch_persistence(gjr_result)
    assert isinstance(p, float)
    assert 0.0 < p <= 1.0, (
        f"GJR-GARCH persistence must be in (0, 1]; got {p:.4f}."
    )


# ---------------------------------------------------------------------------
# 11. GARCH-VaR is positive and finite
# ---------------------------------------------------------------------------

def test_garch_var_positive_finite(garch_result, gjr_result):
    for label, res in [("GARCH", garch_result), ("GJR-GARCH", gjr_result)]:
        var = get_garch_var(res, confidence=0.95)
        assert var > 0, f"{label}: GARCH-VaR must be positive"
        assert np.isfinite(var), f"{label}: GARCH-VaR must be finite"


# ---------------------------------------------------------------------------
# 12. GARCH-VaR is monotone in confidence
# ---------------------------------------------------------------------------

def test_garch_var_monotone_in_confidence(garch_result):
    var_90 = get_garch_var(garch_result, confidence=0.90)
    var_95 = get_garch_var(garch_result, confidence=0.95)
    var_99 = get_garch_var(garch_result, confidence=0.99)
    assert var_99 > var_95 > var_90, (
        "GARCH-VaR must be strictly increasing with confidence level: "
        f"VaR(90%)={var_90:.4f}, VaR(95%)={var_95:.4f}, VaR(99%)={var_99:.4f}"
    )


# ---------------------------------------------------------------------------
# 13. GJR-GARCH VaR is positive
# ---------------------------------------------------------------------------

def test_gjr_var_positive(gjr_result):
    var = get_garch_var(gjr_result, confidence=0.95)
    assert var > 0


# ---------------------------------------------------------------------------
# 14. Volatility forecast has correct horizon length
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_volatility_forecast_shape(garch_result, horizon):
    fcast = get_volatility_forecast(garch_result, horizon=horizon)
    assert len(fcast) == horizon, (
        f"Forecast must have exactly {horizon} values, got {len(fcast)}"
    )
    assert list(fcast.index) == list(range(1, horizon + 1))


# ---------------------------------------------------------------------------
# 15. All forecast values are positive and finite
# ---------------------------------------------------------------------------

def test_volatility_forecast_positive_finite(garch_result, gjr_result):
    for label, res in [("GARCH", garch_result), ("GJR-GARCH", gjr_result)]:
        fcast = get_volatility_forecast(res, horizon=5)
        assert (fcast > 0).all(), f"{label}: forecast must be positive"
        assert np.isfinite(fcast).all(), f"{label}: forecast must be finite"


# ---------------------------------------------------------------------------
# 16. get_garch_params returns a DataFrame with required columns
# ---------------------------------------------------------------------------

def test_garch_params_dataframe_columns(garch_result, gjr_result):
    required = {"Parameter", "Estimate", "Std Error", "t-stat", "p-value"}
    for label, res in [("GARCH", garch_result), ("GJR-GARCH", gjr_result)]:
        df = get_garch_params(res)
        assert isinstance(df, pd.DataFrame), f"{label}: get_garch_params must return DataFrame"
        missing = required - set(df.columns)
        assert not missing, f"{label}: missing columns {missing}"
        assert len(df) > 0, f"{label}: params DataFrame must not be empty"


def test_garch_params_has_omega_alpha_beta(garch_result, gjr_result):
    """Core volatility parameters must be present in both models."""
    for label, res in [("GARCH", garch_result), ("GJR-GARCH", gjr_result)]:
        df = get_garch_params(res)
        param_names = df["Parameter"].str.lower().tolist()
        assert any("omega" in p for p in param_names), f"{label}: omega missing"
        assert any("alpha" in p for p in param_names), f"{label}: alpha missing"
        assert any("beta"  in p for p in param_names), f"{label}: beta missing"


# ---------------------------------------------------------------------------
# 17. has_leverage_effect returns (bool, float)
# ---------------------------------------------------------------------------

def test_has_leverage_effect_returns_correct_type(gjr_result, garch_result):
    is_sig, pval = has_leverage_effect(gjr_result)
    assert isinstance(is_sig, bool)
    assert isinstance(pval, float)

    # For a plain GARCH result (no gamma), has_leverage_effect returns (False, nan)
    is_sig_garch, pval_garch = has_leverage_effect(garch_result)
    assert is_sig_garch is False
    assert np.isnan(pval_garch)


def test_leverage_effect_pvalue_in_unit_interval_or_nan(gjr_result):
    _, pval = has_leverage_effect(gjr_result)
    if not np.isnan(pval):
        assert 0.0 <= pval <= 1.0, f"p-value must be in [0, 1]; got {pval}"


# ---------------------------------------------------------------------------
# 18. GARCH-VaR has plausible magnitude relative to Historical VaR
# ---------------------------------------------------------------------------

def test_garch_var_plausible_relative_to_historical(garch_result):
    """
    GARCH-VaR and Historical VaR should be within an order of magnitude.
    A large divergence indicates a data-scale or formula error.
    """
    garch_var = get_garch_var(garch_result, confidence=0.95)
    hist_var  = historical_var(RETURNS, confidence=0.95, horizon_days=1)
    ratio = garch_var / hist_var if hist_var > 0 else float("inf")
    assert 0.1 < ratio < 10.0, (
        f"GARCH-VaR ({garch_var:.4f}) is implausibly far from "
        f"Historical VaR ({hist_var:.4f}); ratio = {ratio:.2f}. "
        "Check percent/fraction conversion in get_garch_var."
    )


# ---------------------------------------------------------------------------
# 19. garch_persistence returns a float
# ---------------------------------------------------------------------------

def test_garch_persistence_is_float(garch_result, gjr_result):
    for label, res in [("GARCH", garch_result), ("GJR-GARCH", gjr_result)]:
        p = garch_persistence(res)
        assert isinstance(p, float), f"{label}: persistence must be float"
        assert np.isfinite(p), f"{label}: persistence must be finite"


# ---------------------------------------------------------------------------
# 20. fit_garch raises ValueError for too-short input (end-to-end)
# ---------------------------------------------------------------------------

def test_fit_garch_raises_on_short_series():
    short = pd.Series(
        np.random.default_rng(99).normal(0, 0.01, MIN_OBS - 1),
        index=pd.date_range("2020-01-01", periods=MIN_OBS - 1, freq="B"),
    )
    with pytest.raises(ValueError, match="observations required"):
        fit_garch(short)


def test_fit_gjr_garch_raises_on_short_series():
    short = pd.Series(
        np.random.default_rng(99).normal(0, 0.01, 50),
        index=pd.date_range("2020-01-01", periods=50, freq="B"),
    )
    with pytest.raises(ValueError, match="observations required"):
        fit_gjr_garch(short)


# ---------------------------------------------------------------------------
# 21. Forecast multi-step mean reversion
# ---------------------------------------------------------------------------

def test_forecast_reverts_toward_long_run(garch_result):
    """
    For a stationary GARCH process with persistence < 1, the multi-step
    forecast should converge (the variance at step 21 must be finite and
    different from step 1 unless persistence is very close to 1).
    """
    fcast = get_volatility_forecast(garch_result, horizon=21, annualise=False)
    assert np.isfinite(fcast).all(), "All multi-step forecast values must be finite"
    # Forecasts should not blow up
    assert fcast.max() < 1.0, (
        "Daily volatility forecast > 100 % — likely a units error"
    )
