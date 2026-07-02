"""
Risk analytics module for Atlas PM.

Covers:
- Historical Value at Risk (VaR) — empirical, no distributional assumption
- Conditional VaR (CVaR / Expected Shortfall) — average loss beyond VaR
- Parametric VaR — Gaussian assumption, shown alongside historical for comparison
- Marginal / Component risk contribution — which assets drive portfolio risk
- Correlation matrix
- Stress testing — instantaneous shock to current weights
- Factor decomposition (market beta decomposition)

All VaR numbers are expressed as positive magnitudes (loss = positive number)
and as fractions (not percentages) unless the caller converts.

Key concept: we use HISTORICAL simulation as the primary VaR method because
it makes no distributional assumption and naturally captures fat tails and
skewness in financial returns — properties that the 2008 and 2020 crises
demonstrated are very real.  Parametric (Gaussian) VaR is included as a
comparison to show how it underestimates tail risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import (
    VAR_CONFIDENCE,
    TRADING_DAYS_PER_YEAR,
    STRESS_SCENARIOS,
    ASSET_SHORT_NAMES,
)


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

def historical_var(
    returns: pd.Series,
    confidence: float = VAR_CONFIDENCE,
    horizon_days: int = 1,
) -> float:
    """
    Historical (empirical) VaR.

    Sorts observed returns and reads off the (1-confidence) quantile.
    horizon_days > 1 uses the square-root-of-time scaling — an approximation
    valid for i.i.d. returns (a known limitation we document in governance).

    Returns: positive number, e.g. 0.021 = 2.1 % loss at given confidence.
    """
    var_1d = float(-np.percentile(returns.dropna(), (1 - confidence) * 100))
    return var_1d * np.sqrt(horizon_days)


def historical_cvar(
    returns: pd.Series,
    confidence: float = VAR_CONFIDENCE,
    horizon_days: int = 1,
) -> float:
    """
    Historical CVaR (Expected Shortfall).

    Average of all returns that fall below the VaR threshold.
    CVaR is a coherent risk measure; VaR is not — CVaR is therefore
    preferred by regulators and sophisticated risk managers.

    Returns: positive number representing expected loss in the tail.
    """
    threshold = -historical_var(returns, confidence, horizon_days=1)
    tail      = returns[returns <= threshold]
    if len(tail) == 0:
        return historical_var(returns, confidence, horizon_days)
    cvar_1d = float(-tail.mean())
    return cvar_1d * np.sqrt(horizon_days)


def parametric_var(
    returns: pd.Series,
    confidence: float = VAR_CONFIDENCE,
    horizon_days: int = 1,
) -> float:
    """
    Parametric (Gaussian) VaR — assumes normally distributed returns.

    This underestimates tail risk because daily returns have fat tails
    (excess kurtosis > 0).  Included as a reference / comparison.
    """
    mu    = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z     = stats.norm.ppf(1 - confidence)
    var_1d = -(mu + z * sigma)
    return max(var_1d * np.sqrt(horizon_days), 0.0)


def var_summary(
    returns: pd.Series,
    portfolio_value: float = 1_000_000,
    confidence: float = VAR_CONFIDENCE,
) -> pd.DataFrame:
    """
    Return a summary DataFrame comparing 1-day and 10-day VaR/CVaR
    in both percentage and £ terms for a given portfolio value.
    """
    rows = []
    for horizon, label in [(1, "1-Day"), (10, "10-Day")]:
        hist_var  = historical_var(returns, confidence, horizon)
        hist_cvar = historical_cvar(returns, confidence, horizon)
        para_var  = parametric_var(returns, confidence, horizon)
        rows.append({
            "Horizon":           label,
            "Method":            "Historical",
            "VaR (%)":           hist_var,
            "VaR (£)":           hist_var * portfolio_value,
            "CVaR / ES (%)":     hist_cvar,
            "CVaR / ES (£)":     hist_cvar * portfolio_value,
        })
        rows.append({
            "Horizon":           label,
            "Method":            "Parametric (Gaussian)",
            "VaR (%)":           para_var,
            "VaR (£)":           para_var * portfolio_value,
            "CVaR / ES (%)":     float("nan"),
            "CVaR / ES (£)":     float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Risk contribution
# ---------------------------------------------------------------------------

def marginal_risk_contribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Marginal risk contribution (MRC) of each asset.

    MRC_i = (Σ w)_i / σ_p
    where Σ is the covariance matrix and σ_p is portfolio volatility.
    Annualised.
    """
    w     = weights.values
    sigma = cov_matrix.values
    port_var = float(w @ sigma @ w) * TRADING_DAYS_PER_YEAR
    port_vol = np.sqrt(port_var)
    mrc = (sigma @ w) * np.sqrt(TRADING_DAYS_PER_YEAR) / port_vol
    return pd.Series(mrc, index=weights.index, name="Marginal Risk Contribution")


def component_risk_contribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Component risk contribution (CRC) = w_i × MRC_i.
    CRC sums to portfolio volatility.
    Also returns percentage contribution to total risk.
    """
    mrc = marginal_risk_contribution(weights, cov_matrix)
    crc = weights * mrc
    port_vol = float(crc.sum())
    pct = crc / port_vol if port_vol != 0 else crc * float("nan")

    result = pd.DataFrame({
        "Weight":             weights,
        "Marginal RC":        mrc,
        "Component RC":       crc,
        "% Risk Contribution": pct,
    })
    return result


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of daily asset returns."""
    return returns.corr()


def rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 63,
) -> pd.Series:
    """60-day rolling correlation between two return series."""
    corr = returns_a.rolling(window).corr(returns_b)
    corr.name = f"Rolling Corr ({window}d)"
    return corr


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

def run_stress_test(
    weights: dict[str, float],
    scenarios: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Apply pre-defined instantaneous shock scenarios to a portfolio.

    Parameters
    ----------
    weights   : dict of {asset_label: weight}
    scenarios : dict of {scenario_name: {asset_short_name: shock_pct}}
                If None, uses STRESS_SCENARIOS from settings.

    Returns
    -------
    DataFrame with scenario names as index and portfolio P&L (fraction) as column.

    Methodology note: shocks are applied as instantaneous 1-day returns.
    This ignores correlation dynamics (e.g. flight-to-quality) during a
    crisis, which in reality would change correlations — a documented limitation.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    # build a weight series indexed by short names for scenario lookup
    label_to_short = ASSET_SHORT_NAMES
    short_weights = {}
    for label, w in weights.items():
        short = label_to_short.get(label)
        if short:
            short_weights[short] = w

    results = []
    for scenario_name, shocks in scenarios.items():
        pnl = 0.0
        detail = {}
        for short_name, shock in shocks.items():
            w = short_weights.get(short_name, 0.0)
            contribution = w * shock
            pnl += contribution
            detail[short_name] = contribution

        results.append({
            "Scenario":            scenario_name,
            "Portfolio P&L":       pnl,
            **detail,
        })

    df = pd.DataFrame(results).set_index("Scenario")
    return df


def var_backtesting(
    returns: pd.Series,
    confidence: float = VAR_CONFIDENCE,
    window: int = 252,
) -> pd.DataFrame:
    """
    Rolling VaR back-test: at each date, compute the 1-day VaR using
    the trailing 'window' days of returns, then check if the next day's
    return breached it.

    Returns a DataFrame with columns: VaR_estimate, actual_return, breach.
    Used to assess whether our historical VaR model is well-calibrated.
    Kupiec's proportion-of-failures (POF) test: expected breach rate = 1 - confidence.
    """
    var_estimates = returns.rolling(window).apply(
        lambda r: historical_var(pd.Series(r), confidence, 1),
        raw=False,
    ).shift(1)  # yesterday's VaR applied to today's return

    actual = returns.copy()
    breach = actual < -var_estimates  # loss exceeded VaR

    result = pd.DataFrame({
        "VaR Estimate":  var_estimates,
        "Actual Return": actual,
        "Breach":        breach,
    }).dropna()

    expected_breaches = (1 - confidence)
    actual_breach_rate = float(result["Breach"].mean())
    result.attrs["expected_breach_rate"] = expected_breaches
    result.attrs["actual_breach_rate"]   = actual_breach_rate
    result.attrs["n_breaches"]           = int(result["Breach"].sum())

    return result
