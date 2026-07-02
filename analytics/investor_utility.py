"""
Mean-variance investor utility module for Atlas PM.

Implements the standard mean-variance utility function and indifference curves
from modern portfolio theory:

    U = E(r) - 0.5 * A * sigma^2

where:
    E(r)   = annualised expected return (decimal fraction)
    sigma  = annualised volatility (decimal fraction)
    A      = risk-aversion coefficient (positive scalar; typical range 1–10)

Indifference curve at fixed utility level U:
    E(r) = U + 0.5 * A * sigma^2

These functions support the risk-preference overlay on the efficient frontier.
They are illustrative and do not constitute investment advice.

Reference: Markowitz (1952) 'Portfolio Selection', Journal of Finance.
           Bodie, Kane & Marcus 'Investments' (standard MV utility notation).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Canonical investor profile → risk-aversion coefficient mapping
# ---------------------------------------------------------------------------

PROFILE_RISK_AVERSION: dict[str, float] = {
    "Very Conservative": 8.0,
    "Conservative":      6.0,
    "Balanced":          4.0,
    "Growth":            2.0,
    "Aggressive":        1.0,
}

PROFILE_NAMES: list[str] = list(PROFILE_RISK_AVERSION.keys()) + ["Custom"]


def map_profile_to_risk_aversion(profile: str) -> float:
    """Return the risk-aversion coefficient for a named investor profile.

    Raises ValueError for unknown profile names.
    """
    if profile not in PROFILE_RISK_AVERSION:
        raise ValueError(
            f"Unknown profile {profile!r}. "
            f"Valid options: {list(PROFILE_RISK_AVERSION)}"
        )
    return PROFILE_RISK_AVERSION[profile]


# ---------------------------------------------------------------------------
# Core utility formula  U = E(r) - 0.5 * A * sigma^2
# ---------------------------------------------------------------------------

def calculate_mean_variance_utility(
    expected_return: float | np.ndarray,
    volatility: float | np.ndarray,
    risk_aversion: float,
) -> float | np.ndarray:
    """
    Mean-variance utility: U = E(r) - 0.5 * A * sigma^2

    Parameters
    ----------
    expected_return : annualised expected return (fraction, e.g. 0.08 = 8 %)
    volatility      : annualised volatility (fraction, e.g. 0.15 = 15 %)
    risk_aversion   : risk-aversion coefficient A (must be > 0)

    Returns
    -------
    Utility value(s) as a numpy scalar or array; higher is preferred.
    """
    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be positive, got {risk_aversion}"
        )
    return (
        np.asarray(expected_return, dtype=float)
        - 0.5 * risk_aversion * np.asarray(volatility, dtype=float) ** 2
    )


# ---------------------------------------------------------------------------
# Indifference curve  E(r) = U + 0.5 * A * sigma^2
# ---------------------------------------------------------------------------

def calculate_indifference_curve(
    volatility_range: np.ndarray,
    utility_level: float,
    risk_aversion: float,
) -> np.ndarray:
    """
    Return expected returns along a mean-variance indifference curve.

    E(r) = U + 0.5 * A * sigma^2

    Parameters
    ----------
    volatility_range : 1-D array of annualised volatility values (fractions)
    utility_level    : constant utility level for this curve
    risk_aversion    : risk-aversion coefficient A (must be > 0)

    Returns
    -------
    1-D array of expected returns (fractions).
    The curve is strictly upward sloping and convex.
    """
    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be positive, got {risk_aversion}"
        )
    vols = np.asarray(volatility_range, dtype=float)
    return utility_level + 0.5 * risk_aversion * vols ** 2


# ---------------------------------------------------------------------------
# Utility-optimal frontier portfolio
# ---------------------------------------------------------------------------

def find_utility_optimal_portfolio(
    frontier_returns: np.ndarray,
    frontier_volatilities: np.ndarray,
    risk_aversion: float,
) -> dict[str, float]:
    """
    Find the efficient-frontier portfolio with the highest mean-variance utility.

    Parameters
    ----------
    frontier_returns      : array of annualised expected returns (fractions)
    frontier_volatilities : array of annualised volatilities (fractions)
    risk_aversion         : risk-aversion coefficient A (must be > 0)

    Returns
    -------
    dict with keys:
        index           : int   — position in the input arrays
        expected_return : float — annualised expected return at the optimal point
        volatility      : float — annualised volatility at the optimal point
        utility         : float — utility score at the optimal point
    """
    rets = np.asarray(frontier_returns, dtype=float)
    vols = np.asarray(frontier_volatilities, dtype=float)

    if rets.size == 0 or vols.size == 0:
        raise ValueError("Frontier arrays must be non-empty")
    if rets.shape != vols.shape:
        raise ValueError(
            f"frontier_returns and frontier_volatilities must have the same shape; "
            f"got {rets.shape} vs {vols.shape}"
        )
    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be positive, got {risk_aversion}"
        )

    utilities = calculate_mean_variance_utility(rets, vols, risk_aversion)
    idx = int(np.argmax(utilities))

    return {
        "index":           idx,
        "expected_return": float(rets[idx]),
        "volatility":      float(vols[idx]),
        "utility":         float(utilities[idx]),
    }
