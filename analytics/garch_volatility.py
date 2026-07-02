"""
GARCH volatility modelling module for Atlas PM.

Implements two conditional heteroskedasticity models:

GARCH(1,1)  — Bollerslev (1986)
    sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
    Captures volatility clustering: periods of high vol follow high vol,
    and low vol follows low vol.  Alpha+beta (persistence) near 1 means
    shocks decay slowly.

GJR-GARCH(1,1,1)  — Glosten, Jagannathan & Runkle (1993)
    sigma^2_t = omega + alpha * epsilon^2_{t-1}
                      + gamma * epsilon^2_{t-1} * I(epsilon_{t-1} < 0)
                      + beta * sigma^2_{t-1}
    Adds the asymmetry (leverage effect): gamma > 0 means negative return
    shocks increase volatility MORE than positive shocks of equal magnitude.
    This is empirically robust in equity markets.

Both models are fitted using maximum likelihood with Student-t innovations
(better tail fit for daily equity returns than Gaussian).

Convention:
    - All public functions accept/return fractions (e.g. 0.01 = 1 %).
    - The arch library expects percent returns internally; conversion is
      handled transparently.
    - Conditional volatility is annualised by default (multiply by sqrt(252)).

Reference:
    Hansen & Lunde (2005) 'A forecast comparison of volatility models: does
    anything beat a GARCH(1,1)?', Journal of Applied Econometrics.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

MIN_OBS: int = 100  # minimum observations for reliable GARCH estimation


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_returns(returns: pd.Series) -> None:
    """
    Raise ValueError when returns cannot support GARCH estimation.

    Checks: minimum length, non-constant series.
    """
    clean = returns.dropna()
    if len(clean) < MIN_OBS:
        raise ValueError(
            f"At least {MIN_OBS} observations required for GARCH estimation; "
            f"got {len(clean)}."
        )
    if float(clean.std()) < 1e-10:
        raise ValueError(
            "Returns appear constant — GARCH estimation requires variance > 0."
        )


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "studentst",
    mean: str = "Constant",
):
    """
    Fit GARCH(p,q) model to daily simple returns.

    Parameters
    ----------
    returns : pd.Series of daily simple returns (fractions, e.g. 0.01 = 1 %)
    p       : ARCH order (number of lagged squared shocks)
    q       : GARCH order (number of lagged conditional variances)
    dist    : innovation distribution — 'studentst' (default) or 'normal'
    mean    : mean specification — 'Constant' (default) or 'Zero'

    Returns
    -------
    arch ARCHModelResult object.

    Notes
    -----
    Returns are converted to percent internally for numerical stability.
    All output extractors (get_conditional_volatility etc.) convert back
    to fractions.
    """
    from arch import arch_model

    _validate_returns(returns)
    r_pct = returns.dropna() * 100.0

    model = arch_model(r_pct, vol="Garch", p=p, q=q, dist=dist, mean=mean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            disp="off",
            options={"maxiter": 1000, "ftol": 1e-8},
        )
    return result


def fit_gjr_garch(
    returns: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "studentst",
    mean: str = "Constant",
):
    """
    Fit GJR-GARCH(p,o,q) model to daily simple returns.

    Parameters
    ----------
    returns : pd.Series of daily simple returns (fractions)
    p       : ARCH order
    o       : asymmetry (leverage) order — the GJR extension
    q       : GARCH order
    dist    : 'studentst' (default) or 'normal'
    mean    : 'Constant' (default) or 'Zero'

    Returns
    -------
    arch ARCHModelResult object.

    Notes
    -----
    The asymmetry parameter gamma[1] captures the leverage effect.
    A positive, statistically significant gamma means negative shocks
    increase volatility more than positive shocks of equal magnitude.
    """
    from arch import arch_model

    _validate_returns(returns)
    r_pct = returns.dropna() * 100.0

    model = arch_model(r_pct, vol="Garch", p=p, o=o, q=q, dist=dist, mean=mean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            disp="off",
            options={"maxiter": 1000, "ftol": 1e-8},
        )
    return result


# ---------------------------------------------------------------------------
# Conditional volatility
# ---------------------------------------------------------------------------

def get_conditional_volatility(
    result,
    annualise: bool = True,
) -> pd.Series:
    """
    Extract the in-sample conditional volatility series from a fitted model.

    Parameters
    ----------
    result   : ARCHModelResult from fit_garch or fit_gjr_garch
    annualise: if True, multiply by sqrt(252) to annualise (default True)

    Returns
    -------
    pd.Series of conditional volatility values (fractions).
    Same DatetimeIndex as the input returns series.
    """
    # arch stores conditional_volatility in percent (inputs were in percent)
    cond_vol = result.conditional_volatility / 100.0
    if annualise:
        cond_vol = cond_vol * np.sqrt(252)
    cond_vol.name = "Conditional Volatility (Ann.)" if annualise else "Conditional Volatility (Daily)"
    return cond_vol


# ---------------------------------------------------------------------------
# GARCH-based VaR
# ---------------------------------------------------------------------------

def get_garch_var(
    result,
    confidence: float = 0.95,
) -> float:
    """
    Compute 1-day GARCH VaR using the most recent conditional volatility.

    Uses the Student-t quantile when the model was fitted with Student-t
    innovations, otherwise falls back to the Normal quantile.

    VaR formula:  VaR = -(mu + z_alpha * sigma_T)

    where:
        sigma_T = last conditional volatility (fraction, daily)
        mu      = conditional mean (fraction, daily)
        z_alpha = (1-confidence)-th quantile of the standardised innovation dist
                  — negative for standard parameterisations

    Parameters
    ----------
    result     : ARCHModelResult from fit_garch or fit_gjr_garch
    confidence : confidence level (e.g. 0.95 for 95 % VaR)

    Returns
    -------
    Positive scalar (fraction) representing 1-day VaR at the given confidence.
    """
    # Last conditional volatility in fraction (daily)
    sigma_t = float(result.conditional_volatility.iloc[-1]) / 100.0

    # Mean return in fraction (daily)
    params = result.params
    if "mu" in params.index:
        mu = float(params["mu"]) / 100.0
    else:
        mu = 0.0

    # Quantile of the standardised innovation distribution
    alpha = 1.0 - confidence   # e.g. 0.05 for 95 % VaR

    if "nu" in params.index:
        nu = float(params["nu"])
        # arch uses the standardised Student-t (unit variance).
        # scipy.stats.t has variance = nu/(nu-2), so standardise:
        #   z_std = t.ppf(alpha, nu) * sqrt((nu-2)/nu)
        if nu > 2.0:
            scale = np.sqrt((nu - 2.0) / nu)
        else:
            scale = 1.0
        z_alpha = float(stats.t.ppf(alpha, df=nu)) * scale
    else:
        z_alpha = float(stats.norm.ppf(alpha))

    var_1d = -(mu + z_alpha * sigma_t)   # z_alpha < 0 → VaR > 0
    return float(max(var_1d, 0.0))


# ---------------------------------------------------------------------------
# Volatility forecast
# ---------------------------------------------------------------------------

def get_volatility_forecast(
    result,
    horizon: int = 5,
    annualise: bool = True,
) -> pd.Series:
    """
    Produce an N-day ahead conditional volatility forecast.

    Parameters
    ----------
    result   : ARCHModelResult from fit_garch or fit_gjr_garch
    horizon  : number of days ahead (default 5)
    annualise: if True, multiply by sqrt(252) (default True)

    Returns
    -------
    pd.Series indexed 1..horizon, values are conditional volatility (fractions).

    Notes
    -----
    The multi-step forecast is based on the unconditional GARCH recursion.
    For h > 1, the forecast reverts toward the long-run unconditional variance
    at a rate determined by the persistence (alpha+beta).
    """
    forecast = result.forecast(horizon=horizon)
    # forecast.variance: shape (n_obs, horizon); last row = most recent forecast
    var_fcast = forecast.variance.iloc[-1].values   # daily variance, percent^2
    vol_fcast = np.sqrt(var_fcast) / 100.0          # daily vol, fractions
    if annualise:
        vol_fcast = vol_fcast * np.sqrt(252)
    return pd.Series(
        vol_fcast,
        index=range(1, horizon + 1),
        name="Forecast Volatility (Ann.)" if annualise else "Forecast Volatility (Daily)",
    )


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def get_garch_params(result) -> pd.DataFrame:
    """
    Extract model parameters with standard errors, t-statistics, and p-values.

    Returns a tidy DataFrame with columns:
        Parameter, Estimate, Std Error, t-stat, p-value, Significant (10%)
    """
    df = pd.DataFrame({
        "Parameter": result.params.index.tolist(),
        "Estimate":  result.params.values,
        "Std Error": result.std_err.values,
        "t-stat":    result.tvalues.values,
        "p-value":   result.pvalues.values,
    })
    df["Significant (10%)"] = df["p-value"] < 0.10
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence and leverage effect
# ---------------------------------------------------------------------------

def garch_persistence(result) -> float:
    """
    Compute volatility persistence.

    GARCH(1,1)    : alpha + beta
    GJR-GARCH     : alpha + beta + gamma/2

    Persistence < 1 implies mean-reverting volatility (stationary process).
    Persistence close to 1 means shocks decay slowly (long volatility memory).
    An integrated GARCH (persistence = 1) has permanent shocks.
    """
    params = result.params
    alpha_sum = sum(
        float(params[k]) for k in params.index if k.lower().startswith("alpha")
    )
    beta_sum = sum(
        float(params[k]) for k in params.index if k.lower().startswith("beta")
    )
    # GJR: gamma enters persistence with weight 1/2 under symmetric distribution
    gamma_sum = sum(
        float(params[k]) for k in params.index if k.lower().startswith("gamma")
    )
    return float(alpha_sum + beta_sum + 0.5 * gamma_sum)


def has_leverage_effect(
    result,
    significance: float = 0.10,
) -> tuple[bool, float]:
    """
    Test whether the GJR-GARCH leverage parameter (gamma) is statistically
    significant at the given level.

    Parameters
    ----------
    result       : ARCHModelResult from fit_gjr_garch
    significance : significance threshold for the test (default 0.10)

    Returns
    -------
    (is_significant, p_value) : tuple of (bool, float)
        is_significant = True if gamma p-value < significance
        p_value = the smallest gamma p-value (or nan if no gamma params)
    """
    pvals = result.pvalues
    gamma_pvals = [
        float(pvals[k]) for k in pvals.index if k.lower().startswith("gamma")
    ]
    if not gamma_pvals:
        return False, float("nan")
    p = float(min(gamma_pvals))
    if np.isnan(p):
        return False, p
    return bool(p < significance), p
