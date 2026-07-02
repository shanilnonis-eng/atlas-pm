"""
Portfolio optimisation module for Atlas PM.

Implements four construction models used in institutional asset management:

1. Equal Weight (EW)
   - Simplest possible model; strong out-of-sample track record vs complex models
   - 1/N portfolio — useful as a benchmark for all other models

2. Minimum Variance (MinVar)
   - Minimises portfolio variance subject to weight constraints
   - No return assumption required — pure risk model
   - Solved via quadratic programming (scipy/cvxpy)

3. Maximum Sharpe (MaxSharpe)
   - Maximises the Sharpe ratio (tangency portfolio on the efficient frontier)
   - Requires return estimates — we use historical mean returns as the input
     (with a known limitation: return estimation error)
   - Solved by reformulating as a QP (Sharpe maximisation = scaled MinVar)

4. Risk Parity (RP)
   - Each asset contributes equally to portfolio variance
   - Does NOT require return estimates
   - Solved iteratively (no closed-form solution)
   - Also known as "Equal Risk Contribution" (ERC)

Implementation notes:
- We use scipy.optimize.minimize for all models (no cvxpy dependency)
- Constraints: weights sum to 1, min/max per-asset bounds
- Covariance matrix is shrunk toward the identity (Ledoit-Wolf) to reduce
  estimation error when the number of assets is small relative to history
- All optimisers return a pd.Series of weights indexed by asset name
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import eigh

from config.settings import (
    MIN_WEIGHT,
    MAX_WEIGHT,
    TRADING_DAYS_PER_YEAR,
    MODEL_NAMES,
)


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def ledoit_wolf_shrinkage(cov: np.ndarray) -> np.ndarray:
    """
    Apply analytical Ledoit-Wolf shrinkage toward the scaled identity matrix.

    This reduces estimation error in the sample covariance matrix, which is
    especially noisy when T (time observations) is not >> N (assets).
    The shrinkage intensity is determined analytically, not by cross-validation.

    Reference: Ledoit & Wolf (2004) 'A well-conditioned estimator for large-
    dimensional covariance matrices', Journal of Multivariate Analysis.
    """
    n, p = cov.shape
    mu    = np.trace(cov) / p
    delta = ((np.linalg.norm(cov, 'fro') ** 2 + mu ** 2) /
             ((n + 1) * (np.linalg.norm(cov - mu * np.eye(p), 'fro') ** 2)))
    alpha = min(delta, 1.0)
    return (1 - alpha) * cov + alpha * mu * np.eye(p)


def compute_cov_matrix(
    returns: pd.DataFrame,
    shrink: bool = True,
) -> pd.DataFrame:
    """
    Compute annualised covariance matrix with optional Ledoit-Wolf shrinkage.

    Returns a DataFrame indexed and columned by asset names.
    """
    returns = returns.dropna()
    cov = returns.cov() * TRADING_DAYS_PER_YEAR
    if shrink:
        cov_arr = ledoit_wolf_shrinkage(cov.values)
        cov = pd.DataFrame(cov_arr, index=cov.index, columns=cov.columns)
    return cov


def _ensure_psd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clip negative eigenvalues to eps to enforce positive semi-definiteness."""
    vals, vecs = eigh(cov)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(vals) @ vecs.T


# ---------------------------------------------------------------------------
# Model 1: Equal Weight
# ---------------------------------------------------------------------------

def equal_weight(assets: list[str]) -> pd.Series:
    """1/N allocation across all selected assets."""
    n = len(assets)
    return pd.Series(1.0 / n, index=assets, name="Equal Weight")


# ---------------------------------------------------------------------------
# Model 2: Minimum Variance
# ---------------------------------------------------------------------------

def minimum_variance(
    returns: pd.DataFrame,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
    shrink: bool = True,
) -> pd.Series:
    """
    Minimise portfolio variance subject to weight constraints.

    Objective:  min  w' Σ w
    Subject to: Σ w_i = 1,  w_i ∈ [min_weight, max_weight]

    Returns a Series of optimal weights indexed by asset name.
    """
    cov  = compute_cov_matrix(returns, shrink=shrink)
    Sigma = _ensure_psd(cov.values)
    n    = len(returns.columns)
    assets = list(returns.columns)

    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ Sigma @ w)

    def grad_variance(w: np.ndarray) -> np.ndarray:
        return 2 * Sigma @ w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.ones(n) / n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            portfolio_variance,
            w0,
            jac=grad_variance,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

    if not result.success:
        # fallback to equal weight if solver fails
        return equal_weight(assets)

    weights = pd.Series(result.x, index=assets, name="Minimum Variance")
    weights = weights.clip(lower=0).div(weights.clip(lower=0).sum())
    return weights


# ---------------------------------------------------------------------------
# Model 3: Maximum Sharpe
# ---------------------------------------------------------------------------

def maximum_sharpe(
    returns: pd.DataFrame,
    rf_annual: float = 0.04,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
    shrink: bool = True,
) -> pd.Series:
    """
    Maximise the Sharpe ratio (tangency portfolio).

    We use the standard reformulation: instead of maximising (mu-rf)/sigma,
    we minimise the negative Sharpe directly via scipy, which is robust to
    sign changes and avoids the homogeneous transformation pitfalls.

    Inputs:
        rf_annual : annual risk-free rate (e.g. 0.04 = 4 %)
                    We convert to daily for excess return calculation.
    """
    cov    = compute_cov_matrix(returns, shrink=shrink)
    Sigma  = _ensure_psd(cov.values)
    mu_ann = returns.mean() * TRADING_DAYS_PER_YEAR  # annualised mean returns
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess_ann = (returns.mean() - rf_daily) * TRADING_DAYS_PER_YEAR

    n      = len(returns.columns)
    assets = list(returns.columns)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(excess_ann.values @ w)
        port_vol = float(np.sqrt(w @ Sigma @ w))
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret / port_vol)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.ones(n) / n

    best_w, best_sharpe = w0, float("inf")
    # try multiple starting points to avoid local minima
    rng = np.random.default_rng(42)
    starting_points = [w0] + [rng.dirichlet(np.ones(n)) for _ in range(20)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for w_start in starting_points:
            res = minimize(
                neg_sharpe,
                w_start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if res.success and res.fun < best_sharpe:
                best_sharpe = res.fun
                best_w = res.x

    weights = pd.Series(best_w, index=assets, name="Maximum Sharpe")
    weights = weights.clip(lower=0).div(weights.clip(lower=0).sum())
    return weights


# ---------------------------------------------------------------------------
# Model 4: Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------

def risk_parity(
    returns: pd.DataFrame,
    min_weight: float = 0.01,  # RP rarely wants zero weights
    max_weight: float = MAX_WEIGHT,
    shrink: bool = True,
) -> pd.Series:
    """
    Equal Risk Contribution (ERC) portfolio.

    Each asset i contributes equally to total portfolio variance:
        w_i * (Σw)_i = portfolio_variance / N  for all i

    We minimise the sum of squared differences between each asset's
    risk contribution and the target (1/N * portfolio variance).

    Reference: Maillard, Roncalli & Teïletche (2010) 'The Properties of
    Equally Weighted Risk Contribution Portfolios', Journal of Portfolio Management.
    """
    cov   = compute_cov_matrix(returns, shrink=shrink)
    Sigma = _ensure_psd(cov.values)
    n     = len(returns.columns)
    assets = list(returns.columns)
    target = 1.0 / n  # target % risk contribution from each asset

    def risk_contribution_error(w: np.ndarray) -> float:
        port_var = float(w @ Sigma @ w)
        if port_var < 1e-14:
            return 1e10
        mrc = Sigma @ w
        crc = w * mrc / port_var  # % contribution to variance
        return float(np.sum((crc - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(min_weight, max_weight)] * n
    w0 = np.ones(n) / n

    rng = np.random.default_rng(0)
    starting_points = [w0] + [rng.dirichlet(np.ones(n)) for _ in range(15)]
    best_w, best_obj = w0, float("inf")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for w_start in starting_points:
            res = minimize(
                risk_contribution_error,
                w_start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-14, "maxiter": 2000},
            )
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_w = res.x

    weights = pd.Series(best_w, index=assets, name="Risk Parity")
    weights = weights.clip(lower=0).div(weights.clip(lower=0).sum())
    return weights


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

def portfolio_risk_return(
    weights: pd.Series,
    returns: pd.DataFrame,
    shrink: bool = True,
) -> tuple[float, float]:
    """
    Compute (ann_vol, ann_return) using the SAME covariance matrix and expected
    returns as efficient_frontier() and maximum_sharpe().

    Must be used instead of realised-return statistics when overlaying any
    portfolio on the efficient frontier — both must live in the same arithmetic
    space (arithmetic mean × 252, Ledoit-Wolf shrunk Σ).
    """
    cov    = compute_cov_matrix(returns, shrink=shrink)
    Sigma  = _ensure_psd(cov.values)
    mu_ann = returns.mean().values * TRADING_DAYS_PER_YEAR

    w = weights.reindex(returns.columns).fillna(0.0).values
    total = w.sum()
    if total > 0:
        w = w / total

    ann_vol    = float(np.sqrt(max(float(w @ Sigma @ w), 0.0)))
    ann_return = float(mu_ann @ w)
    return ann_vol, ann_return


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
    shrink: bool = True,
    rf_annual: float = 0.04,
) -> pd.DataFrame:
    """
    Compute the efficient frontier by solving MinVar at various target returns.

    Target returns span [min-variance-portfolio return, max-feasible return] so
    the full upper branch of the frontier is always captured and includes the
    max Sharpe portfolio.  Failed optimisation points are silently dropped.

    Returns a DataFrame with columns: Ann. Return, Ann. Volatility, Sharpe.
    """
    returns = returns.dropna()
    if returns.empty or len(returns) < len(returns.columns) + 5:
        return pd.DataFrame()

    cov    = compute_cov_matrix(returns, shrink=shrink)
    Sigma  = _ensure_psd(cov.values)
    mu_ann = returns.mean().values * TRADING_DAYS_PER_YEAR

    if np.any(np.isnan(mu_ann)) or np.any(np.isinf(mu_ann)):
        return pd.DataFrame()

    n      = len(returns.columns)
    bounds = [(min_weight, max_weight)] * n
    w0     = np.ones(n) / n
    eq_sum = {"type": "eq", "fun": lambda w: w.sum() - 1}

    def port_var(w: np.ndarray) -> float:
        return float(w @ Sigma @ w)

    # lower bound: return of the unconstrained-return min-variance portfolio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_mv = minimize(
            port_var, w0, method="SLSQP", bounds=bounds,
            constraints=[eq_sum], options={"ftol": 1e-12, "maxiter": 500},
        )
    min_ret = float(mu_ann @ res_mv.x) if res_mv.success else float(mu_ann.min())

    # upper bound: return of the max-return portfolio (hit by weight constraints)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_mx = minimize(
            lambda w: -float(mu_ann @ w), w0, method="SLSQP", bounds=bounds,
            constraints=[eq_sum], options={"ftol": 1e-12, "maxiter": 500},
        )
    max_ret = float(mu_ann @ res_mx.x) if res_mx.success else float(mu_ann.max())

    targets = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    for target in targets:
        constraints_t = [
            eq_sum,
            {"type": "eq", "fun": lambda w, t=target: float(mu_ann @ w) - t},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                port_var, w0, method="SLSQP", bounds=bounds,
                constraints=constraints_t, options={"ftol": 1e-12, "maxiter": 500},
            )
        if res.success:
            vol = float(np.sqrt(max(res.fun, 0.0)))
            frontier.append({
                "Ann. Return":     target,
                "Ann. Volatility": vol,
                "Sharpe":          (target - rf_annual) / vol if vol > 0 else float("nan"),
            })

    return pd.DataFrame(frontier)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_optimisation(
    model: str,
    returns: pd.DataFrame,
    rf_annual: float = 0.04,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> pd.Series:
    """
    Dispatch to the appropriate model.

    Parameters
    ----------
    model      : one of MODEL_NAMES
    returns    : DataFrame of daily simple returns
    rf_annual  : annual risk-free rate (used only by MaxSharpe)
    """
    if model == "Equal Weight":
        return equal_weight(list(returns.columns))
    elif model == "Minimum Variance":
        return minimum_variance(returns, min_weight, max_weight)
    elif model == "Maximum Sharpe":
        return maximum_sharpe(returns, rf_annual, min_weight, max_weight)
    elif model == "Risk Parity":
        return risk_parity(returns, max(min_weight, 0.01), max_weight)
    else:
        raise ValueError(f"Unknown model: {model!r}. Choose from {MODEL_NAMES}")
