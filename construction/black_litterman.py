"""
Black-Litterman Model for Atlas PM.

The Black-Litterman model (Fischer Black & Robert Litterman, Goldman Sachs, 1990)
solves a fundamental problem with mean-variance optimisation: historical mean
returns are terrible forward-looking estimates, causing optimisers to produce
extreme, unstable portfolios.

BL starts from a 'neutral' equilibrium — the market-implied expected returns —
and then adjusts that equilibrium based on the manager's specific views,
weighted by the manager's confidence in each view.

The result is a posterior expected return vector that:
  1. Starts near the equilibrium (stable baseline)
  2. Tilts toward assets where the manager has high-confidence views
  3. Is fully quantitative — views are explicitly stated and auditable

Mathematical summary
--------------------
Equilibrium returns (reverse-optimisation):
    π = λ × Σ × w_ref
    where λ = risk aversion, Σ = covariance matrix, w_ref = reference weights

Views:
    P × μ = Q + ε,  ε ~ N(0, Ω)
    P: K×N matrix (K views, N assets)
    Q: K×1 view returns
    Ω: K×K diagonal uncertainty matrix

Posterior expected returns (BL formula):
    M = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
    μ_BL = M × [(τΣ)⁻¹π + P'Ω⁻¹Q]

Posterior covariance:
    Σ_BL = Σ + M

Optimal portfolio: MaxSharpe on (μ_BL, Σ_BL)

Reference: Black & Litterman (1992), 'Global Portfolio Optimization',
Financial Analysts Journal, 48(5), 28-43.
He & Litterman (1999), 'The intuition behind Black-Litterman model portfolios',
Goldman Sachs Investment Management Division.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Literal

from config.settings import TRADING_DAYS_PER_YEAR, MIN_WEIGHT, MAX_WEIGHT
from construction.optimiser import (
    compute_cov_matrix, _ensure_psd, equal_weight, ledoit_wolf_shrinkage,
)


# ---------------------------------------------------------------------------
# View dataclass — clean interface for expressing manager views
# ---------------------------------------------------------------------------

@dataclass
class View:
    """
    A single manager view for the Black-Litterman model.

    Attributes
    ----------
    view_type   : 'absolute' | 'relative'
        Absolute: "I expect [asset] to return Q% per year"
        Relative: "I expect [long_asset] to outperform [short_asset] by Q% per year"
    long_asset  : asset label for the long side of the view
    short_asset : asset label for the short side (relative views only)
    view_return : annualised expected return or excess return (e.g. 0.05 = 5%)
    confidence  : 0.0 to 1.0 — manager confidence in this view
                  0.0 = view has no weight; 1.0 = view is treated as certain
    """
    view_type:   Literal["absolute", "relative"]
    long_asset:  str
    short_asset: str | None
    view_return: float
    confidence:  float  # 0 to 1

    def __post_init__(self):
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in (0, 1], got {self.confidence}")
        if self.view_type == "relative" and self.short_asset is None:
            raise ValueError("Relative views require a short_asset")


# ---------------------------------------------------------------------------
# Core BL computation
# ---------------------------------------------------------------------------

class BlackLitterman:
    """
    Black-Litterman model.

    Usage
    -----
    bl = BlackLitterman(returns, reference_weights)
    bl.add_view(View('absolute', 'US Equities', None, 0.10, 0.7))
    bl.add_view(View('relative', 'Gold', 'US Aggregate Bonds', 0.03, 0.5))
    weights = bl.optimal_weights()
    posterior_returns = bl.posterior_returns()
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        reference_weights: pd.Series | None = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        shrink_cov: bool = True,
    ):
        """
        Parameters
        ----------
        returns           : daily simple returns DataFrame
        reference_weights : 'neutral' portfolio weights (defaults to equal weight)
                            Represents the market equilibrium starting point
        risk_aversion     : λ — scales the equilibrium risk premium
                            Typical values: 2.0–3.5
        tau               : uncertainty in equilibrium returns
                            Typical values: 0.025–0.1
                            Smaller τ → equilibrium dominates views more
        shrink_cov        : apply Ledoit-Wolf shrinkage
        """
        self.returns     = returns
        self.assets      = list(returns.columns)
        self.n           = len(self.assets)
        self.risk_aversion = risk_aversion
        self.tau         = tau
        self.views: list[View] = []

        # covariance (annualised)
        cov_df = compute_cov_matrix(returns, shrink=shrink_cov)
        self.Sigma = _ensure_psd(cov_df.values)

        # reference (equilibrium) weights
        if reference_weights is not None:
            w = reference_weights.reindex(self.assets).fillna(0.0).values
            self.w_ref = w / w.sum()
        else:
            self.w_ref = np.ones(self.n) / self.n

    def add_view(self, view: View) -> "BlackLitterman":
        """Add a manager view. Returns self for chaining."""
        if view.long_asset not in self.assets:
            raise ValueError(f"'{view.long_asset}' not in asset universe")
        if view.short_asset and view.short_asset not in self.assets:
            raise ValueError(f"'{view.short_asset}' not in asset universe")
        self.views.append(view)
        return self

    def clear_views(self) -> "BlackLitterman":
        self.views = []
        return self

    # ------------------------------------------------------------------
    # Equilibrium
    # ------------------------------------------------------------------

    def equilibrium_returns(self) -> pd.Series:
        """
        Reverse-optimise the reference portfolio to get implied equilibrium returns.

        π = λ × Σ × w_ref

        These are the expected returns that would make the reference portfolio
        the optimal MaxSharpe portfolio under the given covariance structure.
        """
        pi = self.risk_aversion * self.Sigma @ self.w_ref
        return pd.Series(pi, index=self.assets, name="Equilibrium Returns")

    # ------------------------------------------------------------------
    # View matrices
    # ------------------------------------------------------------------

    def _build_view_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build P (pick matrix), Q (view returns), Ω (uncertainty) from self.views.

        Returns
        -------
        P : (K, N) float array
        Q : (K,)   float array
        Omega : (K, K) diagonal float array
        """
        k = len(self.views)
        asset_idx = {a: i for i, a in enumerate(self.assets)}

        P = np.zeros((k, self.n))
        Q = np.zeros(k)

        for i, view in enumerate(self.views):
            # Sigma is annualised, so Q must also be in annual units
            Q[i] = view.view_return
            li = asset_idx[view.long_asset]
            P[i, li] = 1.0
            if view.view_type == "relative" and view.short_asset:
                si = asset_idx[view.short_asset]
                P[i, si] = -1.0

        # uncertainty matrix Ω (He & Litterman proportional method)
        # Ω_ii = τ × P_i Σ P_i' × (1 - confidence) / confidence
        # Low confidence → high Ω → views have less impact
        omega_diag = np.zeros(k)
        for i, view in enumerate(self.views):
            p_i = P[i, :]
            tau_pSigmaPt = self.tau * float(p_i @ self.Sigma @ p_i)
            c = view.confidence
            omega_diag[i] = tau_pSigmaPt * (1 - c) / c

        Omega = np.diag(omega_diag)
        return P, Q, Omega

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------

    def posterior_returns(self) -> pd.Series:
        """
        Compute the BL posterior expected returns.

        If no views have been added, returns the equilibrium.
        """
        pi = self.equilibrium_returns().values

        if not self.views:
            return pd.Series(pi, index=self.assets, name="BL Posterior Returns")

        P, Q, Omega = self._build_view_matrices()
        tau_Sigma_inv = np.linalg.inv(self.tau * self.Sigma)
        Omega_inv     = np.linalg.inv(Omega)

        # M = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
        M_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
        M     = np.linalg.inv(M_inv)

        # μ_BL = M × [(τΣ)⁻¹π + P'Ω⁻¹Q]
        mu_bl = M @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)

        return pd.Series(mu_bl, index=self.assets, name="BL Posterior Returns")

    def posterior_covariance(self) -> pd.DataFrame:
        """
        Posterior covariance: Σ_BL = Σ + M.

        The M term adds estimation uncertainty — uncertainty is larger in
        directions where you have high-confidence views.
        """
        if not self.views:
            return pd.DataFrame(self.Sigma, index=self.assets, columns=self.assets)

        P, Q, Omega = self._build_view_matrices()
        tau_Sigma_inv = np.linalg.inv(self.tau * self.Sigma)
        Omega_inv     = np.linalg.inv(Omega)
        M_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
        M     = np.linalg.inv(M_inv)

        Sigma_bl = self.Sigma + M
        return pd.DataFrame(Sigma_bl, index=self.assets, columns=self.assets)

    # ------------------------------------------------------------------
    # Optimal weights
    # ------------------------------------------------------------------

    def optimal_weights(
        self,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
    ) -> pd.Series:
        """
        Compute the BL optimal portfolio: MaxSharpe using posterior returns.
        """
        mu_bl    = self.posterior_returns().values
        Sigma_bl = _ensure_psd(self.posterior_covariance().values)

        def neg_sharpe(w):
            ret = float(mu_bl @ w) * TRADING_DAYS_PER_YEAR
            vol = float(np.sqrt(w @ Sigma_bl @ w))
            return -(ret / vol) if vol > 1e-10 else 0.0

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(min_weight, max_weight)] * self.n
        w0 = self.w_ref.copy()

        rng = np.random.default_rng(42)
        best_w, best_obj = w0, float("inf")
        starting_points = [w0] + [rng.dirichlet(np.ones(self.n)) for _ in range(20)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for w_start in starting_points:
                res = minimize(
                    neg_sharpe, w_start,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 1000},
                )
                if res.success and res.fun < best_obj:
                    best_obj = res.fun
                    best_w = res.x

        weights = pd.Series(best_w, index=self.assets, name="Black-Litterman")
        weights = weights.clip(lower=0).div(weights.clip(lower=0).sum())
        return weights

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def view_impact_table(self) -> pd.DataFrame:
        """
        Show how each view shifts expected returns from equilibrium.

        Returns a DataFrame useful for explaining BL to non-quants.
        Sigma is annualised, so equilibrium / posterior returns are already annual.
        """
        eq_returns   = self.equilibrium_returns()   # already annualised
        post_returns = self.posterior_returns()      # already annualised

        rows = []
        for asset in self.assets:
            rows.append({
                "Asset":             asset,
                "Equilibrium Return": eq_returns[asset],
                "BL Posterior Return": post_returns[asset],
                "Shift":             post_returns[asset] - eq_returns[asset],
            })

        return pd.DataFrame(rows).set_index("Asset")

    def summary(self) -> dict:
        """Return a summary dict for display in the UI."""
        # equilibrium_returns() and posterior_returns() are already annualised
        # (Sigma is computed as daily_cov × 252, so π = λΣw is already in annual units).
        # Do NOT multiply by TRADING_DAYS_PER_YEAR again.
        eq = self.equilibrium_returns()
        post = self.posterior_returns()
        w_opt = self.optimal_weights()

        return {
            "n_views":           len(self.views),
            "risk_aversion":     self.risk_aversion,
            "tau":               self.tau,
            "reference_weights": dict(zip(self.assets, self.w_ref)),
            "equilibrium_returns": eq.to_dict(),
            "posterior_returns":   post.to_dict(),
            "optimal_weights":     w_opt.to_dict(),
        }
