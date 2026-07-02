"""
Walk-Forward Backtesting Engine for Atlas PM.

This module provides a rigorous out-of-sample performance evaluation of all
portfolio construction models. It is the honest answer to: "does this optimiser
actually add value, or does it just look good in-sample?"

Methodology
-----------
At each rebalancing date t:
  1. Compute portfolio weights using ONLY prices[t - train_window : t]
  2. Apply those weights to prices[t : t + test_window] (the unseen test period)
  3. Record realised daily returns, turnover, and cost drag
  4. Roll forward by test_window and repeat

No look-ahead bias: training data ends strictly before the first day of the
test period. Weights are a deterministic function of past data only.

Buy-and-hold approximation: within each test window the weights are held
constant. Weight drift within a 21–63 day window is negligible for this
purpose and consistent with institutional quarterly/monthly rebalancing.

Reference: DeMiguel, Garlappi & Uppal (2009) "Optimal versus Naive
Diversification: How Inefficient Is the 1/N Portfolio Strategy?"
Review of Financial Studies, 22(5), 1915–1953.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from construction.optimiser import (
    equal_weight,
    minimum_variance,
    maximum_sharpe,
    risk_parity,
)
from analytics.returns import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    drawdown_series,
)
from analytics.turnover import compute_turnover
from config.settings import TRADING_DAYS_PER_YEAR, MIN_WEIGHT, MAX_WEIGHT


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PeriodResult:
    """Results for one rebalancing period."""
    train_start:   pd.Timestamp
    train_end:     pd.Timestamp
    test_start:    pd.Timestamp
    test_end:      pd.Timestamp
    weights:       pd.Series        # weights applied during test period
    prev_weights:  pd.Series        # weights from previous period (for turnover)
    oos_returns:   pd.Series        # daily portfolio returns during test window
    is_returns:    pd.Series        # daily portfolio returns on the training window
    turnover:      float            # one-way turnover at this rebalance


@dataclass
class BacktestResult:
    """Complete walk-forward results for one model."""
    model:   str
    periods: list[PeriodResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived series (computed from periods)
    # ------------------------------------------------------------------

    @property
    def oos_returns(self) -> pd.Series:
        """Concatenated daily out-of-sample returns across all periods."""
        if not self.periods:
            return pd.Series(dtype=float)
        return pd.concat([p.oos_returns for p in self.periods]).sort_index()

    @property
    def is_returns(self) -> pd.Series:
        """
        Concatenated in-sample returns: portfolio applied to its own training data.
        This overstates what an investor would have seen (it is fitted on the same data
        it is evaluated on) — used only for the degradation comparison.
        """
        if not self.periods:
            return pd.Series(dtype=float)
        return pd.concat([p.is_returns for p in self.periods]).sort_index()

    @property
    def weights_history(self) -> pd.DataFrame:
        """Weights at each rebalancing date, indexed by test_start."""
        if not self.periods:
            return pd.DataFrame()
        rows = {p.test_start: p.weights for p in self.periods}
        return pd.DataFrame(rows).T.fillna(0.0)

    @property
    def turnover_series(self) -> pd.Series:
        return pd.Series(
            [p.turnover for p in self.periods],
            index=[p.test_start for p in self.periods],
            name="One-Way Turnover",
        )

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def oos_summary(self, rf_annual: float = 0.04) -> dict:
        """Out-of-sample performance summary."""
        r = self.oos_returns
        if r.empty:
            return {}
        rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
        rf_s = pd.Series(rf_daily, index=r.index)
        avg_turnover = float(self.turnover_series.iloc[1:].mean())  # exclude initial buy
        return {
            "Model":           self.model,
            "OOS Ann. Return": annualised_return(r),
            "OOS Ann. Vol":    annualised_volatility(r),
            "OOS Sharpe":      sharpe_ratio(r, rf_s),
            "OOS Sortino":     sortino_ratio(r, rf_s),
            "OOS Max DD":      max_drawdown(r),
            "Avg Turnover":    avg_turnover,
            "N Periods":       self.n_periods,
        }

    def is_summary(self, rf_annual: float = 0.04) -> dict:
        """
        In-sample performance summary — average of per-period metrics.

        WHY PER-PERIOD AVERAGE (not concatenated series):
        Training windows overlap by (train_window - test_window) days.
        Concatenating all IS return series creates duplicate timestamps
        (the same calendar date appears in multiple periods with different
        weights). Computing Sharpe on a series with duplicate timestamps
        biases IS Sharpe upward because pandas includes every duplicate row
        in mean and std calculations, effectively over-weighting the overlap
        period. Per-period averaging avoids this entirely.
        """
        if not self.periods:
            return {}

        rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
        per_sharpes  = []
        per_returns  = []
        per_vols     = []
        per_dds      = []

        for p in self.periods:
            r = p.is_returns
            if r.empty or len(r) < 2:
                continue
            rf_s = pd.Series(rf_daily, index=r.index)
            sr   = sharpe_ratio(r, rf_s)
            if np.isfinite(sr):
                per_sharpes.append(sr)
            ar = annualised_return(r)
            if np.isfinite(ar):
                per_returns.append(ar)
            av = annualised_volatility(r)
            if np.isfinite(av):
                per_vols.append(av)
            dd = max_drawdown(r)
            if np.isfinite(dd):
                per_dds.append(dd)

        def _mean(lst):
            return float(np.mean(lst)) if lst else float("nan")

        return {
            "Model":          self.model,
            "IS Ann. Return": _mean(per_returns),
            "IS Ann. Vol":    _mean(per_vols),
            "IS Sharpe":      _mean(per_sharpes),
            "IS Max DD":      float(np.min(per_dds)) if per_dds else float("nan"),
        }

    def degradation(self, rf_annual: float = 0.04) -> dict:
        """IS vs OOS Sharpe degradation."""
        oos = self.oos_summary(rf_annual)
        is_ = self.is_summary(rf_annual)
        if not oos or not is_:
            return {}
        return {
            "Model":           self.model,
            "IS Sharpe":       is_["IS Sharpe"],
            "OOS Sharpe":      oos["OOS Sharpe"],
            "Degradation":     is_["IS Sharpe"] - oos["OOS Sharpe"],
        }


# ---------------------------------------------------------------------------
# Core walk-forward engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    prices: pd.DataFrame,
    models: list[str],
    train_window: int = 756,     # ~3 years of trading days
    test_window:  int = 63,      # ~1 quarter
    rf_annual:    float = 0.04,
    min_weight:   float = MIN_WEIGHT,
    max_weight:   float = MAX_WEIGHT,
    shrink:       bool  = True,
) -> dict[str, BacktestResult]:
    """
    Run walk-forward backtest for one or more portfolio construction models.

    Parameters
    ----------
    prices       : adjusted close price DataFrame, columns = asset labels
    models       : list of model names from ["Equal Weight", "Minimum Variance",
                   "Maximum Sharpe", "Risk Parity"]
    train_window : number of trading days to use as training history
    test_window  : number of trading days in each out-of-sample test period
    rf_annual    : annualised risk-free rate (used by Maximum Sharpe)
    min_weight   : minimum per-asset weight constraint
    max_weight   : maximum per-asset weight constraint
    shrink       : apply Ledoit-Wolf shrinkage to covariance estimates

    Returns
    -------
    dict mapping model name → BacktestResult

    Raises
    ------
    ValueError if there is insufficient data to run even one test period.
    """
    simple_returns = prices.pct_change().dropna()
    n = len(simple_returns)

    min_required = train_window + test_window
    if n < min_required:
        raise ValueError(
            f"Insufficient data: {n} observations, need ≥ {min_required} "
            f"(train_window={train_window} + test_window={test_window})."
        )

    # Index positions of each test-period start
    test_start_indices = list(range(train_window, n, test_window))

    results: dict[str, BacktestResult] = {m: BacktestResult(model=m) for m in models}

    for model in models:
        prev_weights: pd.Series | None = None

        for t in test_start_indices:
            # ── Training window ───────────────────────────────────────────
            # Strictly historical: indices [t - train_window, t)
            train_rets = simple_returns.iloc[t - train_window : t]

            # ── Compute weights ───────────────────────────────────────────
            # All information used here is from the training window only.
            weights = _compute_weights(
                model, train_rets, rf_annual, min_weight, max_weight, shrink
            )

            # ── In-sample portfolio returns ───────────────────────────────
            # Apply these weights to the training data (for degradation analysis).
            is_port = _apply_weights(train_rets, weights)

            # ── Test window ───────────────────────────────────────────────
            # Indices [t, min(t + test_window, n))
            test_end   = min(t + test_window, n)
            test_rets  = simple_returns.iloc[t : test_end]
            oos_port   = _apply_weights(test_rets, weights)

            # ── Turnover ──────────────────────────────────────────────────
            if prev_weights is None:
                turnover = 1.0  # first investment: full portfolio deployed
            else:
                turnover = compute_turnover(prev_weights, weights)

            # ── Store period ──────────────────────────────────────────────
            period = PeriodResult(
                train_start  = simple_returns.index[t - train_window],
                train_end    = simple_returns.index[t - 1],
                test_start   = simple_returns.index[t],
                test_end     = simple_returns.index[test_end - 1],
                weights      = weights,
                prev_weights = prev_weights if prev_weights is not None else weights,
                oos_returns  = oos_port,
                is_returns   = is_port,
                turnover     = turnover,
            )
            results[model].periods.append(period)
            prev_weights = weights

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_weights(
    model:      str,
    returns:    pd.DataFrame,
    rf_annual:  float,
    min_weight: float,
    max_weight: float,
    shrink:     bool,
) -> pd.Series:
    """Dispatch to the correct optimiser and return weights."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "Equal Weight":
            return equal_weight(list(returns.columns))
        elif model == "Minimum Variance":
            return minimum_variance(returns, min_weight, max_weight, shrink)
        elif model == "Maximum Sharpe":
            return maximum_sharpe(returns, rf_annual, min_weight, max_weight, shrink)
        elif model == "Risk Parity":
            return risk_parity(returns, max(min_weight, 0.01), max_weight, shrink)
        else:
            raise ValueError(f"Unknown model: {model!r}")


def _apply_weights(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Compute daily portfolio returns for given weights (buy-and-hold, constant weights).

    Only assets present in both returns.columns and weights.index are used.
    Weights are re-normalised to sum to 1 over common assets.
    """
    common = [a for a in weights.index if a in returns.columns]
    w = weights[common]
    w = w / w.sum()
    port = (returns[common] * w).sum(axis=1)
    port.name = "Portfolio"
    return port


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

def build_summary_table(
    results: dict[str, BacktestResult],
    benchmark_returns: pd.Series | None = None,
    rf_annual: float = 0.04,
) -> pd.DataFrame:
    """
    Build a consolidated out-of-sample summary DataFrame for all models.

    Optionally includes benchmark statistics (computed over the same OOS period).
    """
    rows = []
    for model, result in results.items():
        rows.append(result.oos_summary(rf_annual))

    if benchmark_returns is not None and rows:
        first_result = next(iter(results.values()))
        if not first_result.oos_returns.empty:
            oos_idx = first_result.oos_returns.index
            bench   = benchmark_returns.reindex(oos_idx).dropna()
            if not bench.empty:
                rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
                rf_s = pd.Series(rf_daily, index=bench.index)
                rows.append({
                    "Model":           "Benchmark (Buy & Hold)",
                    "OOS Ann. Return": annualised_return(bench),
                    "OOS Ann. Vol":    annualised_volatility(bench),
                    "OOS Sharpe":      sharpe_ratio(bench, rf_s),
                    "OOS Sortino":     sortino_ratio(bench, rf_s),
                    "OOS Max DD":      max_drawdown(bench),
                    "Avg Turnover":    0.0,
                    "N Periods":       None,
                })

    df = pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()
    return df


def build_degradation_table(
    results: dict[str, BacktestResult],
    rf_annual: float = 0.04,
) -> pd.DataFrame:
    """
    Build IS vs OOS Sharpe degradation table.

    A large positive degradation means the model looked much better in-sample
    than it delivered out-of-sample — a sign of over-fitting / estimation error.
    Equal Weight should show the lowest degradation because it uses no
    estimated parameters.
    """
    rows = [result.degradation(rf_annual) for result in results.values()]
    df = pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()
    return df
