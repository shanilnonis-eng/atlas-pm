"""
Active Share and Tracking Error module for Atlas PM.

Active Share measures how different a portfolio is from its benchmark.
It was introduced by Cremers & Petajisto (2009) 'How Active Is Your Fund Manager?'
and is now one of the most widely cited metrics in institutional asset management.

Formula:
    Active Share = 0.5 × Σ_i |w_i^portfolio - w_i^benchmark|

where the sum runs over the union of all assets in either portfolio or benchmark,
and missing assets are treated as having weight 0.

Properties:
    - AS = 0  →  portfolio is identical to benchmark (pure index fund)
    - AS = 1  →  portfolio has no overlap with benchmark (completely different)
    - AS is always in [0, 1] when both weights sum to 1

Tracking Error:
    TE = annualised standard deviation of (portfolio returns − benchmark returns)
    TE measures how different the return *behaviour* is, not just the weights.

The two metrics together characterise how a manager adds (or fails to add) value:
    High AS + High TE  →  genuine active management (stock picking / factor bets)
    High AS + Low TE   →  diversified factor bets (different weights, correlated returns)
    Low  AS + Low  TE  →  closet indexing (benchmark-hugging)

Reference:
    Cremers, K.J.M. and Petajisto, A. (2009) 'How Active Is Your Fund Manager?
    A New Measure That Predicts Performance', Review of Financial Studies, 22(9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tracking error
# ---------------------------------------------------------------------------

def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Annualised tracking error: std(active returns) × sqrt(252).

    Active return = portfolio return − benchmark return (daily).
    The benchmark series is aligned to the portfolio's DatetimeIndex;
    days with no benchmark return are filled with 0.

    Returns
    -------
    Positive float (fraction), e.g. 0.08 = 8 % annualised TE.
    """
    bench_aligned = benchmark_returns.reindex(portfolio_returns.index).fillna(0.0)
    active = portfolio_returns - bench_aligned
    te = float(active.std(ddof=1) * np.sqrt(252))
    return max(te, 0.0)


# ---------------------------------------------------------------------------
# Active Share
# ---------------------------------------------------------------------------

def calculate_active_share(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
) -> float:
    """
    Active Share = 0.5 × Σ |w_portfolio_i - w_benchmark_i|

    Both series are normalised to sum to 1 before computation.
    Missing assets are implicitly 0-weighted.

    Parameters
    ----------
    portfolio_weights : pd.Series, index = asset labels
    benchmark_weights : pd.Series, index = asset labels

    Returns
    -------
    Float in [0, 1].  0 = identical to benchmark.  1 = no overlap.
    """
    # Union of all asset labels — missing assets default to 0
    all_assets = portfolio_weights.index.union(benchmark_weights.index)
    w_p = portfolio_weights.reindex(all_assets).fillna(0.0)
    w_b = benchmark_weights.reindex(all_assets).fillna(0.0)

    # Normalise defensively (weights should already sum to 1)
    if w_p.sum() > 1e-10:
        w_p = w_p / w_p.sum()
    if w_b.sum() > 1e-10:
        w_b = w_b / w_b.sum()

    return float(0.5 * (w_p - w_b).abs().sum())


# ---------------------------------------------------------------------------
# Benchmark weight construction
# ---------------------------------------------------------------------------

def build_benchmark_weights(portfolio_weights: pd.Series) -> pd.Series:
    """
    Build the benchmark weight vector for the default benchmark.

    The benchmark ticker is read from config.settings (BENCHMARK_TICKER = 'SPY').
    The corresponding universe label (e.g. 'US Equities (S&P 500)') is given
    weight 1.0; all other labels get 0.0.

    The returned Series may contain labels not present in portfolio_weights
    (if the benchmark asset was not selected by the user) — this is intentional;
    calculate_active_share handles the union correctly.

    Returns
    -------
    pd.Series  — benchmark weights (sums to 1.0 exactly).
    """
    from config.settings import UNIVERSE, BENCHMARK_TICKER

    bench_label = next(
        (label for label, ticker in UNIVERSE.items() if ticker == BENCHMARK_TICKER),
        None,
    )

    if bench_label is not None:
        return pd.Series({bench_label: 1.0}, name="Benchmark")

    # Fallback: equal weight across portfolio assets (no matching benchmark found)
    n = len(portfolio_weights)
    if n == 0:
        return pd.Series(dtype=float, name="Benchmark")
    return pd.Series(1.0 / n, index=portfolio_weights.index, name="Benchmark")


# ---------------------------------------------------------------------------
# Active weight breakdown
# ---------------------------------------------------------------------------

def active_weight_breakdown(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
) -> pd.DataFrame:
    """
    Per-asset breakdown of portfolio vs benchmark weights and active positions.

    Active Weight = Portfolio Weight - Benchmark Weight
    Positive = overweight vs benchmark (active long bet)
    Negative = underweight vs benchmark (active short/no-hold)

    Returns
    -------
    DataFrame with columns:
        Asset, Portfolio Weight, Benchmark Weight, Active Weight
    Sorted by Active Weight descending (largest overweights first).
    """
    all_assets = portfolio_weights.index.union(benchmark_weights.index)
    w_p = portfolio_weights.reindex(all_assets).fillna(0.0)
    w_b = benchmark_weights.reindex(all_assets).fillna(0.0)

    if w_p.sum() > 1e-10:
        w_p = w_p / w_p.sum()
    if w_b.sum() > 1e-10:
        w_b = w_b / w_b.sum()

    df = pd.DataFrame({
        "Asset":             all_assets.tolist(),
        "Portfolio Weight":  w_p.values,
        "Benchmark Weight":  w_b.values,
        "Active Weight":     (w_p - w_b).values,
    })
    return df.sort_values("Active Weight", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# TE thresholds (annualised) used in the 2-by-2 quadrant
TE_THRESHOLD_LOW  = 0.04   #  4 % — boundary between low / moderate TE
TE_THRESHOLD_HIGH = 0.08   #  8 % — boundary between moderate / high TE

# Active Share thresholds (Cremers & Petajisto 2009, adapted for multi-asset)
AS_CLOSET_INDEX  = 0.20    # < 20 % → closet indexing
AS_MODERATE      = 0.60    # 20–60 % → moderately active
AS_GENUINE       = 0.80    # 60–80 % → genuinely active  |  > 80 % → high conviction


def active_share_classification(active_share: float) -> dict:
    """
    Classify a portfolio based on its Active Share.

    Returns a dict with keys: label, description, color.

    Notes
    -----
    Thresholds are adapted from Cremers & Petajisto (2009) for equity funds.
    Multi-asset portfolios vs equity-only benchmarks are expected to have
    higher Active Share since bonds, gold, and commodities are not in SPY.
    """
    if active_share >= AS_GENUINE:
        return {
            "label":       "High Conviction",
            "description": (
                "Portfolio deviates substantially from the benchmark. "
                "Large active positions across multiple assets — "
                "high potential for excess return and elevated tracking error."
            ),
            "color": "#2ecc71",
        }
    elif active_share >= AS_MODERATE:
        return {
            "label":       "Genuinely Active",
            "description": (
                "Portfolio meaningfully differs from the benchmark. "
                "Consistent with an active multi-asset mandate."
            ),
            "color": "#f4a261",
        }
    elif active_share >= AS_CLOSET_INDEX:
        return {
            "label":       "Moderately Active",
            "description": (
                "Portfolio has active positions but retains significant "
                "benchmark overlap. Consider whether active management fees "
                "are justified by the degree of differentiation."
            ),
            "color": "#e67e22",
        }
    else:
        return {
            "label":       "Closet Indexer",
            "description": (
                "Portfolio closely mirrors the benchmark. "
                "Investors are effectively paying active fees for near-passive exposure."
            ),
            "color": "#e84855",
        }


def te_classification(tracking_error: float) -> str:
    """Return a short label for the tracking error regime."""
    if tracking_error >= TE_THRESHOLD_HIGH:
        return "High TE"
    elif tracking_error >= TE_THRESHOLD_LOW:
        return "Moderate TE"
    else:
        return "Low TE"


def quadrant_label(active_share: float, tracking_error: float) -> str:
    """
    Return the Cremers & Petajisto 2×2 quadrant label.

    High AS + High TE  →  Stock Picker / Active Allocator
    High AS + Low TE   →  Diversified Factor Bets
    Low  AS + Low  TE  →  Closet Indexer
    Low  AS + High TE  →  (unusual — similar weights but different returns)
    """
    high_as = active_share >= AS_MODERATE
    high_te = tracking_error >= TE_THRESHOLD_LOW

    if high_as and high_te:
        return "Active Allocator"
    elif high_as and not high_te:
        return "Diversified Factor Bets"
    elif not high_as and not high_te:
        return "Closet Indexer"
    else:
        return "Selective Active"
