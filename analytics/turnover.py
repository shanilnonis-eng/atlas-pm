"""
Transaction cost and turnover analysis for Atlas PM.

Institutional portfolio managers care deeply about implementation costs.
A strategy that looks great on paper can be significantly eroded by:
  - Bid-offer spreads (most important for ETFs)
  - Market impact (function of trade size vs daily volume)
  - Rebalancing frequency (more rebalancing = more cost drag)

This module models these costs and produces cost-adjusted performance metrics,
giving a more honest view of what the portfolio would actually achieve.

Methodology
-----------
Cost drag per rebalance:
    turnover = Σ_i |w_i_new - w_i_old| / 2  (one-way turnover)
    cost = turnover × spread_pct

Annual cost drag = cost_per_rebalance × rebalances_per_year

ETF bid-offer spreads (approximate, as of 2024):
    Liquid US ETFs (SPY, AGG): 1–2 bps
    Less liquid (PDBC, REET):  3–8 bps
    GBP-listed (ISF.L):        3–5 bps

These are approximations. Real costs depend on trade size, time of day,
market conditions, and the specific broker.

Limitations explicitly documented:
  - Market impact not modelled (assumes institutional but not fund-scale trading)
  - Tax drag not modelled
  - FX hedging costs not modelled
  - Bid-offer spreads are approximate and time-varying
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from config.settings import TRADING_DAYS_PER_YEAR, UNIVERSE


# ---------------------------------------------------------------------------
# Estimated bid-offer spreads by ETF (in basis points, 1 bp = 0.0001)
# ---------------------------------------------------------------------------

ETF_SPREADS_BPS: dict[str, float] = {
    "SPY":   0.5,    # most liquid ETF in the world
    "AGG":   1.0,
    "GLD":   1.5,
    "EEM":   2.5,
    "ISF.L": 4.0,    # less liquid; GBP-denominated
    "IEUR":  3.0,
    "BNDW":  2.0,
    "PDBC":  6.0,    # commodity ETF; wider spread
    "REET":  5.0,
    "BIL":   0.5,    # T-bill; nearly cash
}

DEFAULT_SPREAD_BPS = 5.0  # conservative fallback for unknown tickers


def get_spread(asset_label: str, universe: dict[str, str] = UNIVERSE) -> float:
    """
    Return the estimated bid-offer spread in basis points for an asset.

    Looks up by ticker in ETF_SPREADS_BPS.
    """
    ticker = universe.get(asset_label, "")
    return ETF_SPREADS_BPS.get(ticker, DEFAULT_SPREAD_BPS) / 10000  # convert to fraction


# ---------------------------------------------------------------------------
# Turnover calculations
# ---------------------------------------------------------------------------

def compute_turnover(
    weights_before: pd.Series,
    weights_after:  pd.Series,
) -> float:
    """
    One-way portfolio turnover from one set of weights to another.

    Turnover = Σ |w_i_after - w_i_before| / 2

    Dividing by 2 gives one-way turnover (the fraction of the portfolio traded).
    A turnover of 0.10 means 10% of the portfolio was repositioned.
    """
    all_assets = weights_before.index.union(weights_after.index)
    w_before = weights_before.reindex(all_assets).fillna(0.0)
    w_after  = weights_after.reindex(all_assets).fillna(0.0)
    return float((w_before - w_after).abs().sum() / 2)


def rebalancing_cost(
    weights_before: pd.Series,
    weights_after:  pd.Series,
    universe: dict[str, str] = UNIVERSE,
) -> dict:
    """
    Estimate the total cost of rebalancing from one set of weights to another.

    Returns a dict with total cost (as fraction of NAV) and asset-level detail.
    """
    all_assets = weights_before.index.union(weights_after.index)
    w_before = weights_before.reindex(all_assets).fillna(0.0)
    w_after  = weights_after.reindex(all_assets).fillna(0.0)

    detail = []
    total_cost = 0.0

    for asset in all_assets:
        trade    = abs(w_after[asset] - w_before[asset])
        spread   = get_spread(asset, universe)
        cost     = trade * spread / 2  # only pay half-spread per side
        total_cost += cost
        detail.append({
            "Asset":         asset,
            "Trade Size":    w_after[asset] - w_before[asset],
            "Spread (bps)":  spread * 10000,
            "Cost (bps)":    cost * 10000,
        })

    return {
        "total_cost":        total_cost,
        "total_cost_bps":    total_cost * 10000,
        "one_way_turnover":  compute_turnover(weights_before, weights_after),
        "detail":            pd.DataFrame(detail),
    }


# ---------------------------------------------------------------------------
# Rebalancing simulation
# ---------------------------------------------------------------------------

def simulate_rebalancing(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    rebal_freq: str = "ME",   # pandas offset alias: ME=month-end, QE=quarter-end
    universe: dict[str, str] = UNIVERSE,
    initial_value: float = 1_000_000,
) -> dict:
    """
    Simulate a buy-and-hold portfolio with periodic rebalancing back to target weights,
    including realistic transaction costs.

    Returns gross (no-cost) and net (after-cost) cumulative return series,
    plus detailed rebalancing log.

    Parameters
    ----------
    prices          : adjusted close prices DataFrame
    target_weights  : constant target weights (static strategy)
    rebal_freq      : 'ME' (monthly), 'QE' (quarterly), 'YE' (annual)
    universe        : asset label → ticker dict for spread lookup
    initial_value   : portfolio starting value in £
    """
    # align weights to available assets
    available = [a for a in target_weights.index if a in prices.columns]
    w = target_weights[available] / target_weights[available].sum()
    px = prices[available].copy()

    simple_returns = px.pct_change().fillna(0)

    # identify rebalancing dates
    rebal_dates = set(
        simple_returns.resample(rebal_freq).last().index.normalize()
    )

    # simulation state
    portfolio_value = initial_value
    current_weights = w.copy()  # start with target weights
    gross_values = []
    net_values   = []
    rebal_log    = []

    gross_value = initial_value
    net_value   = initial_value

    dates = simple_returns.index
    for i, date in enumerate(dates):
        day_returns = simple_returns.iloc[i]

        # drift weights
        new_weights = current_weights * (1 + day_returns[current_weights.index])
        total = new_weights.sum()
        if total > 0:
            new_weights = new_weights / total

        # update values
        day_gross_return = float((current_weights * day_returns[current_weights.index]).sum())
        gross_value *= (1 + day_gross_return)
        net_value   *= (1 + day_gross_return)

        current_weights = new_weights

        # rebalance if on a rebalancing date
        if date.normalize() in rebal_dates:
            cost_info = rebalancing_cost(current_weights, w, universe)
            cost = cost_info["total_cost"]
            net_value   *= (1 - cost)
            rebal_log.append({
                "Date":             date,
                "Turnover":         cost_info["one_way_turnover"],
                "Cost (bps)":       cost_info["total_cost_bps"],
                "Portfolio Value":  net_value,
            })
            current_weights = w.copy()  # reset to target

        gross_values.append(gross_value)
        net_values.append(net_value)

    gross_series = pd.Series(gross_values, index=dates, name="Gross")
    net_series   = pd.Series(net_values,   index=dates, name="Net of Costs")
    rebal_df     = pd.DataFrame(rebal_log)

    gross_returns = gross_series.pct_change().dropna()
    net_returns   = net_series.pct_change().dropna()

    # cost drag
    total_cost_drag = float((gross_series.iloc[-1] - net_series.iloc[-1]) / gross_series.iloc[-1])
    annual_cost_drag = total_cost_drag / (len(dates) / TRADING_DAYS_PER_YEAR)

    return {
        "gross_wealth":       gross_series / initial_value,
        "net_wealth":         net_series   / initial_value,
        "gross_returns":      gross_returns,
        "net_returns":        net_returns,
        "rebalancing_log":    rebal_df,
        "total_cost_drag":    total_cost_drag,
        "annual_cost_drag":   annual_cost_drag,
        "n_rebalances":       len(rebal_df),
        "avg_turnover":       float(rebal_df["Turnover"].mean()) if not rebal_df.empty else 0,
        "avg_cost_bps":       float(rebal_df["Cost (bps)"].mean()) if not rebal_df.empty else 0,
    }


def turnover_comparison(
    prices: pd.DataFrame,
    weights: pd.Series,
    universe: dict[str, str] = UNIVERSE,
) -> pd.DataFrame:
    """
    Compare gross vs net performance at monthly, quarterly, and annual rebalancing.
    """
    from analytics.returns import annualised_return, annualised_volatility, sharpe_ratio

    rows = []
    for freq, label in [("ME","Monthly"), ("QE","Quarterly"), ("YE","Annual")]:
        result = simulate_rebalancing(prices, weights, freq, universe)
        gr  = result["gross_returns"]
        nr  = result["net_returns"]
        rows.append({
            "Rebalancing":        label,
            "Gross Ann. Return":  annualised_return(gr),
            "Net Ann. Return":    annualised_return(nr),
            "Annual Cost Drag":   result["annual_cost_drag"],
            "Avg Cost/Rebal (bps)": result["avg_cost_bps"],
            "Avg Turnover":       result["avg_turnover"],
            "# Rebalances":       result["n_rebalances"],
        })

    return pd.DataFrame(rows)
