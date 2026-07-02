"""
Return analytics module for Atlas PM.

Covers all CFA-relevant return metrics:
- Cumulative / total return
- Annualised return (CAGR)
- Annualised volatility
- Sharpe ratio (with proper risk-free adjustment)
- Sortino ratio (downside deviation only)
- Rolling returns and rolling volatility
- Monthly return heatmap data

All functions operate on pandas Series of SIMPLE daily returns unless
otherwise noted.  This keeps the interface consistent and testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR, ROLLING_WINDOW


# ---------------------------------------------------------------------------
# Core scalar metrics
# ---------------------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    """Geometric compounding of daily simple returns → total period return."""
    return float((1 + returns).prod() - 1)


def annualised_return(returns: pd.Series) -> float:
    """
    CAGR-equivalent annualised return.

    Formula: (1 + total_return) ^ (252 / n_days) - 1
    This is the correct annualisation for geometric compounding,
    not the naive arithmetic mean × 252.
    """
    n = len(returns)
    if n == 0:
        return float("nan")
    tr = total_return(returns)
    return float((1 + tr) ** (TRADING_DAYS_PER_YEAR / n) - 1)


def annualised_volatility(returns: pd.Series) -> float:
    """
    Annualised standard deviation of daily returns.

    Uses sample std (ddof=1) then scales by √252.  This is standard
    practice; some providers use ddof=0 — the difference is trivial for
    daily data but worth knowing.
    """
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf_returns: pd.Series | None = None) -> float:
    """
    Ex-post Sharpe ratio: (annualised excess return) / (annualised vol).

    If rf_returns is None, assumes a zero risk-free rate (conservative).
    rf_returns should be on the same daily frequency as returns.
    """
    if rf_returns is not None:
        # align and subtract
        excess = returns - rf_returns.reindex(returns.index).fillna(0)
    else:
        excess = returns

    ann_excess = annualised_return(excess)
    ann_vol    = annualised_volatility(returns)    # vol of total, not excess
    if ann_vol < 1e-12:  # float-safe zero check; exact == 0 fails for constant returns
        return float("nan")
    return float(ann_excess / ann_vol)


def sortino_ratio(returns: pd.Series, rf_returns: pd.Series | None = None) -> float:
    """
    Sortino ratio: (annualised excess return) / (annualised downside deviation).

    Downside deviation uses a MAR (Minimum Acceptable Return) of zero,
    i.e. we only penalise days with negative returns.
    This is the standard CFA / institutional definition.
    """
    if rf_returns is not None:
        excess = returns - rf_returns.reindex(returns.index).fillna(0)
    else:
        excess = returns

    ann_excess   = annualised_return(excess)
    # Use excess returns below zero, not total returns below zero.
    # When rf > 0 the two series differ; the numerator uses excess so the
    # denominator must be consistent.
    negative     = excess[excess < 0]
    if len(negative) == 0:
        return float("inf")
    downside_std = float(negative.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_std == 0:
        return float("nan")
    return float(ann_excess / downside_std)


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return divided by maximum drawdown magnitude."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return float(annualised_return(returns) / abs(mdd))


# ---------------------------------------------------------------------------
# Drawdown analytics
# ---------------------------------------------------------------------------

def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute the drawdown series: at each date, how far is the portfolio
    from its previous peak, expressed as a negative fraction.

    Returns a Series with the same index as returns.
    """
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max
    dd.name = "Drawdown"
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """Maximum (most negative) drawdown over the period. Returns negative value."""
    return float(drawdown_series(returns).min())


def drawdown_duration(returns: pd.Series) -> pd.DataFrame:
    """
    Identify all distinct drawdown episodes.

    Returns a DataFrame with columns:
        start, trough, end, depth, duration_days, recovery_days
    """
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max

    in_drawdown = dd < 0
    episodes = []
    start_idx = None

    for i, (date, val) in enumerate(dd.items()):
        if val < 0 and start_idx is None:
            start_idx = date
        elif val == 0 and start_idx is not None:
            # episode ended
            episode = dd[start_idx:date]
            trough_date = episode.idxmin()
            depth = float(episode.min())
            duration = (date - start_idx).days
            trough_to_recovery = (date - trough_date).days
            episodes.append({
                "Start":         start_idx,
                "Trough":        trough_date,
                "Recovery":      date,
                "Depth":         depth,
                "Duration (days)": duration,
                "Recovery (days)": trough_to_recovery,
            })
            start_idx = None

    # handle open drawdown (not yet recovered)
    if start_idx is not None:
        episode = dd[start_idx:]
        trough_date = episode.idxmin()
        depth = float(episode.min())
        duration = (dd.index[-1] - start_idx).days
        episodes.append({
            "Start":           start_idx,
            "Trough":          trough_date,
            "Recovery":        None,
            "Depth":           depth,
            "Duration (days)": duration,
            "Recovery (days)": None,
        })

    return pd.DataFrame(episodes)


# ---------------------------------------------------------------------------
# Time-series metrics
# ---------------------------------------------------------------------------

def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Wealth index starting at 1.0."""
    cum = (1 + returns).cumprod()
    cum.name = "Cumulative Return"
    return cum


def rolling_volatility(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Annualised rolling volatility over a trailing window."""
    rv = returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    rv.name = f"Rolling Vol ({window}d)"
    return rv


def rolling_sharpe(
    returns: pd.Series,
    rf_returns: pd.Series | None = None,
    window: int = ROLLING_WINDOW,
) -> pd.Series:
    """Rolling Sharpe ratio (annualised) over a trailing window."""
    if rf_returns is not None:
        excess = returns - rf_returns.reindex(returns.index).fillna(0)
    else:
        excess = returns

    roll_mean = excess.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    roll_std  = returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    rs = roll_mean / roll_std
    rs.name = f"Rolling Sharpe ({window}d)"
    return rs


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Pivot table of monthly returns, rows = year, columns = month (Jan…Dec).
    Useful for the classic institutional 'monthly returns heatmap'.
    """
    monthly = (1 + returns).resample("ME").prod() - 1
    df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    pivot = df.pivot(index="Year", columns="Month", values="Return")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]
    # add annual return column
    annual = returns.groupby(returns.index.year).apply(
        lambda r: (1 + r).prod() - 1
    )
    pivot["Annual"] = annual
    return pivot


# ---------------------------------------------------------------------------
# Portfolio-level aggregation
# ---------------------------------------------------------------------------

def portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
) -> pd.Series:
    """
    Compute daily portfolio returns given a weights dict or Series.

    Assumes buy-and-hold (no rebalancing within period).  For rebalanced
    portfolios, call this function on each rebalancing sub-period and
    concatenate.
    """
    if isinstance(weights, dict):
        w = pd.Series(weights)
    else:
        w = weights

    # align
    common_assets = [a for a in w.index if a in asset_returns.columns]
    w = w[common_assets] / w[common_assets].sum()  # normalise to 1
    port = (asset_returns[common_assets] * w).sum(axis=1)
    port.name = "Portfolio"
    return port


def beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """OLS beta of portfolio against benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    cov_mat = aligned.cov()
    b = cov_mat.iloc[0, 1] / cov_mat.iloc[1, 1]
    return float(b)


def alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_returns: pd.Series | None = None,
) -> float:
    """
    Jensen's alpha (annualised): portfolio annualised excess return over
    what CAPM predicts given its beta.

    alpha = ann_port_excess - beta * ann_bench_excess
    """
    b = beta(portfolio_returns, benchmark_returns)
    ann_port  = annualised_return(portfolio_returns)
    ann_bench = annualised_return(benchmark_returns)

    if rf_returns is not None:
        rf_ann = annualised_return(rf_returns.reindex(portfolio_returns.index).fillna(0))
    else:
        rf_ann = 0.0

    return float((ann_port - rf_ann) - b * (ann_bench - rf_ann))


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Information ratio: annualised active return / annualised tracking error.
    Measures skill in generating alpha per unit of active risk.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["port", "bench"]
    active = aligned["port"] - aligned["bench"]
    ann_active    = annualised_return(active)
    tracking_error = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if tracking_error == 0:
        return float("nan")
    return float(ann_active / tracking_error)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_statistics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    rf_returns: pd.Series | None = None,
    label: str = "Portfolio",
) -> dict:
    """Return a dict of all key statistics, suitable for display in a table."""
    stats = {
        "Label":                label,
        "Total Return":         total_return(returns),
        "Ann. Return":          annualised_return(returns),
        "Ann. Volatility":      annualised_volatility(returns),
        "Sharpe Ratio":         sharpe_ratio(returns, rf_returns),
        "Sortino Ratio":        sortino_ratio(returns, rf_returns),
        "Calmar Ratio":         calmar_ratio(returns),
        "Max Drawdown":         max_drawdown(returns),
        "Skewness":             float(returns.skew()),
        "Kurtosis (excess)":    float(returns.kurtosis()),
        "Best Day":             float(returns.max()),
        "Worst Day":            float(returns.min()),
        "% Positive Days":      float((returns > 0).mean()),
        "Observations":         len(returns),
    }
    if benchmark_returns is not None:
        stats["Beta"]               = beta(returns, benchmark_returns)
        stats["Alpha (ann.)"]       = alpha(returns, benchmark_returns, rf_returns)
        stats["Information Ratio"]  = information_ratio(returns, benchmark_returns)
    return stats
