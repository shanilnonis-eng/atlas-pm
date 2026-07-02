"""
Data loading and caching layer for Atlas PM.

Design decisions:
- All price data is fetched as adjusted closing prices (accounts for splits,
  dividends) using yfinance.
- Results are cached in-memory via st.cache_data so repeated Streamlit
  interactions don't hit Yahoo Finance on every re-run.
- We return DAILY LOG RETURNS, not prices, because log returns are additive
  across time and better behaved statistically.  Simple returns are also
  returned for compounding calculations.
- Missing data (e.g. ISF.L not trading on a US holiday) is forward-filled
  then any leading NaNs are dropped, so all assets share a common date index.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import streamlit as st
    _STREAMLIT = True
except ImportError:
    _STREAMLIT = False

from config.settings import (
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    BENCHMARK_TICKER,
    RISK_FREE_TICKER,
    UNIVERSE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_raw(tickers: list[str], start: str, end: Optional[str]) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # single ticker returns flat columns
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return prices


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill gaps then drop any dates where ANY ticker is still NaN."""
    prices = prices.ffill()
    prices = prices.dropna()
    return prices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_prices(
    assets: dict[str, str],
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """
    Return clean adjusted close prices for the given asset universe.

    Parameters
    ----------
    assets : dict mapping label → ticker, e.g. {"US Equities": "SPY"}
    start  : ISO date string
    end    : ISO date string or None (→ today)

    Returns
    -------
    DataFrame with DatetimeIndex, columns = asset labels (not tickers)
    """
    ticker_to_label = {v: k for k, v in assets.items()}
    tickers = list(assets.values())

    prices = _fetch_raw(tickers, start, end)
    prices = _clean_prices(prices)
    prices = prices.rename(columns=ticker_to_label)

    # keep only requested columns (some tickers may have been dropped)
    available = [c for c in prices.columns if c in assets]
    return prices[available]


def load_benchmark(
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = DEFAULT_END_DATE,
) -> pd.Series:
    """Return benchmark adjusted close prices as a named Series."""
    prices = _fetch_raw([BENCHMARK_TICKER], start, end)
    series = prices[BENCHMARK_TICKER].dropna()
    series.name = "Benchmark"
    return series


def load_risk_free(
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = DEFAULT_END_DATE,
) -> pd.Series:
    """Return risk-free proxy (T-Bill ETF) adjusted close prices."""
    prices = _fetch_raw([RISK_FREE_TICKER], start, end)
    series = prices[RISK_FREE_TICKER].dropna()
    series.name = "Risk-Free"
    return series


def compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute simple and log returns from a prices DataFrame.

    Returns
    -------
    simple_returns : (P_t / P_{t-1}) - 1
    log_returns    : ln(P_t / P_{t-1})
    Both have the same index/columns as prices, with the first row dropped.
    """
    simple = prices.pct_change().dropna()
    log    = np.log(prices / prices.shift(1)).dropna()
    return simple, log


def align_series(
    portfolio_prices: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align portfolio and benchmark to a shared date index."""
    combined = portfolio_prices.join(benchmark_prices, how="inner")
    bench    = combined["Benchmark"]
    port     = combined.drop(columns=["Benchmark"])
    return port, bench


# ---------------------------------------------------------------------------
# Streamlit-cached wrappers (used by the UI layer)
# ---------------------------------------------------------------------------

if _STREAMLIT:
    @st.cache_data(ttl=3600, show_spinner="Fetching market data…")
    def cached_load_prices(assets_json: str, start: str, end: Optional[str]) -> pd.DataFrame:
        """
        Streamlit-cache-safe wrapper.  We serialise the dict as JSON because
        st.cache_data requires hashable arguments.
        """
        import json
        assets = json.loads(assets_json)
        return load_prices(assets, start, end)

    @st.cache_data(ttl=3600, show_spinner="Fetching benchmark data…")
    def cached_load_benchmark(start: str, end: Optional[str]) -> pd.Series:
        return load_benchmark(start, end)

    @st.cache_data(ttl=3600, show_spinner="Fetching risk-free rate proxy…")
    def cached_load_risk_free(start: str, end: Optional[str]) -> pd.Series:
        return load_risk_free(start, end)
