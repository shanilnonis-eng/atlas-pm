"""
conftest.py — shared path setup and synthetic dataset fixtures
for walk-forward backtest validation tests.

All datasets are deterministic (fixed seeds / algebraic construction).
No live market data is used anywhere in this test suite.
"""

from __future__ import annotations

import sys
import os

# Make atlas-pm root importable from any test file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analytics.backtest import run_walk_forward, BacktestResult


# ─── Shared date factory ──────────────────────────────────────────────────────

def biz_dates(n: int, start: str = "2015-01-02") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


# ─── Dataset A: constant returns ──────────────────────────────────────────────
# Every asset returns exactly R per day. Portfolio return = R regardless of weights.
# Expected cumulative wealth after n days: (1 + R)^n

def make_dataset_a(
    n_prices: int = 450,
    r: float = 0.001,
    n_assets: int = 3,
    seed: int = 0,      # unused — algebraic, not random
) -> pd.DataFrame:
    """All assets have identical, constant daily return R."""
    dates = biz_dates(n_prices)
    prices = {
        f"A{i}": (1 + r) ** np.arange(n_prices) * 100.0
        for i in range(n_assets)
    }
    return pd.DataFrame(prices, index=dates)


# ─── Dataset B: regime reversal ───────────────────────────────────────────────
# Asset A: high mean return in training window, low mean in test window.
# Asset B: low mean return in training window, high mean in test window.
# MaxSharpe trained on history should overweight A; then suffer in test.
# Confirms no look-ahead bias: if future data leaked in, weights would favour B.

def make_dataset_b(
    train_window: int = 120,
    test_window:  int = 60,
    r_high: float = 0.003,
    r_low:  float = 0.0001,
    vol:    float = 0.005,
    seed:   int = 42,
) -> pd.DataFrame:
    """Regime reversal: asset rankings flip between training and test window."""
    rng = np.random.default_rng(seed)
    total = train_window + test_window + 10   # small buffer after test
    dates = biz_dates(total + 1)              # +1 because pct_change drops first row

    noise_a = rng.normal(0, vol, total + 1)
    noise_b = rng.normal(0, vol, total + 1)

    # Phase 1 (rows 0 .. train_window-1): A outperforms, B lags
    # Phase 2 (rows train_window .. end):  B outperforms, A lags
    mean_a = np.where(np.arange(total + 1) < train_window, r_high, r_low)
    mean_b = np.where(np.arange(total + 1) < train_window, r_low,  r_high)

    ret_a = mean_a + noise_a
    ret_b = mean_b + noise_b

    price_a = np.cumprod(1 + ret_a) * 100
    price_b = np.cumprod(1 + ret_b) * 100

    return pd.DataFrame({"A": price_a, "B": price_b}, index=dates[: total + 1])


# ─── Dataset C: identical assets ─────────────────────────────────────────────
# Both assets have exactly the same return series.
# Weights should still sum to 1; equal weight is always valid.

def make_dataset_c(n_prices: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = biz_dates(n_prices)
    base = rng.normal(0.0004, 0.01, n_prices)
    prices_base = np.cumprod(1 + base) * 100
    return pd.DataFrame({"A": prices_base, "B": prices_base.copy()}, index=dates)


# ─── Dataset D: low-volatility asset ─────────────────────────────────────────
# Asset LowVol: daily std = 0.002
# Asset HighVol: daily std = 0.02
# Minimum Variance should heavily overweight LowVol (unconstrained).

def make_dataset_d(n_prices: int = 500, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = biz_dates(n_prices)
    ret_low  = rng.normal(0.0003, 0.002, n_prices)
    ret_high = rng.normal(0.0003, 0.020, n_prices)
    p_low  = np.cumprod(1 + ret_low)  * 100
    p_high = np.cumprod(1 + ret_high) * 100
    return pd.DataFrame({"LowVol": p_low, "HighVol": p_high}, index=dates)


# ─── Dataset E: transaction cost / turnover ───────────────────────────────────
# Prices constructed so MaxSharpe flips from favouring A to favouring B
# between the first and second rebalance. Turnover must match oracle.

def make_dataset_e(train_window: int = 100, seed: int = 17) -> pd.DataFrame:
    """
    Phase 1 (rows 0..train_window): A has high Sharpe, B has low Sharpe.
    Phase 2 (rows train_window..2*train_window+buffer): B has high Sharpe.
    """
    rng = np.random.default_rng(seed)
    half = train_window
    total = 3 * half + 10
    dates = biz_dates(total + 1)

    ret_a = np.concatenate([
        rng.normal(0.003, 0.005, half),
        rng.normal(0.0001, 0.005, half + half + 10 + 1),
    ])
    ret_b = np.concatenate([
        rng.normal(0.0001, 0.005, half),
        rng.normal(0.003, 0.005, half + half + 10 + 1),
    ])

    p_a = np.cumprod(1 + ret_a) * 100
    p_b = np.cumprod(1 + ret_b) * 100
    return pd.DataFrame({"A": p_a, "B": p_b}, index=dates[: total + 1])


# ─── Dataset F: explicit date alignment ───────────────────────────────────────
# Monthly business-day prices with clear, traceable dates.

def make_dataset_f(n_prices: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = biz_dates(n_prices, start="2018-01-02")
    rets = rng.normal(0.0002, 0.01, (n_prices, 2))
    prices = np.cumprod(1 + rets, axis=0) * 100
    return pd.DataFrame(prices, index=dates, columns=["X", "Y"])


# ─── Shared pytest fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ew_backtest_small():
    """Equal Weight, small dataset, fast to run."""
    prices = make_dataset_a(n_prices=350, n_assets=2)
    return run_walk_forward(
        prices,
        models=["Equal Weight"],
        train_window=100,
        test_window=50,
        shrink=False,
    )


@pytest.fixture(scope="module")
def multi_model_backtest():
    """All four models on a medium dataset."""
    rng = np.random.default_rng(21)
    n = 500
    dates = biz_dates(n)
    rets = rng.normal(0.0003, 0.012, (n, 3))
    prices = pd.DataFrame(
        np.cumprod(1 + rets, axis=0) * 100,
        index=dates,
        columns=["X", "Y", "Z"],
    )
    return run_walk_forward(
        prices,
        models=["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity"],
        train_window=120,
        test_window=40,
        shrink=False,
        min_weight=0.0,
        max_weight=1.0,
    )
