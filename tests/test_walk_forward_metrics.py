"""
test_walk_forward_metrics.py
------------------------------
Validates OOS performance metrics: Sharpe, volatility, annualised return,
drawdown, and their annualisation. All computed against independent oracles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from analytics.backtest import run_walk_forward, BacktestResult
from analytics.returns import (
    annualised_return, annualised_volatility, sharpe_ratio, max_drawdown
)
from tests.conftest import make_dataset_a, biz_dates

TRAIN = 100
TEST  = 40
TRADING_DAYS = 252


def _run(prices, models=None, train=TRAIN, test=TEST, rf=0.04):
    if models is None:
        models = ["Equal Weight"]
    return run_walk_forward(
        prices, models, train_window=train, test_window=test,
        rf_annual=rf, shrink=False, min_weight=0.0, max_weight=1.0,
    )


class TestOOSMetricsOracle:

    def test_oos_annualised_return_oracle(self):
        """
        Oracle: annualised_return = (1 + total_return)^(252/n) - 1.
        Applied to the concatenated OOS return series independently.
        """
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices)
        for model, result in results.items():
            oos   = result.oos_returns
            app   = result.oos_summary(rf_annual=0.04)["OOS Ann. Return"]
            oracle = annualised_return(oos)       # independent call
            assert_allclose(app, oracle, rtol=1e-9)

    def test_oos_annualised_volatility_oracle(self):
        """
        Oracle: ann_vol = std(r, ddof=1) * sqrt(252).
        Computed from returns, not prices.
        """
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices)
        for model, result in results.items():
            oos    = result.oos_returns
            app    = result.oos_summary()["OOS Ann. Vol"]
            oracle = float(oos.std(ddof=1) * np.sqrt(TRADING_DAYS))
            assert_allclose(app, oracle, rtol=1e-9)

    def test_oos_vol_from_returns_not_prices(self):
        """
        Volatility must be computed from percentage RETURNS, not raw price levels.
        If computed from prices, vol would scale with price level (nonsensical).
        """
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices)
        for model, result in results.items():
            oos_r = result.oos_returns
            app_vol = result.oos_summary()["OOS Ann. Vol"]

            # If computed from returns: typical range 0-50% annual
            # If computed from prices: would be enormous (prices at 100+)
            assert app_vol < 5.0, (
                f"{model}: ann vol = {app_vol:.2f}, suspiciously large — "
                f"may be computing from price levels instead of returns"
            )

    def test_oos_sharpe_oracle(self):
        """
        Oracle: Sharpe = annualised_return(excess) / annualised_vol(returns).
        Computed independently from OOS return series and rf.
        """
        rf_annual = 0.04
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices, rf=rf_annual)
        for model, result in results.items():
            oos   = result.oos_returns
            app   = result.oos_summary(rf_annual=rf_annual)["OOS Sharpe"]
            rf_s  = pd.Series(rf_annual / TRADING_DAYS, index=oos.index)
            oracle = sharpe_ratio(oos, rf_s)
            assert_allclose(app, oracle, rtol=1e-9)

    def test_oos_sharpe_annualisation_uses_252(self):
        """
        Verify the Sharpe formula uses 252 trading days, not 365 calendar days.
        Oracle: compute Sharpe with 252 vs 365, both should differ — confirm
        app matches 252.
        """
        rng = np.random.default_rng(31)
        n = 400
        dates = biz_dates(n)
        rets = rng.normal(0.0005, 0.012, (n, 2))
        prices = pd.DataFrame(np.cumprod(1 + rets, axis=0) * 100,
                               index=dates, columns=["A", "B"])
        results = _run(prices)
        oos = results["Equal Weight"].oos_returns
        rf_s = pd.Series(0.04 / 252, index=oos.index)

        ann_exc = annualised_return(oos - rf_s.values)
        oracle_252 = ann_exc / float(oos.std(ddof=1) * np.sqrt(252))
        oracle_365 = ann_exc / float(oos.std(ddof=1) * np.sqrt(365))

        app_sharpe = results["Equal Weight"].oos_summary()["OOS Sharpe"]

        diff_252 = abs(app_sharpe - oracle_252)
        diff_365 = abs(app_sharpe - oracle_365)

        assert diff_252 < diff_365, (
            f"Sharpe appears to use 365 rather than 252 trading days. "
            f"|app - oracle_252| = {diff_252:.6f}, |app - oracle_365| = {diff_365:.6f}"
        )

    def test_oos_max_drawdown_oracle(self):
        """
        Oracle: max_drawdown = min((wealth - peak) / peak).
        Computed from OOS wealth series independently.
        """
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices)
        for model, result in results.items():
            oos    = result.oos_returns
            app_dd = result.oos_summary()["OOS Max DD"]
            oracle = max_drawdown(oos)
            assert_allclose(app_dd, oracle, rtol=1e-9)

    def test_oos_drawdown_is_non_positive(self):
        """Drawdown is a loss measure and must be <= 0."""
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices)
        for model, result in results.items():
            dd = result.oos_summary()["OOS Max DD"]
            assert dd <= 1e-10, f"{model}: max drawdown is positive ({dd:.6f})"

    def test_constant_return_portfolio_has_zero_oos_vol(self):
        """
        Dataset A: identical constant returns → portfolio vol = 0.
        Ann vol of OOS returns must be effectively zero.
        """
        prices = make_dataset_a(n_prices=350, r=0.001, n_assets=3)
        results = _run(prices)
        for model, result in results.items():
            vol = result.oos_summary()["OOS Ann. Vol"]
            assert vol < 1e-10, (
                f"{model}: OOS vol = {vol:.2e} for constant-return data, expected ~0"
            )


class TestMetricSeparation:

    def test_oos_metrics_do_not_use_training_data(self):
        """
        Corrupt training prices while leaving the boundary price intact.
        OOS return on day t = P[t] / P[t-1] - 1. The first OOS return
        depends on P[t-1] = the last training price. To test that OOS
        results are unaffected by training data, we must leave that
        boundary price unchanged. We corrupt only prices[0..TRAIN-1],
        so P[TRAIN] (the boundary) is intact.

        Equal Weight uses no training data for weights → OOS returns must
        be bit-for-bit identical after corruption.
        """
        prices = make_dataset_a(n_prices=400, r=0.001, n_assets=2)
        results_clean = _run(prices)
        oos_clean = results_clean["Equal Weight"].oos_returns.copy()

        # Corrupt prices up to but NOT including the boundary price
        # prices.index[TRAIN] is the boundary (used as denominator for first OOS return)
        # So we corrupt only prices.index[0 .. TRAIN-1]
        boundary_idx = TRAIN  # price position (one more than last training return)
        prices_corrupt = prices.copy()
        rng = np.random.default_rng(37)
        corrupt_noise = pd.DataFrame(
            rng.uniform(50, 200, (boundary_idx, 2)),
            index=prices.index[:boundary_idx],
            columns=prices.columns,
        )
        prices_corrupt.iloc[:boundary_idx] = corrupt_noise.values

        results_corrupt = _run(prices_corrupt)
        oos_corrupt = results_corrupt["Equal Weight"].oos_returns

        oos_common = oos_clean.index.intersection(oos_corrupt.index)
        assert_allclose(
            oos_clean[oos_common].values,
            oos_corrupt[oos_common].values,
            rtol=1e-9,
        ), (
            "OOS returns changed when training prices (before boundary) "
            "were corrupted. Equal Weight does not use training data — "
            "this indicates the OOS returns depend on training prices."
        )

    def test_oos_sharpe_uses_only_oos_returns(self):
        """
        Directly verify that OOS Sharpe is computed on the OOS return series only,
        not the combined IS+OOS series.

        Oracle: compute Sharpe on oos_returns independently, compare to summary.
        """
        rng = np.random.default_rng(39)
        n = 400
        dates = biz_dates(n)
        rets = rng.normal(0.0005, 0.01, (n, 2))
        prices = pd.DataFrame(np.cumprod(1 + rets, axis=0) * 100,
                               index=dates, columns=["A", "B"])

        results = _run(prices)
        result = results["Equal Weight"]
        rf = 0.04 / TRADING_DAYS

        oos = result.oos_returns
        rf_s = pd.Series(rf, index=oos.index)

        oracle_sharpe = sharpe_ratio(oos, rf_s)
        app_sharpe    = result.oos_summary(rf_annual=0.04)["OOS Sharpe"]

        assert_allclose(oracle_sharpe, app_sharpe, rtol=1e-9)
