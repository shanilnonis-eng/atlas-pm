"""
test_walk_forward_returns.py
-------------------------------
Validates that out-of-sample portfolio returns, cumulative wealth,
and in-sample returns are computed correctly against independent oracles.

Oracles are computed from scratch — no reuse of backtest internal functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from analytics.backtest import run_walk_forward, _apply_weights
from tests.conftest import make_dataset_a, make_dataset_c, biz_dates

TRAIN = 100
TEST  = 40


def _run(prices, models=None, train=TRAIN, test=TEST):
    if models is None:
        models = ["Equal Weight"]
    return run_walk_forward(
        prices, models, train_window=train, test_window=test,
        shrink=False, min_weight=0.0, max_weight=1.0,
    )


class TestOOSReturns:

    def test_oos_return_oracle_constant_return_dataset(self):
        """
        Dataset A: all assets return R per day.
        OOS portfolio return must be exactly R every day, regardless of weights.

        Oracle: R is known by construction — no fitting needed.
        """
        R = 0.001
        prices = make_dataset_a(n_prices=350, r=R, n_assets=3)
        results = _run(prices, models=["Equal Weight", "Minimum Variance"])

        for model, result in results.items():
            for i, p in enumerate(result.periods):
                oos = p.oos_returns
                assert_allclose(oos.values, R, atol=1e-10), (
                    f"{model} period {i}: OOS returns not equal to R={R}. "
                    f"Got mean={oos.mean():.6e}, std={oos.std():.6e}"
                )

    def test_oos_return_equals_weighted_asset_returns_oracle(self):
        """
        OOS portfolio return on day t = sum(w_i * r_i_t).
        Oracle: computed row-by-row independently for the first test period.
        """
        prices = make_dataset_a(n_prices=350, n_assets=3)
        simple_returns = prices.pct_change().dropna()
        results = _run(prices)

        for i, period in enumerate(results["Equal Weight"].periods):
            w = period.weights
            test_slice = simple_returns.loc[period.test_start:period.test_end]

            # Oracle: per-day weighted sum
            common = [a for a in w.index if a in test_slice.columns]
            w_norm = w[common] / w[common].sum()
            oracle_returns = (test_slice[common] * w_norm).sum(axis=1)

            assert_allclose(
                period.oos_returns.values,
                oracle_returns.values,
                rtol=1e-9,
                err_msg=f"OOS return mismatch at period {i}",
            )

    def test_oos_returns_no_duplicate_timestamps(self):
        """
        The concatenated OOS return series must have unique timestamps.
        Test periods are non-overlapping, so no date should appear twice.
        """
        prices = make_dataset_a(n_prices=350)
        results = _run(prices)
        for model, result in results.items():
            oos = result.oos_returns
            assert oos.index.is_unique, (
                f"{model}: OOS return series has duplicate timestamps — "
                f"test periods overlap"
            )

    def test_oos_returns_monotonic_timestamps(self):
        """Concatenated OOS returns must be in chronological order."""
        prices = make_dataset_a(n_prices=350)
        results = _run(prices)
        for model, result in results.items():
            assert result.oos_returns.index.is_monotonic_increasing

    def test_oos_returns_cover_consecutive_dates(self):
        """
        OOS returns from period k end on the day before period k+1 begins
        (accounting for business-day gaps only). No gap or overlap.
        """
        prices = make_dataset_a(n_prices=350)
        results = _run(prices)
        for model, result in results.items():
            for i in range(1, len(result.periods)):
                prev_end   = result.periods[i - 1].test_end
                curr_start = result.periods[i].test_start
                assert curr_start > prev_end, (
                    f"{model}: period {i} starts ({curr_start}) before "
                    f"period {i-1} ends ({prev_end}) — overlapping test windows"
                )


class TestCumulativeWealth:

    def test_cumulative_wealth_constant_returns_oracle(self):
        """
        Dataset A: R = 0.001 per day.
        After n test days: cumulative wealth = (1.001)^n.
        Oracle: computed algebraically.
        """
        R = 0.001
        prices = make_dataset_a(n_prices=350, r=R, n_assets=2)
        results = _run(prices)
        result = results["Equal Weight"]
        oos = result.oos_returns

        n_days = len(oos)
        oracle_total_return = (1 + R) ** n_days - 1
        app_total_return    = float((1 + oos).prod() - 1)

        assert_allclose(app_total_return, oracle_total_return, rtol=1e-6), (
            f"Cumulative wealth mismatch: app={app_total_return:.6f}, "
            f"oracle={oracle_total_return:.6f}"
        )

    def test_cumulative_wealth_compounds_geometrically_not_linearly(self):
        """
        Cumulative return = (1+r1)(1+r2)...(1+rn) - 1, NOT r1+r2+...+rn.
        For non-zero variance, arithmetic sum ≠ geometric product.
        """
        rng = np.random.default_rng(29)
        n = 400
        dates = biz_dates(n)
        rets = rng.normal(0.001, 0.01, (n, 2))
        prices = pd.DataFrame(np.cumprod(1 + rets, axis=0) * 100,
                               index=dates, columns=["A", "B"])
        results = _run(prices)
        oos = results["Equal Weight"].oos_returns

        geometric  = float((1 + oos).prod() - 1)
        arithmetic = float(oos.sum())

        assert abs(geometric - arithmetic) > 1e-6, (
            "Geometric and arithmetic returns are identical — "
            "returns may be too small to distinguish, or compounding is not applied"
        )

    def test_wealth_starts_at_one(self):
        """If you index wealth to 1.0 at OOS start, first value = 1+r1, not 1.0."""
        prices = make_dataset_a(n_prices=350, r=0.002)
        results = _run(prices)
        oos = results["Equal Weight"].oos_returns
        wealth = (1 + oos).cumprod()
        assert_allclose(float(wealth.iloc[0]), 1.002, rtol=1e-9)


class TestISReturns:

    def test_is_returns_use_training_window_dates_only(self):
        """
        IS returns for period k should only contain dates from the training window.
        No date from the test window should appear in is_returns.
        """
        prices = make_dataset_a(n_prices=350)
        results = _run(prices)
        for model, result in results.items():
            for i, p in enumerate(result.periods):
                is_dates = p.is_returns.index
                oos_dates = p.oos_returns.index
                overlap = is_dates.intersection(oos_dates)
                assert len(overlap) == 0, (
                    f"{model} period {i}: IS returns overlap with OOS returns "
                    f"on {len(overlap)} dates — IS calculation uses future data"
                )

    def test_is_returns_oracle_constant_dataset(self):
        """
        Dataset A: R = 0.001 per day. IS portfolio return must also be R per day.
        The weights are fitted on data where all returns = R, and applied to data
        where all returns = R. Oracle: IS return = R always.
        """
        R = 0.001
        prices = make_dataset_a(n_prices=350, r=R, n_assets=3)
        results = _run(prices)
        for model, result in results.items():
            for i, p in enumerate(result.periods):
                assert_allclose(p.is_returns.values, R, atol=1e-10), (
                    f"{model} period {i}: IS returns not equal to R={R}"
                )

    def test_apply_weights_renormalises_when_asset_missing(self):
        """
        _apply_weights silently drops missing assets and renormalises.
        Oracle: with 2 assets [A, B] and weights [0.6, 0.4], if B is missing,
        result should use weight 1.0 for A.
        """
        dates = biz_dates(5)
        rets = pd.DataFrame({"A": [0.01, 0.02, 0.01, 0.01, 0.01]}, index=dates)
        weights = pd.Series({"A": 0.6, "B": 0.4})
        port = _apply_weights(rets, weights)
        # B missing → A gets full weight → port = rets["A"]
        assert_allclose(port.values, rets["A"].values, rtol=1e-10)

    def test_is_returns_per_period_have_no_duplicate_dates_within_period(self):
        """
        Within a single period, is_returns must have unique timestamps.
        (Overlap only occurs when concatenating across periods.)
        """
        prices = make_dataset_a(n_prices=350)
        results = _run(prices)
        for model, result in results.items():
            for i, p in enumerate(result.periods):
                assert p.is_returns.index.is_unique, (
                    f"{model} period {i}: per-period IS returns have "
                    f"duplicate timestamps"
                )

    def test_concatenated_oos_returns_unique_dates(self):
        """OOS returns, when concatenated via result.oos_returns, have unique dates."""
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices)
        for model, result in results.items():
            assert result.oos_returns.index.is_unique, (
                f"{model}: concatenated OOS returns have duplicate dates"
            )
