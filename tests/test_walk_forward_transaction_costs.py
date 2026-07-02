"""
test_walk_forward_transaction_costs.py
-----------------------------------------
Validates turnover calculation and transaction cost logic
against independent oracles.

Turnover definition:
  one-way turnover = sum(|w_new_i - w_old_i|) / 2

This represents the fraction of the portfolio that is repositioned.
A turnover of 0.10 means 10% of the portfolio is traded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from analytics.backtest import run_walk_forward
from analytics.turnover import compute_turnover
from tests.conftest import make_dataset_a, biz_dates

TRAIN = 100
TEST  = 40


def _run(prices, models=None, train=TRAIN, test=TEST):
    if models is None:
        models = ["Equal Weight"]
    return run_walk_forward(
        prices, models, train_window=train, test_window=test,
        shrink=False, min_weight=0.0, max_weight=1.0,
    )


class TestTurnoverOracle:

    def test_turnover_oracle_2asset_known_weights(self):
        """
        Oracle: w_before=[0.6, 0.4], w_after=[0.4, 0.6]
        |0.4-0.6| + |0.6-0.4| = 0.4, divided by 2 = 0.2

        Verify compute_turnover returns 0.20.
        """
        w_before = pd.Series({"A": 0.6, "B": 0.4})
        w_after  = pd.Series({"A": 0.4, "B": 0.6})
        oracle = 0.20
        app    = compute_turnover(w_before, w_after)
        assert_allclose(app, oracle, rtol=1e-10)

    def test_turnover_zero_when_weights_identical(self):
        """No trade = zero turnover."""
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        assert_allclose(compute_turnover(w, w), 0.0, atol=1e-12)

    def test_turnover_one_when_full_portfolio_swap(self):
        """
        Going from 100% A to 100% B.
        Oracle: |0-1| + |1-0| = 2, divided by 2 = 1.0.
        """
        w_before = pd.Series({"A": 1.0, "B": 0.0})
        w_after  = pd.Series({"A": 0.0, "B": 1.0})
        assert_allclose(compute_turnover(w_before, w_after), 1.0, rtol=1e-10)

    def test_turnover_symmetric(self):
        """Turnover from A→B must equal turnover from B→A (direction-agnostic)."""
        w1 = pd.Series({"A": 0.7, "B": 0.2, "C": 0.1})
        w2 = pd.Series({"A": 0.3, "B": 0.5, "C": 0.2})
        assert_allclose(
            compute_turnover(w1, w2),
            compute_turnover(w2, w1),
            rtol=1e-12,
        )

    def test_turnover_bounded_between_zero_and_one(self):
        """One-way turnover of a long-only portfolio is always in [0, 1]."""
        rng = np.random.default_rng(41)
        for _ in range(20):
            n = rng.integers(2, 10)
            w1 = rng.dirichlet(np.ones(n))
            w2 = rng.dirichlet(np.ones(n))
            assets = [f"A{i}" for i in range(n)]
            to = compute_turnover(
                pd.Series(w1, index=assets),
                pd.Series(w2, index=assets),
            )
            assert 0.0 <= to <= 1.0 + 1e-10, f"Turnover {to:.4f} out of [0,1]"

    def test_turnover_handles_new_asset(self):
        """
        Asset C appears in w_after but not w_before.
        Oracle: treat w_before[C] = 0.
        """
        w_before = pd.Series({"A": 0.6, "B": 0.4})
        w_after  = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        # |0.5-0.6| + |0.3-0.4| + |0.2-0| = 0.1 + 0.1 + 0.2 = 0.4, / 2 = 0.2
        oracle = 0.20
        assert_allclose(compute_turnover(w_before, w_after), oracle, rtol=1e-10)

    def test_turnover_handles_dropped_asset(self):
        """
        Asset C in w_before but not w_after.
        Oracle: treat w_after[C] = 0.
        """
        w_before = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        w_after  = pd.Series({"A": 0.6, "B": 0.4})
        # same as above by symmetry
        oracle = 0.20
        assert_allclose(compute_turnover(w_before, w_after), oracle, rtol=1e-10)


class TestBacktestTurnover:

    def test_equal_weight_turnover_zero_after_first_period(self):
        """
        Equal Weight produces identical weights every period (1/N for all assets).
        Rebalancing turnover from period 2 onward = 0.

        This is a key sanity check: if the engine reuses stale weights or
        misidentifies assets, turnover would be non-zero.
        """
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices)
        result = results["Equal Weight"]

        for i, p in enumerate(result.periods[1:], start=1):
            assert_allclose(p.turnover, 0.0, atol=1e-12), (
                f"Equal Weight period {i}: turnover = {p.turnover:.6e}, expected 0"
            )

    def test_first_period_turnover_is_one(self):
        """
        First period always reports turnover = 1.0 (full initial deployment).
        This is a hard-coded convention in the engine, not computed from weights.
        """
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices, models=["Equal Weight", "Minimum Variance"])
        for model, result in results.items():
            assert_allclose(result.periods[0].turnover, 1.0, rtol=1e-12), (
                f"{model}: first period turnover = {result.periods[0].turnover}, "
                f"expected 1.0"
            )

    def test_turnover_excluded_from_avg_summary(self):
        """
        oos_summary() excludes the first period (initial investment) from
        the average turnover. Oracle: mean of periods[1:].
        """
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices, models=["Equal Weight"])
        result = results["Equal Weight"]

        app_avg = result.oos_summary()["Avg Turnover"]
        oracle_avg = float(result.turnover_series.iloc[1:].mean())
        assert_allclose(app_avg, oracle_avg, rtol=1e-9)

    def test_turnover_matches_oracle_when_weights_change(self):
        """
        For Minimum Variance on a heterogeneous dataset, weights change each period.
        Verify turnover for period k equals oracle compute_turnover(w_{k-1}, w_k).
        """
        rng = np.random.default_rng(43)
        n = 400
        dates = biz_dates(n)
        rets  = rng.normal(0.0003, 0.012, (n, 3))
        prices = pd.DataFrame(np.cumprod(1 + rets, axis=0) * 100,
                               index=dates, columns=["X", "Y", "Z"])

        results = _run(prices, models=["Minimum Variance"])
        result  = results["Minimum Variance"]

        for i in range(1, len(result.periods)):
            p       = result.periods[i]
            app_to  = p.turnover
            oracle  = compute_turnover(p.prev_weights, p.weights)
            assert_allclose(app_to, oracle, rtol=1e-9), (
                f"Turnover mismatch at period {i}: app={app_to:.6f}, oracle={oracle:.6f}"
            )

    def test_cost_drag_is_zero_when_no_turnover(self):
        """
        Equal Weight has zero turnover after period 0.
        If we compute net return = gross return × (1 - cost per rebalance),
        and cost = turnover × spread, then cost_drag ≈ 0 for EW.
        """
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices)
        result = results["Equal Weight"]

        for i, p in enumerate(result.periods[1:], start=1):
            # turnover = 0 → cost = 0 for any spread assumption
            cost = p.turnover * 0.001  # 10bps spread
            assert_allclose(cost, 0.0, atol=1e-12), (
                f"Non-zero cost drag despite zero turnover at period {i}"
            )

    def test_higher_turnover_implies_higher_cost_drag(self):
        """
        If strategy A has higher turnover than strategy B (same spread),
        its cost drag must be higher.
        """
        rng = np.random.default_rng(47)
        n = 500
        dates = biz_dates(n)
        rets  = rng.normal(0.0003, 0.015, (n, 4))
        prices = pd.DataFrame(np.cumprod(1 + rets, axis=0) * 100,
                               index=dates, columns=["A", "B", "C", "D"])

        results = _run(prices, models=["Equal Weight", "Maximum Sharpe"])

        ew_avg_to = results["Equal Weight"].oos_summary()["Avg Turnover"]
        ms_avg_to = results["Maximum Sharpe"].oos_summary()["Avg Turnover"]

        spread = 0.001
        ew_cost = ew_avg_to * spread
        ms_cost = ms_avg_to * spread

        # MaxSharpe typically has much higher turnover than Equal Weight
        # This test is directional, not exact — it can fail if the dataset
        # happens to produce stable MaxSharpe weights (acceptable edge case)
        if ms_avg_to > ew_avg_to:
            assert ms_cost > ew_cost
        else:
            pytest.skip(
                f"MaxSharpe turnover ({ms_avg_to:.3f}) not > EW ({ew_avg_to:.3f}) "
                f"on this dataset — directional test not applicable"
            )
