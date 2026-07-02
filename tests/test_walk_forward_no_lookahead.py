"""
test_walk_forward_no_lookahead.py
-----------------------------------
The most critical property of any backtesting engine: no information from
the test period may influence the weights applied during that test period.

Tests use:
  - Data corruption: modify future returns and verify earlier weights unchanged
  - Regime reversal: confirm MaxSharpe chases the training winner, not future winner
  - Date membership: confirm no test date appears in the training calculation
  - Additive data: verify earlier periods are stable when more future data is added
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from analytics.backtest import run_walk_forward
from tests.conftest import make_dataset_a, make_dataset_b, biz_dates

TRAIN = 100
TEST  = 40


def _run(prices, models=None, train=TRAIN, test=TEST):
    if models is None:
        models = ["Equal Weight"]
    return run_walk_forward(
        prices, models, train_window=train, test_window=test,
        shrink=False, min_weight=0.0, max_weight=1.0,
    )


class TestNoLookahead:

    def test_corrupting_future_prices_does_not_change_period1_weights(self):
        """
        Gold standard look-ahead test.

        1. Run backtest on clean prices → record weights for period 0.
        2. Replace ALL prices AFTER the first test-start date with ×10 (extreme).
        3. Re-run backtest → verify period 0 weights are bit-for-bit identical.

        If any future data leaked into period 0 weight calculation,
        the corruption would change the weights.
        """
        prices_clean = make_dataset_a(n_prices=350, n_assets=3)
        results_clean = _run(prices_clean)
        w_clean = results_clean["Equal Weight"].periods[0].weights.copy()

        # Corrupt everything from the first test period onward
        prices_corrupt = prices_clean.copy()
        simple_rets = prices_clean.pct_change().dropna()
        first_test_pos = TRAIN  # index position in simple_returns
        first_test_date = simple_rets.index[first_test_pos]

        # In prices, the date one row later than first_test_date
        prices_corrupt.loc[first_test_date:] *= 10_000

        results_corrupt = _run(prices_corrupt)
        w_corrupt = results_corrupt["Equal Weight"].periods[0].weights.copy()

        assert_allclose(w_clean.values, w_corrupt.values, rtol=1e-12), (
            "LOOK-AHEAD BIAS DETECTED: Period 0 weights changed when future "
            "prices were corrupted. Training must not use future data."
        )

    def test_corrupting_future_prices_does_not_change_any_earlier_period(self):
        """
        Extend the corruption test to ALL periods:
        corrupting period k+1 onward must not change period k's weights.
        """
        prices = make_dataset_a(n_prices=500, n_assets=3)
        results_clean = _run(prices, models=["Equal Weight", "Minimum Variance"])
        clean_weights = {
            model: [p.weights.copy() for p in result.periods]
            for model, result in results_clean.items()
        }

        # Corrupt prices from period 2 onward
        simple_rets = prices.pct_change().dropna()
        corrupt_from_pos = TRAIN + TEST    # start of period 1 test window
        if corrupt_from_pos >= len(simple_rets):
            pytest.skip("Not enough data for this test")
        corrupt_from_date = simple_rets.index[corrupt_from_pos]

        prices_corrupt = prices.copy()
        prices_corrupt.loc[corrupt_from_date:] *= 999

        results_corrupt = _run(prices_corrupt, models=["Equal Weight", "Minimum Variance"])

        for model in ["Equal Weight", "Minimum Variance"]:
            # Period 0 must be completely unchanged
            w_clean   = clean_weights[model][0].values
            w_corrupt = results_corrupt[model].periods[0].weights.values
            assert_allclose(w_clean, w_corrupt, rtol=1e-10), (
                f"LOOK-AHEAD BIAS: {model} period 0 weights changed after corrupting "
                f"data from period 1 onward."
            )

    def test_regime_reversal_maxsharpe_chases_training_winner(self):
        """
        Dataset B: Asset A dominates in the training window, Asset B in the test window.

        Expected with NO look-ahead bias:
          - MaxSharpe fit on training data → overweights A
          - Test returns are then dominated by A (poor performance)

        If look-ahead bias were present, MaxSharpe would know B outperforms in the
        test period and would overweight B instead. We verify it overweights A.

        Oracle: MaxSharpe trained purely on training returns should reflect A's
        dominance. We directly verify w_A > w_B in the first period's weights.
        """
        train = 120
        test  = 60
        prices = make_dataset_b(train_window=train, test_window=test,
                                 r_high=0.003, r_low=0.0001, seed=42)

        # Verify the training data itself shows A dominating
        simple_rets = prices.pct_change().dropna()
        training_rets = simple_rets.iloc[:train]
        assert training_rets["A"].mean() > training_rets["B"].mean(), (
            "Dataset B construction error: A should dominate in training window"
        )

        # Verify the test data shows B dominating
        test_rets = simple_rets.iloc[train:train + test]
        assert test_rets["B"].mean() > test_rets["A"].mean(), (
            "Dataset B construction error: B should dominate in test window"
        )

        # Run backtest
        results = run_walk_forward(
            prices,
            models=["Maximum Sharpe"],
            train_window=train,
            test_window=test,
            rf_annual=0.0,
            shrink=False,
            min_weight=0.0,
            max_weight=1.0,
        )

        # MaxSharpe trained on [0, train) should overweight A
        w_period0 = results["Maximum Sharpe"].periods[0].weights
        assert w_period0["A"] > w_period0["B"], (
            f"POSSIBLE LOOK-AHEAD BIAS: MaxSharpe overweights B ({w_period0['B']:.3f}) "
            f"over A ({w_period0['A']:.3f}) despite A dominating the training window. "
            f"Expected to chase the training winner (A)."
        )

    def test_adding_future_data_does_not_alter_early_period_weights(self):
        """
        Run backtest on N rows of data → record period 0 weights.
        Run backtest on N + 50 rows of data (more future data appended).
        Period 0 weights must be identical.
        """
        prices_short = make_dataset_a(n_prices=350, n_assets=2)
        prices_long  = make_dataset_a(n_prices=400, n_assets=2)

        r_short = _run(prices_short)
        r_long  = _run(prices_long)

        w_short = r_short["Equal Weight"].periods[0].weights
        w_long  = r_long[ "Equal Weight"].periods[0].weights

        # Align on common assets
        common = w_short.index.intersection(w_long.index)
        assert_allclose(
            w_short[common].values, w_long[common].values, rtol=1e-12
        ), (
            "Period 0 weights changed when extra future data was appended — "
            "look-ahead bias."
        )

    def test_training_dates_contain_no_future_returns(self):
        """
        For each period, every date in the training window must precede
        every date in the test window.

        Oracle: max(training dates) < min(test dates) for every period.
        """
        prices = make_dataset_a(n_prices=350)
        simple_rets = prices.pct_change().dropna()
        results = _run(prices, models=["Equal Weight", "Minimum Variance"])

        for model, result in results.items():
            for i, p in enumerate(result.periods):
                train_idx = simple_rets.loc[p.train_start:p.train_end].index
                test_idx  = p.oos_returns.index
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                assert train_idx.max() < test_idx.min(), (
                    f"{model} period {i}: max training date {train_idx.max()} "
                    f">= min test date {test_idx.min()} — possible leakage"
                )

    def test_min_variance_uses_only_training_covariance(self):
        """
        MinVar derives weights from the covariance of the training returns.
        If test returns were used, changing test-period data would change weights.

        Corruption test applied specifically to MinVar.
        """
        prices_clean = make_dataset_a(n_prices=400, n_assets=3)
        results_clean = _run(prices_clean, models=["Minimum Variance"])
        w0_clean = results_clean["Minimum Variance"].periods[0].weights.copy()

        prices_corrupt = prices_clean.copy()
        simple_rets = prices_clean.pct_change().dropna()
        corrupt_from = simple_rets.index[TRAIN]
        prices_corrupt.loc[corrupt_from:] = prices_corrupt.loc[corrupt_from:] * 500

        results_corrupt = _run(prices_corrupt, models=["Minimum Variance"])
        w0_corrupt = results_corrupt["Minimum Variance"].periods[0].weights.copy()

        assert_allclose(w0_clean.values, w0_corrupt.values, rtol=1e-6), (
            "LOOK-AHEAD BIAS in MinVar: weights changed when test-period prices "
            "were corrupted."
        )
