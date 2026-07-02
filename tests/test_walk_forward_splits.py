"""
test_walk_forward_splits.py
----------------------------
Validates that the rolling train/test split logic is correct:
- Training windows contain only past data
- Test windows start strictly after training windows end
- No overlap between train and test
- Rolling step advances by exactly test_window
- Final incomplete windows are handled without crashing
- Period count matches the algebraic expectation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from analytics.backtest import run_walk_forward
from tests.conftest import (
    make_dataset_a, make_dataset_f, biz_dates
)

TRAIN = 100
TEST  = 40


def _run(prices, train=TRAIN, test=TEST, models=None):
    if models is None:
        models = ["Equal Weight"]
    return run_walk_forward(
        prices, models, train_window=train, test_window=test,
        shrink=False, min_weight=0.0, max_weight=1.0,
    )


# ─── Split structure ──────────────────────────────────────────────────────────

class TestSplitStructure:

    def test_train_ends_strictly_before_test_starts(self):
        """For every period in every model, train_end < test_start."""
        prices = make_dataset_a(n_prices=350)
        results = _run(prices)
        for model, result in results.items():
            for i, p in enumerate(result.periods):
                assert p.train_end < p.test_start, (
                    f"{model} period {i}: train_end={p.train_end} "
                    f"not before test_start={p.test_start}"
                )

    def test_no_date_overlap_between_train_and_test(self):
        """
        No date that appears in the test window may appear in the training window.
        Oracle: intersection of training index and test index must be empty.
        """
        prices = make_dataset_a(n_prices=350)
        simple_returns = prices.pct_change().dropna()
        results = _run(prices)
        for model, result in results.items():
            for i, p in enumerate(result.periods):
                train_dates = simple_returns.loc[p.train_start:p.train_end].index
                test_dates  = p.oos_returns.index
                overlap = train_dates.intersection(test_dates)
                assert len(overlap) == 0, (
                    f"{model} period {i}: {len(overlap)} overlapping dates "
                    f"between training and test windows"
                )

    def test_training_window_length_correct(self):
        """
        Each training window contains exactly train_window observations.
        Tolerance of ±1 for boundary-date inclusion edge cases.
        """
        prices = make_dataset_a(n_prices=350)
        simple_returns = prices.pct_change().dropna()
        results = _run(prices)
        for model, result in results.items():
            for i, p in enumerate(result.periods):
                n_train = len(simple_returns.loc[p.train_start:p.train_end])
                assert abs(n_train - TRAIN) <= 1, (
                    f"{model} period {i}: training window has {n_train} rows, "
                    f"expected {TRAIN}"
                )

    def test_test_window_length_correct_except_last(self):
        """
        Every period except possibly the last has exactly test_window rows.
        The last period may be shorter if the data does not divide evenly.
        """
        prices = make_dataset_a(n_prices=350)
        results = _run(prices)
        for model, result in results.items():
            for i, p in enumerate(result.periods[:-1]):  # all except last
                n_test = len(p.oos_returns)
                assert n_test == TEST, (
                    f"{model} period {i}: test window has {n_test} rows, "
                    f"expected {TEST}"
                )
            # last period: at most TEST rows
            last = result.periods[-1]
            assert len(last.oos_returns) <= TEST

    def test_rolling_step_advances_by_test_window(self):
        """
        Each successive test_start advances by exactly test_window trading days.
        Oracle: difference in sequential test_start positions in simple_returns.
        """
        prices = make_dataset_a(n_prices=350)
        simple_returns = prices.pct_change().dropna()
        results = _run(prices)
        for model, result in results.items():
            periods = result.periods
            for i in range(1, len(periods)):
                prev_start = simple_returns.index.get_loc(periods[i - 1].test_start)
                curr_start = simple_returns.index.get_loc(periods[i].test_start)
                step = curr_start - prev_start
                assert step == TEST, (
                    f"{model} periods {i-1}→{i}: step is {step} rows, "
                    f"expected {TEST}"
                )

    def test_train_start_also_advances_by_test_window(self):
        """
        As test_start advances by test_window, train_start must also advance
        by test_window (the window slides, not expands).
        """
        prices = make_dataset_a(n_prices=350)
        simple_returns = prices.pct_change().dropna()
        results = _run(prices)
        for model, result in results.items():
            periods = result.periods
            for i in range(1, len(periods)):
                prev_ts = simple_returns.index.get_loc(periods[i - 1].train_start)
                curr_ts = simple_returns.index.get_loc(periods[i].train_start)
                step = curr_ts - prev_ts
                assert step == TEST, (
                    f"{model}: train_start step {step} ≠ {TEST}"
                )

    def test_period_count_matches_range_oracle(self):
        """
        Number of periods = len(range(train_window, n_returns, test_window)).
        Oracle computed algebraically from data dimensions.
        """
        prices = make_dataset_a(n_prices=350)
        n_returns = len(prices) - 1   # pct_change drops first row
        expected = len(range(TRAIN, n_returns, TEST))
        results = _run(prices)
        for model, result in results.items():
            assert result.n_periods == expected, (
                f"{model}: got {result.n_periods} periods, expected {expected}"
            )

    def test_insufficient_data_raises_value_error(self):
        """Fewer observations than train_window + test_window must raise ValueError."""
        tiny = make_dataset_a(n_prices=TRAIN + TEST - 1)
        with pytest.raises(ValueError, match="Insufficient data"):
            _run(tiny)

    def test_final_period_does_not_crash(self):
        """
        When n_returns is not divisible by test_window, the last period is
        shorter but must not raise any exception.
        """
        # 350 prices → 349 returns. 349 - 100 = 249. 249 / 40 = 6 remainder 9.
        # Last period has 9 rows, not 40.
        prices = make_dataset_a(n_prices=350)
        results = _run(prices)   # should not raise
        for model, result in results.items():
            last = result.periods[-1]
            assert len(last.oos_returns) >= 1

    def test_test_end_within_data_bounds(self):
        """test_end must not exceed the last available date in simple_returns."""
        prices = make_dataset_a(n_prices=350)
        simple_returns = prices.pct_change().dropna()
        last_date = simple_returns.index[-1]
        results = _run(prices)
        for model, result in results.items():
            for p in result.periods:
                assert p.test_end <= last_date, (
                    f"test_end {p.test_end} is beyond last data date {last_date}"
                )

    def test_all_periods_dates_monotonically_increasing(self):
        """test_start dates across periods must be strictly increasing."""
        prices = make_dataset_a(n_prices=350)
        results = _run(prices)
        for model, result in results.items():
            starts = [p.test_start for p in result.periods]
            for i in range(1, len(starts)):
                assert starts[i] > starts[i - 1], (
                    f"{model}: period {i} test_start not after period {i-1}"
                )

    def test_explicit_date_alignment_dataset_f(self):
        """
        Dataset F: explicit traceable dates.
        Verify no test-period date appears in training index of that period.
        """
        prices = make_dataset_f()
        simple_returns = prices.pct_change().dropna()
        results = _run(prices, train=100, test=40)
        for model, result in results.items():
            for p in result.periods:
                train_idx = simple_returns.loc[p.train_start:p.train_end].index
                test_idx  = p.oos_returns.index
                for d in test_idx:
                    assert d not in train_idx, (
                        f"Date {d} appears in both test window and training index"
                    )
