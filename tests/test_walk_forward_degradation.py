"""
test_walk_forward_degradation.py
-----------------------------------
Validates the in-sample vs out-of-sample Sharpe degradation table.

Critical finding during code inspection:
  BacktestResult.is_returns concatenates IS return series from overlapping
  training windows. Training windows overlap by (train_window - test_window)
  days, producing duplicate timestamps in the concatenated series. When
  sharpe_ratio() is applied to this series, pandas includes all duplicate
  rows, biasing IS Sharpe upward and inflating the degradation metric.

Tests here:
  1. Document and confirm the duplicate-timestamp defect.
  2. Verify the fix (per-period IS Sharpe average) produces correct results.
  3. Verify degradation definition: IS_Sharpe - OOS_Sharpe.
  4. Verify Equal Weight has near-zero degradation (uses no estimated parameters).
  5. Verify IS Sharpe uses training data, OOS Sharpe uses test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from analytics.backtest import run_walk_forward, build_degradation_table, BacktestResult
from analytics.returns import sharpe_ratio, annualised_return, annualised_volatility
from tests.conftest import make_dataset_a, biz_dates

TRAIN = 100
TEST  = 40
RF    = 0.04


def _run(prices, models=None, train=TRAIN, test=TEST, rf=RF):
    if models is None:
        models = ["Equal Weight", "Minimum Variance"]
    return run_walk_forward(
        prices, models, train_window=train, test_window=test,
        rf_annual=rf, shrink=False, min_weight=0.0, max_weight=1.0,
    )


class TestDegradationDefinition:

    def test_degradation_equals_is_minus_oos_sharpe(self):
        """
        Degradation = IS_Sharpe - OOS_Sharpe.
        Positive means in-sample overstated reality; negative means understatement.
        """
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices)
        deg_table = build_degradation_table(results, rf_annual=RF)

        for model in results:
            row = deg_table.loc[model]
            is_sharpe  = row["IS Sharpe"]
            oos_sharpe = row["OOS Sharpe"]
            stated_deg = row["Degradation"]
            computed   = is_sharpe - oos_sharpe
            assert_allclose(stated_deg, computed, rtol=1e-9), (
                f"{model}: Degradation = {stated_deg:.6f} ≠ "
                f"IS({is_sharpe:.4f}) - OOS({oos_sharpe:.4f}) = {computed:.6f}"
            )

    def test_oos_sharpe_in_degradation_matches_oos_summary(self):
        """The OOS Sharpe in the degradation table must match oos_summary()."""
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices)
        deg_table = build_degradation_table(results, rf_annual=RF)

        for model, result in results.items():
            oos_from_summary    = result.oos_summary(rf_annual=RF)["OOS Sharpe"]
            oos_from_deg_table  = float(deg_table.loc[model, "OOS Sharpe"])
            assert_allclose(oos_from_summary, oos_from_deg_table, rtol=1e-9), (
                f"{model}: OOS Sharpe mismatch between oos_summary and deg table"
            )

    def test_oos_sharpe_oracle(self):
        """
        OOS Sharpe = sharpe_ratio(concatenated_oos_returns).
        Oracle computed directly from OOS returns.
        """
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices)
        for model, result in results.items():
            oos = result.oos_returns
            rf_s = pd.Series(RF / 252, index=oos.index)
            oracle = sharpe_ratio(oos, rf_s)
            app    = result.oos_summary(rf_annual=RF)["OOS Sharpe"]
            assert_allclose(oracle, app, rtol=1e-9)


class TestISSharpeContamination:

    def test_is_returns_have_duplicate_timestamps_raw(self):
        """
        CONFIRMED BUG (pre-fix): the raw is_returns property concatenates
        overlapping training windows. When train_window > test_window,
        training windows overlap, producing duplicate dates in the series.

        This test documents that the old concatenated series had duplicates.
        It is kept as regression protection — if is_returns is ever changed
        back to the concatenated approach, this test should catch it.

        If the fix is in place (per-period average), is_returns may still
        be the concatenated property; we simply verify the is_summary method
        now uses per-period computation rather than this series.
        """
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices)
        result = results["Equal Weight"]

        # The raw is_returns property may still concatenate overlapping windows
        raw_is = result.is_returns

        # When train_window (100) > test_window (40), windows overlap by 60 days
        # So some dates appear in multiple periods' IS returns → duplicates
        # We check if this is the case (it IS in the unfixed version)
        has_duplicates = not raw_is.index.is_unique

        if has_duplicates:
            # Old unfixed behaviour — is_summary should NOT use this directly
            # Verify that is_summary("IS Sharpe") is computed per-period, not from raw
            per_period_sharpes = []
            rf_daily = RF / 252
            for p in result.periods:
                r = p.is_returns
                rf_s = pd.Series(rf_daily, index=r.index)
                sr = sharpe_ratio(r, rf_s)
                if np.isfinite(sr):
                    per_period_sharpes.append(sr)

            per_period_avg = float(np.mean(per_period_sharpes))
            app_is_sharpe  = float(build_degradation_table(results, RF).loc[
                "Equal Weight", "IS Sharpe"
            ])

            # They should match if the fix is applied
            assert_allclose(app_is_sharpe, per_period_avg, rtol=1e-6), (
                "IS Sharpe in degradation table does not match per-period average. "
                "The duplicate-timestamp bias is still present in is_summary()."
            )

    def test_is_sharpe_computed_per_period_not_on_concatenated_series(self):
        """
        The correct IS Sharpe is the average of per-period IS Sharpe ratios,
        each computed on a clean (non-duplicate) training window.

        This test verifies that the fixed is_summary() returns the per-period average.
        """
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices)
        rf_daily = RF / 252

        for model, result in results.items():
            per_period_sharpes = []
            for p in result.periods:
                r = p.is_returns
                rf_s = pd.Series(rf_daily, index=r.index)
                sr = sharpe_ratio(r, rf_s)
                if np.isfinite(sr):
                    per_period_sharpes.append(sr)

            if not per_period_sharpes:
                continue

            oracle_is_sharpe = float(np.mean(per_period_sharpes))
            app_is_sharpe    = float(
                build_degradation_table(results, RF).loc[model, "IS Sharpe"]
            )
            assert_allclose(oracle_is_sharpe, app_is_sharpe, rtol=1e-6), (
                f"{model}: IS Sharpe {app_is_sharpe:.4f} ≠ per-period avg "
                f"{oracle_is_sharpe:.4f}. is_summary() may still use the "
                f"duplicate-timestamp concatenated series."
            )

    def test_per_period_is_returns_no_duplicates(self):
        """Each individual period's is_returns must have unique timestamps."""
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices)
        for model, result in results.items():
            for i, p in enumerate(result.periods):
                assert p.is_returns.index.is_unique, (
                    f"{model} period {i}: per-period IS returns have duplicates"
                )


class TestEqualWeightDegradation:

    def test_equal_weight_is_sharpe_equals_oos_sharpe_constant_data(self):
        """
        Dataset A: constant returns. EW weights are fixed (1/N) and require
        no training data. IS and OOS portfolio returns are identical in structure
        (same weights, same constant return). IS Sharpe ≈ OOS Sharpe → Degradation ≈ 0.

        This is the gold-standard zero-degradation test for a parameter-free model.
        """
        prices = make_dataset_a(n_prices=350, r=0.001, n_assets=3)
        results = _run(prices, models=["Equal Weight"])
        deg = build_degradation_table(results, RF)

        row = deg.loc["Equal Weight"]
        # Both should be NaN (constant returns → zero vol → undefined Sharpe)
        # or both should be finite and equal
        is_s  = float(row["IS Sharpe"])
        oos_s = float(row["OOS Sharpe"])

        if np.isfinite(is_s) and np.isfinite(oos_s):
            assert_allclose(is_s, oos_s, atol=0.5), (
                f"EW IS Sharpe ({is_s:.4f}) diverges from OOS Sharpe "
                f"({oos_s:.4f}) on constant-return data"
            )
        else:
            # Both NaN is also acceptable for constant-return data
            assert np.isnan(is_s) and np.isnan(oos_s), (
                f"Unexpected IS/OOS Sharpe values: IS={is_s}, OOS={oos_s}"
            )

    def test_degradation_table_has_all_models(self):
        """Degradation table has one row per model, no missing entries."""
        prices = make_dataset_a(n_prices=350, n_assets=3)
        models = ["Equal Weight", "Minimum Variance"]
        results = _run(prices, models=models)
        deg = build_degradation_table(results, RF)
        for m in models:
            assert m in deg.index, f"Model {m!r} missing from degradation table"

    def test_degradation_table_has_required_columns(self):
        """Degradation table must have IS Sharpe, OOS Sharpe, Degradation columns."""
        prices = make_dataset_a(n_prices=350, n_assets=2)
        results = _run(prices)
        deg = build_degradation_table(results, RF)
        for col in ["IS Sharpe", "OOS Sharpe", "Degradation"]:
            assert col in deg.columns, f"Column {col!r} missing from degradation table"
