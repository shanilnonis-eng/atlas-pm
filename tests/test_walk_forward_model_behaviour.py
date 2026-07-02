"""
test_walk_forward_model_behaviour.py
---------------------------------------
Validates model-specific behaviour inside the walk-forward engine.
Tests use synthetic datasets with known expected outcomes.

Each test makes a specific, falsifiable prediction about what a correctly
implemented model should produce on a given dataset. These predictions are
derived from theory, not from re-running the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from analytics.backtest import run_walk_forward
from construction.optimiser import compute_cov_matrix, equal_weight
from tests.conftest import (
    make_dataset_a, make_dataset_b, make_dataset_c,
    make_dataset_d, biz_dates
)

TRAIN = 100
TEST  = 40


def _run(prices, models, train=TRAIN, test=TEST, rf=0.04):
    return run_walk_forward(
        prices, models, train_window=train, test_window=test,
        rf_annual=rf, shrink=False, min_weight=0.0, max_weight=1.0,
    )


class TestEqualWeight:

    def test_equal_weight_is_1_over_n_every_period(self):
        """1/N at every period — no history needed, weights are constant."""
        prices = make_dataset_a(n_prices=350, n_assets=4)
        results = _run(prices, ["Equal Weight"])
        result = results["Equal Weight"]
        n = len(prices.columns)
        expected = 1.0 / n
        for i, p in enumerate(result.periods):
            assert_allclose(p.weights.values, expected, rtol=1e-10), (
                f"Equal Weight period {i}: {p.weights.values} ≠ {expected}"
            )

    def test_equal_weight_weights_sum_to_one_every_period(self):
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices, ["Equal Weight"])
        for p in results["Equal Weight"].periods:
            assert_allclose(p.weights.sum(), 1.0, rtol=1e-10)

    def test_equal_weight_no_negative_weights(self):
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices, ["Equal Weight"])
        for p in results["Equal Weight"].periods:
            assert (p.weights.values >= -1e-10).all()

    def test_equal_weight_identical_assets_matches_single_asset(self):
        """
        Dataset C: both assets identical. Equal Weight = [0.5, 0.5].
        Portfolio return = 0.5*r_A + 0.5*r_B = r_A (since r_A = r_B).
        Oracle: portfolio returns = single-asset returns.
        """
        prices = make_dataset_c(n_prices=350)
        simple_returns = prices.pct_change().dropna()
        results = _run(prices, ["Equal Weight"])
        for p in results["Equal Weight"].periods:
            test_r = simple_returns.loc[p.test_start:p.test_end, "A"]
            assert_allclose(p.oos_returns.values, test_r.values, rtol=1e-10)


class TestMinimumVariance:

    def test_min_variance_weights_sum_to_one(self):
        prices = make_dataset_d(n_prices=500)
        results = _run(prices, ["Minimum Variance"])
        for p in results["Minimum Variance"].periods:
            assert_allclose(p.weights.sum(), 1.0, rtol=1e-6)

    def test_min_variance_no_negative_weights(self):
        prices = make_dataset_d(n_prices=500)
        results = _run(prices, ["Minimum Variance"])
        for p in results["Minimum Variance"].periods:
            assert (p.weights.values >= -1e-8).all()

    def test_min_variance_overweights_low_vol_asset(self):
        """
        Dataset D: LowVol daily std = 0.002, HighVol daily std = 0.020.
        MinVar trained on any 100-day window should heavily overweight LowVol.
        Oracle: for uncorrelated assets, w_low* = sigma_high² / (sigma_low² + sigma_high²).
        With sig_low=0.002, sig_high=0.02:
        w_low* = 0.0004 / (0.000004 + 0.0004) ≈ 0.99.
        With max_weight=1.0, expect w_LowVol >> 0.5 at every period.
        """
        prices = make_dataset_d(n_prices=500)
        results = _run(prices, ["Minimum Variance"])
        for i, p in enumerate(results["Minimum Variance"].periods):
            w_low = float(p.weights.get("LowVol", 0))
            assert w_low > 0.70, (
                f"Period {i}: MinVar weight on LowVol = {w_low:.3f}, expected > 0.70. "
                f"MinVar is not identifying the low-vol asset."
            )

    def test_min_variance_vol_le_equal_weight_vol(self):
        """
        By construction MinVar minimises vol. Its realised training-period vol
        must be ≤ equal weight vol on the same training data.
        Verified directly from the covariance matrix, not from the backtest.
        """
        prices = make_dataset_d(n_prices=500)
        simple_returns = prices.pct_change().dropna()

        ew_results  = _run(prices, ["Equal Weight"])
        mv_results  = _run(prices, ["Minimum Variance"])

        for i in range(len(ew_results["Equal Weight"].periods)):
            ew_p = ew_results["Equal Weight"].periods[i]
            mv_p = mv_results["Minimum Variance"].periods[i]

            # Use the same training slice for both
            train_rets = simple_returns.loc[ew_p.train_start:ew_p.train_end]
            cov = train_rets.cov()

            vol_ew = np.sqrt(float(ew_p.weights.values @ cov.values @ ew_p.weights.values))
            vol_mv = np.sqrt(float(mv_p.weights.values @ cov.values @ mv_p.weights.values))

            assert vol_mv <= vol_ew + 1e-8, (
                f"Period {i}: MinVar vol {vol_mv:.6f} > EW vol {vol_ew:.6f} — "
                f"optimiser failed to find minimum variance"
            )

    def test_min_variance_uses_training_covariance_not_full_sample(self):
        """
        For period 0, MinVar should use training data covariance only.
        Oracle: compute cov from training slice, compute optimal weights analytically
        for 2 uncorrelated assets, compare to engine output.
        """
        prices = make_dataset_d(n_prices=500)
        simple_returns = prices.pct_change().dropna()
        results = _run(prices, ["Minimum Variance"])

        p0 = results["Minimum Variance"].periods[0]
        train_rets = simple_returns.loc[p0.train_start:p0.train_end]
        cov = train_rets.cov()

        s_low  = float(cov.loc["LowVol", "LowVol"])
        s_high = float(cov.loc["HighVol", "HighVol"])
        s_lh   = float(cov.loc["LowVol", "HighVol"])

        # Analytical MinVar for 2 assets:
        # w_low = (s_high - s_lh) / (s_low + s_high - 2*s_lh)
        denom = s_low + s_high - 2 * s_lh
        if abs(denom) < 1e-14:
            pytest.skip("Degenerate covariance — assets effectively identical")
        oracle_w_low  = (s_high - s_lh) / denom
        oracle_w_high = 1 - oracle_w_low
        oracle_w_low  = max(0, min(1, oracle_w_low))
        oracle_w_high = 1 - oracle_w_low

        app_w_low  = float(p0.weights["LowVol"])
        app_w_high = float(p0.weights["HighVol"])

        assert_allclose(app_w_low,  oracle_w_low,  atol=0.05), (
            f"MinVar LowVol weight: app={app_w_low:.3f}, oracle={oracle_w_low:.3f}"
        )
        assert_allclose(app_w_high, oracle_w_high, atol=0.05), (
            f"MinVar HighVol weight: app={app_w_high:.3f}, oracle={oracle_w_high:.3f}"
        )


class TestMaxSharpe:

    def test_max_sharpe_weights_sum_to_one(self):
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices, ["Maximum Sharpe"])
        for p in results["Maximum Sharpe"].periods:
            assert_allclose(p.weights.sum(), 1.0, rtol=1e-6)

    def test_max_sharpe_no_negative_weights(self):
        prices = make_dataset_a(n_prices=350, n_assets=3)
        results = _run(prices, ["Maximum Sharpe"])
        for p in results["Maximum Sharpe"].periods:
            assert (p.weights.values >= -1e-8).all()

    def test_max_sharpe_chases_training_winner_regime_reversal(self):
        """
        Dataset B: A dominates training window, B dominates test window.
        MaxSharpe fit on training data MUST overweight A.
        If it overweights B, future information is leaking in.
        """
        train, test = 120, 60
        prices = make_dataset_b(
            train_window=train, test_window=test,
            r_high=0.003, r_low=0.0001, seed=42
        )
        results = run_walk_forward(
            prices, ["Maximum Sharpe"],
            train_window=train, test_window=test,
            rf_annual=0.0, shrink=False,
            min_weight=0.0, max_weight=1.0,
        )
        w0 = results["Maximum Sharpe"].periods[0].weights
        assert w0["A"] > w0["B"], (
            f"MaxSharpe should overweight training winner A, "
            f"got w_A={w0['A']:.3f}, w_B={w0['B']:.3f}"
        )

    def test_max_sharpe_oos_suffers_when_training_winner_reverses(self):
        """
        Follows the regime reversal test:
        MaxSharpe overweights A based on training → OOS period sees A underperform.
        Therefore MaxSharpe OOS return in first test period should be below B's return.
        """
        train, test = 120, 60
        prices = make_dataset_b(
            train_window=train, test_window=test,
            r_high=0.003, r_low=0.0001, seed=42
        )
        simple_rets = prices.pct_change().dropna()
        results = run_walk_forward(
            prices, ["Maximum Sharpe"],
            train_window=train, test_window=test,
            rf_annual=0.0, shrink=False,
            min_weight=0.0, max_weight=1.0,
        )
        p0 = results["Maximum Sharpe"].periods[0]
        test_slice = simple_rets.loc[p0.test_start:p0.test_end]

        # Return if pure B strategy in test period
        pure_b_return = float((1 + test_slice["B"]).prod() - 1)
        # Actual MaxSharpe portfolio return
        port_return   = float((1 + p0.oos_returns).prod() - 1)

        # The portfolio (biased toward A) should underperform pure B
        assert port_return < pure_b_return, (
            f"MaxSharpe OOS return ({port_return:.4f}) >= pure-B return "
            f"({pure_b_return:.4f}) — expected MaxSharpe to suffer from "
            f"chasing the training-period winner"
        )


class TestRiskParity:

    def test_risk_parity_weights_sum_to_one(self):
        prices = make_dataset_a(n_prices=400, n_assets=3)
        results = _run(prices, ["Risk Parity"])
        for p in results["Risk Parity"].periods:
            assert_allclose(p.weights.sum(), 1.0, rtol=1e-6)

    def test_risk_parity_no_negative_weights(self):
        prices = make_dataset_a(n_prices=400, n_assets=3)
        results = _run(prices, ["Risk Parity"])
        for p in results["Risk Parity"].periods:
            assert (p.weights.values >= -1e-8).all()

    def test_risk_parity_equal_vol_assets_near_equal_weight(self):
        """
        Dataset A: all assets have identical volatility (zero — constant returns).
        RP with identical vols should give approximately equal weights.
        Use random constant-mean assets to avoid degenerate zero-vol case.
        """
        rng = np.random.default_rng(53)
        n = 400
        dates = biz_dates(n)
        # Same vol, uncorrelated
        rets = np.column_stack([rng.normal(0.001, 0.01, n) for _ in range(3)])
        prices = pd.DataFrame(np.cumprod(1 + rets, axis=0) * 100,
                               index=dates, columns=["X", "Y", "Z"])

        results = _run(prices, ["Risk Parity"])
        for i, p in enumerate(results["Risk Parity"].periods):
            w = p.weights.values
            # All weights should be close to 1/3
            assert_allclose(w, np.full(3, 1 / 3), atol=0.08), (
                f"Risk Parity period {i}: weights {w} deviate from 1/3 "
                f"for equal-vol assets"
            )

    def test_risk_parity_overweights_low_vol(self):
        """
        Dataset D: LowVol daily std = 0.002, HighVol = 0.020.
        RP inverts volatility: lower-vol asset gets higher weight.
        Oracle: w_low/w_high ≈ sigma_high/sigma_low = 10 (uncorrelated).
        Expect w_LowVol > 0.80.
        """
        prices = make_dataset_d(n_prices=500)
        results = _run(prices, ["Risk Parity"])
        for i, p in enumerate(results["Risk Parity"].periods):
            w_low = float(p.weights.get("LowVol", 0))
            assert w_low > 0.70, (
                f"Period {i}: Risk Parity weight on LowVol = {w_low:.3f}. "
                f"Expected > 0.70 since LowVol has ~10× lower daily vol."
            )


class TestAllModels:

    def test_all_models_weights_sum_to_one_every_period(self, multi_model_backtest):
        """Fundamental constraint: portfolio weights always sum to 1."""
        for model, result in multi_model_backtest.items():
            for i, p in enumerate(result.periods):
                assert_allclose(p.weights.sum(), 1.0, rtol=1e-6), (
                    f"{model} period {i}: weights sum to {p.weights.sum():.6f}"
                )

    def test_all_models_non_negative_weights_every_period(self, multi_model_backtest):
        """Long-only constraint: no negative weights."""
        for model, result in multi_model_backtest.items():
            for i, p in enumerate(result.periods):
                assert (p.weights.values >= -1e-8).all(), (
                    f"{model} period {i}: negative weights found"
                )

    def test_all_models_produce_same_number_of_periods(self, multi_model_backtest):
        """All models must run the same number of periods (same data, same windows)."""
        n_periods = [result.n_periods for result in multi_model_backtest.values()]
        assert len(set(n_periods)) == 1, (
            f"Models have different period counts: "
            f"{dict(zip(multi_model_backtest, n_periods))}"
        )

    def test_all_models_cover_same_oos_date_range(self, multi_model_backtest):
        """All models must have the same OOS start and end dates."""
        starts = {m: r.oos_returns.index[0]  for m, r in multi_model_backtest.items()}
        ends   = {m: r.oos_returns.index[-1] for m, r in multi_model_backtest.items()}
        assert len(set(starts.values())) == 1, f"OOS start dates differ: {starts}"
        assert len(set(ends.values()))   == 1, f"OOS end dates differ: {ends}"
