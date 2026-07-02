"""
test_brinson_attribution.py
────────────────────────────
Full test suite for the Brinson-Hood-Beebower / Brinson-Fachler attribution engine.

All datasets are deterministic (algebraic or fixed-seed random).
No live market data is used.  Every test has a named oracle so failure messages
describe WHAT was expected and WHY, not just that something was wrong.

Test categories
───────────────
 1.  Basic reconciliation (BF) — all effects sum to active return
 2.  Basic reconciliation (BHB) — same check, original 1986 method
 3.  Zero active weights — portfolio = benchmark weights
 4.  Zero selection difference — group returns identical
 5.  Identical portfolio and benchmark — all effects zero
 6.  Overweight winner — allocation effect positive (BF)
 7.  Overweight loser  — allocation effect negative (BF)
 8.  Selection effect sign — positive when within-group outperforms
 9.  Interaction effect formula — active_weight × within-group active return
10.  Weight normalisation — build_benchmark_weights sums to 1
11.  Cumulative attribution — arithmetic cumsum consistency
12.  IC proxy — correlation formula + edge cases
13.  BHB vs BF — different allocation, same total
14.  Missing / zero-weight group — handled without crash or silent error
15.  Validate reconciliation helper — detects residuals correctly
16.  Asset name normalisation — normalize_asset_name handles hidden characters
17.  align_asset_names — robust alignment with mismatched unicode names
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from analytics.brinson_attribution import (
    normalize_asset_name,
    align_asset_names,
    calculate_group_weights,
    build_benchmark_weights,
    calculate_group_returns,
    calculate_brinson_attribution,
    calculate_period_active_return,
    calculate_cumulative_attribution,
    validate_brinson_reconciliation,
    calculate_ic_proxy,
    DEFAULT_CLASSIFICATION,
    BENCHMARK_60_40_GROUP_WEIGHTS,
)


# ─── Shared helpers ────────────────────────────────────────────────────────────

MONTHLY_DATES = pd.date_range("2023-01-31", periods=6, freq="ME")


def _make_attribution_inputs(
    wp: list[float],
    wb: list[float],
    rp_g: list[list[float]],   # shape [n_periods][n_groups]
    rb_g: list[list[float]],
    groups: list[str] | None = None,
    periods: pd.DatetimeIndex | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build the four core inputs for calculate_brinson_attribution."""
    n_groups = len(wp)
    if groups is None:
        groups = [f"G{i}" for i in range(n_groups)]
    if periods is None:
        periods = MONTHLY_DATES[: len(rp_g)]

    w_p = pd.Series(wp, index=groups)
    w_b = pd.Series(wb, index=groups)
    port_ret  = pd.DataFrame(rp_g, index=periods, columns=groups)
    bench_ret = pd.DataFrame(rb_g, index=periods, columns=groups)
    return w_p, w_b, port_ret, bench_ret


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Basic reconciliation — BF
# ═══════════════════════════════════════════════════════════════════════════════

class TestReconciliationBF:
    """
    For every period:
        Σ_g [Alloc + Select + Inter] == r_p,t - r_b,t
    This must hold analytically when both weight vectors sum to 1.
    """

    def test_reconciliation_two_groups_three_periods(self):
        wp = [0.6, 0.4]
        wb = [0.5, 0.5]
        # [period][group]
        rp_g = [[0.04, 0.01], [-0.02, 0.03], [0.03, -0.01]]
        rb_g = [[0.03, 0.02], [-0.01, 0.02], [0.02, -0.02]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        period_sums = attr.groupby("Period")["Total Effect"].sum()
        assert_allclose(period_sums.values, active.values, atol=1e-12,
                        err_msg="BF: period attribution totals do not equal active return")

    def test_reconciliation_four_groups_five_periods(self):
        rng = np.random.default_rng(42)
        n_groups  = 4
        n_periods = 5

        wp = rng.dirichlet(np.ones(n_groups))
        wb = rng.dirichlet(np.ones(n_groups))
        rp_g = rng.normal(0.005, 0.03, (n_periods, n_groups)).tolist()
        rb_g = rng.normal(0.004, 0.025, (n_periods, n_groups)).tolist()

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp.tolist(), wb.tolist(), rp_g, rb_g)
        attr   = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        period_sums = attr.groupby("Period")["Total Effect"].sum()
        assert_allclose(period_sums.values, active.values, atol=1e-12,
                        err_msg="BF 4-group: reconciliation failed")

    def test_reconciliation_across_full_period_cumulative(self):
        """Cumulative sum of total effects equals sum of active returns."""
        wp = [0.7, 0.3]
        wb = [0.6, 0.4]
        rp_g = [[0.05, 0.02], [0.03, -0.01], [0.04, 0.01], [-0.02, 0.03]]
        rb_g = [[0.04, 0.01], [0.01, -0.005], [0.03, 0.02], [-0.01, 0.025]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr   = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        total_attr   = float(attr["Total Effect"].sum())
        total_active = float(active.sum())
        assert_allclose(total_attr, total_active, atol=1e-12,
                        err_msg="Cumulative attribution total does not equal cumulative active return")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Basic reconciliation — BHB (original 1986)
# ═══════════════════════════════════════════════════════════════════════════════

class TestReconciliationBHB:

    def test_reconciliation_bhb_two_groups(self):
        wp = [0.6, 0.4]
        wb = [0.5, 0.5]
        rp_g = [[0.04, 0.01], [-0.02, 0.03]]
        rb_g = [[0.03, 0.02], [-0.01, 0.02]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr   = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="bhb")
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        period_sums = attr.groupby("Period")["Total Effect"].sum()
        assert_allclose(period_sums.values, active.values, atol=1e-12,
                        err_msg="BHB: period attribution totals do not equal active return")

    def test_reconciliation_bhb_four_groups(self):
        rng = np.random.default_rng(99)
        n_groups  = 4
        n_periods = 6

        wp = rng.dirichlet(np.ones(n_groups))
        wb = rng.dirichlet(np.ones(n_groups))
        rp_g = rng.normal(0.005, 0.03, (n_periods, n_groups)).tolist()
        rb_g = rng.normal(0.004, 0.025, (n_periods, n_groups)).tolist()

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp.tolist(), wb.tolist(), rp_g, rb_g)
        attr   = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="bhb")
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        period_sums = attr.groupby("Period")["Total Effect"].sum()
        assert_allclose(period_sums.values, active.values, atol=1e-12,
                        err_msg="BHB 4-group: reconciliation failed")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Zero active weights
# Portfolio weights = benchmark weights → allocation and interaction = 0
# Active return is entirely from selection.
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroActiveWeights:

    def test_allocation_effect_zero_when_equal_weights(self):
        wp = [0.5, 0.5]
        wb = [0.5, 0.5]   # same as portfolio
        rp_g = [[0.05, 0.01], [0.03, -0.01], [0.04, 0.02]]
        rb_g = [[0.03, 0.02], [0.01, 0.00],  [0.02, 0.01]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")

        assert_allclose(attr["Alloc Effect"].values, 0.0, atol=1e-12,
                        err_msg="Allocation effect should be zero when portfolio = benchmark weights")

    def test_interaction_effect_zero_when_equal_weights(self):
        wp = [0.5, 0.5]
        wb = [0.5, 0.5]
        rp_g = [[0.05, 0.01]]
        rb_g = [[0.03, 0.02]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)

        assert_allclose(attr["Inter Effect"].values, 0.0, atol=1e-12,
                        err_msg="Interaction effect should be zero when active weights are zero")

    def test_selection_is_only_driver_when_equal_weights(self):
        """With equal weights, active return comes entirely from selection."""
        wp = [0.5, 0.5]
        wb = [0.5, 0.5]
        rp_g = [[0.06, 0.02]]
        rb_g = [[0.03, 0.01]]

        # oracle: sel_A = 0.5*(0.06-0.03)=0.015; sel_B = 0.5*(0.02-0.01)=0.005; total=0.02
        # active return = 0.5*0.06+0.5*0.02 - (0.5*0.03+0.5*0.01) = 0.04-0.02 = 0.02
        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)

        period_sel = attr.groupby("Period")["Select Effect"].sum().iloc[0]
        active     = calculate_period_active_return(w_p, w_b, port_ret, bench_ret).iloc[0]

        assert_allclose(period_sel, active, atol=1e-12,
                        err_msg="When active weights=0, all active return must come from selection")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Zero selection difference
# r_p,g = r_b,g for all groups → selection = 0, interaction = 0
# Active return comes only from allocation.
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroSelectionDifference:

    def test_selection_zero_when_group_returns_equal(self):
        wp = [0.7, 0.3]
        wb = [0.5, 0.5]
        # same returns for portfolio and benchmark in each group
        r_g = [[0.04, 0.01], [0.02, -0.01], [0.03, 0.02]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(
            wp, wb, r_g, r_g  # identical
        )
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)

        assert_allclose(attr["Select Effect"].values, 0.0, atol=1e-12,
                        err_msg="Selection should be zero when portfolio group return = benchmark group return")

    def test_interaction_zero_when_group_returns_equal(self):
        wp = [0.7, 0.3]
        wb = [0.5, 0.5]
        r_g = [[0.04, 0.01]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, r_g, r_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)

        assert_allclose(attr["Inter Effect"].values, 0.0, atol=1e-12,
                        err_msg="Interaction should be zero when group returns are equal")

    def test_allocation_is_only_driver_when_selection_zero(self):
        """With identical group returns, all active return comes from allocation."""
        wp = [0.7, 0.3]
        wb = [0.5, 0.5]
        r_g = [[0.04, 0.01]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, r_g, r_g)
        attr   = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        period_alloc = attr.groupby("Period")["Alloc Effect"].sum().iloc[0]
        assert_allclose(period_alloc, float(active.iloc[0]), atol=1e-12,
                        err_msg="When selection=0, all active return must come from allocation")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Identical portfolio and benchmark
# All weights equal AND all returns equal → all effects = 0
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdenticalPortfolioAndBenchmark:

    def test_all_effects_zero_when_portfolio_equals_benchmark(self):
        wp = [0.5, 0.3, 0.2]
        wb = [0.5, 0.3, 0.2]
        r_g = [[0.04, 0.02, 0.01], [-0.01, 0.03, 0.02]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, r_g, r_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)

        for effect_col in ["Alloc Effect", "Select Effect", "Inter Effect", "Total Effect"]:
            assert_allclose(attr[effect_col].values, 0.0, atol=1e-12,
                            err_msg=f"{effect_col} should be zero when portfolio = benchmark")

    def test_active_return_zero_when_portfolio_equals_benchmark(self):
        wp = [0.5, 0.3, 0.2]
        wb = [0.5, 0.3, 0.2]
        r_g = [[0.04, 0.02, 0.01]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, r_g, r_g)
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        assert_allclose(active.values, 0.0, atol=1e-12,
                        err_msg="Active return must be zero when portfolio = benchmark")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Overweight winner — BF allocation effect should be positive
# Condition: w_p,g > w_b,g AND r_b,g > r_b (group beats benchmark total)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverweightWinner:

    def test_allocation_positive_when_overweight_outperforming_group(self):
        """
        Oracle:
          G0: w_p=0.70, w_b=0.50, r_b,G0=0.05, r_b,G1=0.01
          r_b_total = 0.5*0.05 + 0.5*0.01 = 0.03
          BF alloc for G0 = (0.70-0.50) * (0.05 - 0.03) = 0.20 * 0.02 = +0.004
        """
        wp = [0.70, 0.30]
        wb = [0.50, 0.50]
        # r_b,G0=0.05 > r_b_total=0.03 → G0 is a winner in benchmark
        rp_g = [[0.05, 0.01]]   # portfolio earns same as bench within groups
        rb_g = [[0.05, 0.01]]   # (selection = 0 for clarity)

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")

        alloc_g0 = float(attr[attr["Group"] == "G0"]["Alloc Effect"].iloc[0])
        oracle   = 0.20 * (0.05 - 0.03)
        assert_allclose(alloc_g0, oracle, atol=1e-12,
                        err_msg="Overweight winner: allocation effect should match oracle")
        assert alloc_g0 > 0, "Overweight winner must produce positive allocation effect (BF)"

    def test_total_attribution_positive_overweight_winner(self):
        wp = [0.70, 0.30]
        wb = [0.50, 0.50]
        rp_g = [[0.05, 0.01]]
        rb_g = [[0.05, 0.01]]   # selection = 0, only allocation

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr   = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        assert float(active.iloc[0]) > 0, "Active return should be positive when overweighting winner"
        assert_allclose(attr.groupby("Period")["Total Effect"].sum().iloc[0],
                        float(active.iloc[0]), atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Overweight loser — BF allocation effect should be negative
# Condition: w_p,g > w_b,g AND r_b,g < r_b (group lags benchmark total)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverweightLoser:

    def test_allocation_negative_when_overweight_underperforming_group(self):
        """
        Oracle:
          G0: w_p=0.70, w_b=0.50, r_b,G0=0.01, r_b,G1=0.05
          r_b_total = 0.5*0.01 + 0.5*0.05 = 0.03
          BF alloc for G0 = (0.70-0.50) * (0.01 - 0.03) = 0.20 * (-0.02) = -0.004
        """
        wp = [0.70, 0.30]
        wb = [0.50, 0.50]
        # G0 underperforms benchmark total: r_b,G0=0.01 < r_b=0.03
        rp_g = [[0.01, 0.05]]
        rb_g = [[0.01, 0.05]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")

        alloc_g0 = float(attr[attr["Group"] == "G0"]["Alloc Effect"].iloc[0])
        oracle   = 0.20 * (0.01 - 0.03)
        assert_allclose(alloc_g0, oracle, atol=1e-12,
                        err_msg="Overweight loser: allocation effect should match oracle")
        assert alloc_g0 < 0, "Overweight loser must produce negative allocation effect (BF)"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Selection effect sign
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelectionEffect:

    def test_selection_positive_when_port_beats_bench_within_group(self):
        """
        Oracle: sel_G0 = w_b * (r_p,G0 - r_b,G0) = 0.5 * (0.06 - 0.03) = 0.015
        """
        wp = [0.5, 0.5]
        wb = [0.5, 0.5]   # equal weights → interaction = 0
        rp_g = [[0.06, 0.01]]
        rb_g = [[0.03, 0.01]]   # G0: port beats bench; G1: equal

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)

        sel_g0 = float(attr[attr["Group"] == "G0"]["Select Effect"].iloc[0])
        oracle = 0.5 * (0.06 - 0.03)
        assert_allclose(sel_g0, oracle, atol=1e-12,
                        err_msg="Selection effect for G0 does not match oracle")
        assert sel_g0 > 0, "Portfolio beats benchmark within group → selection must be positive"

    def test_selection_negative_when_port_lags_bench_within_group(self):
        """
        sel_G0 = 0.5 * (0.01 - 0.04) = -0.015
        """
        wp = [0.5, 0.5]
        wb = [0.5, 0.5]
        rp_g = [[0.01, 0.04]]
        rb_g = [[0.04, 0.04]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)

        sel_g0 = float(attr[attr["Group"] == "G0"]["Select Effect"].iloc[0])
        assert sel_g0 < 0, "Portfolio lags benchmark within group → selection must be negative"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Interaction effect formula verification
# Inter = (w_p,g - w_b,g) × (r_p,g - r_b,g)
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionEffect:

    def test_interaction_equals_active_weight_times_active_group_return(self):
        """
        For each (period, group): interaction = active_weight × (r_p,g - r_b,g)
        This is the same for both BF and BHB.
        """
        rng = np.random.default_rng(7)
        n_groups  = 3
        n_periods = 4

        wp = rng.dirichlet(np.ones(n_groups)).tolist()
        wb = rng.dirichlet(np.ones(n_groups)).tolist()
        rp_g = rng.normal(0.005, 0.03, (n_periods, n_groups)).tolist()
        rb_g = rng.normal(0.004, 0.025, (n_periods, n_groups)).tolist()

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)

        for _, row in attr.iterrows():
            oracle = row["Active Weight"] * (row["Port Return"] - row["Bench Return"])
            assert_allclose(row["Inter Effect"], oracle, atol=1e-12,
                            err_msg=f"Interaction formula mismatch at period={row['Period']}, group={row['Group']}")

    def test_interaction_positive_when_overweight_and_outperform(self):
        """Overweight group AND portfolio beats benchmark within group → positive interaction."""
        # G0: overweight (active_w > 0) AND outperforms (r_p > r_b) → inter > 0
        wp = [0.6, 0.4]
        wb = [0.4, 0.6]
        rp_g = [[0.05, 0.02]]
        rb_g = [[0.03, 0.03]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)

        inter_g0 = float(attr[attr["Group"] == "G0"]["Inter Effect"].iloc[0])
        # oracle: (0.6-0.4) * (0.05-0.03) = 0.2*0.02 = 0.004
        assert_allclose(inter_g0, 0.004, atol=1e-12)
        assert inter_g0 > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Weight normalisation — build_benchmark_weights
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightNormalisation:

    def test_equal_weight_benchmark_sums_to_one(self):
        assets = ["US Equities (S&P 500)", "US Aggregate Bonds", "Gold", "Cash Proxy (T-Bills)"]
        w = build_benchmark_weights(assets, DEFAULT_CLASSIFICATION, method="equal_weight")
        assert_allclose(w.sum(), 1.0, atol=1e-12)

    def test_equal_weight_all_identical(self):
        assets = ["A", "B", "C", "D"]
        classification = {a: "G" for a in assets}
        w = build_benchmark_weights(assets, classification, method="equal_weight")
        assert_allclose(w.values, 0.25, atol=1e-12)

    def test_group_weighted_benchmark_sums_to_one(self):
        assets = [
            "US Equities (S&P 500)", "UK Equities (FTSE 100)",
            "US Aggregate Bonds",
            "Gold",
            "Cash Proxy (T-Bills)",
        ]
        w = build_benchmark_weights(
            assets, DEFAULT_CLASSIFICATION,
            method="group_weighted",
            custom_group_weights=BENCHMARK_60_40_GROUP_WEIGHTS,
        )
        assert_allclose(w.sum(), 1.0, atol=1e-12)

    def test_group_weighted_equities_heavier_than_cash(self):
        """With 60/40 scheme, equities weight should greatly exceed cash weight."""
        assets = [
            "US Equities (S&P 500)", "UK Equities (FTSE 100)",
            "US Aggregate Bonds",
            "Cash Proxy (T-Bills)",
        ]
        w = build_benchmark_weights(
            assets, DEFAULT_CLASSIFICATION,
            method="group_weighted",
            custom_group_weights=BENCHMARK_60_40_GROUP_WEIGHTS,
        )
        equity_total = w["US Equities (S&P 500)"] + w["UK Equities (FTSE 100)"]
        cash_total   = w["Cash Proxy (T-Bills)"]
        assert equity_total > cash_total, "Equities should outweigh cash in 60/40 benchmark"

    def test_calculate_group_weights_sums_match_asset_weights(self):
        """Group weights must sum to the same total as asset weights."""
        asset_w = pd.Series({
            "US Equities (S&P 500)":  0.30,
            "US Aggregate Bonds":     0.25,
            "Gold":                   0.20,
            "Cash Proxy (T-Bills)":   0.25,
        })
        gw = calculate_group_weights(asset_w, DEFAULT_CLASSIFICATION)
        assert_allclose(gw.sum(), asset_w.sum(), atol=1e-12)

    def test_calculate_group_weights_equities_bucket(self):
        asset_w = pd.Series({
            "US Equities (S&P 500)":          0.30,
            "UK Equities (FTSE 100)":         0.20,
            "European Equities (Euro Stoxx)": 0.10,
            "US Aggregate Bonds":             0.40,
        })
        gw = calculate_group_weights(asset_w, DEFAULT_CLASSIFICATION)
        assert_allclose(float(gw["Equities"]), 0.60, atol=1e-12)

    def test_unknown_asset_classified_as_other(self):
        asset_w = pd.Series({"mystery_asset": 0.5, "US Aggregate Bonds": 0.5})
        gw = calculate_group_weights(asset_w, DEFAULT_CLASSIFICATION)
        assert "Other" in gw.index, "Unknown assets should be classified as 'Other'"
        assert_allclose(float(gw["Other"]), 0.5, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Cumulative attribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestCumulativeAttribution:

    def test_cumulative_total_matches_arithmetic_sum(self):
        """Final cumulative total must equal arithmetic sum of all period totals."""
        wp = [0.6, 0.4]
        wb = [0.5, 0.5]
        rp_g = [[0.04, 0.01], [0.03, -0.01], [0.05, 0.02], [-0.01, 0.03]]
        rb_g = [[0.03, 0.02], [0.01, 0.00],  [0.03, 0.02], [-0.02, 0.025]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)
        cum  = calculate_cumulative_attribution(attr)

        expected_cum_total = float(attr["Total Effect"].sum())
        actual_cum_total   = float(cum["Cum Total"].iloc[-1])
        assert_allclose(actual_cum_total, expected_cum_total, atol=1e-12,
                        err_msg="Final cumulative total does not equal sum of all period totals")

    def test_cumulative_is_monotonically_consistent(self):
        """Cumulative Alloc at period k = sum of Alloc Effects for periods 1..k."""
        rng = np.random.default_rng(55)
        wp = rng.dirichlet(np.ones(3)).tolist()
        wb = rng.dirichlet(np.ones(3)).tolist()
        n_periods = 5
        rp_g = rng.normal(0.005, 0.02, (n_periods, 3)).tolist()
        rb_g = rng.normal(0.004, 0.02, (n_periods, 3)).tolist()

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)
        cum  = calculate_cumulative_attribution(attr)

        period_alloc = attr.groupby("Period")["Alloc Effect"].sum()
        oracle_cum   = period_alloc.cumsum().values
        assert_allclose(cum["Cum Alloc"].values, oracle_cum, atol=1e-12,
                        err_msg="Cumulative allocation does not match arithmetic cumsum")

    def test_cumulative_has_same_number_of_rows_as_periods(self):
        wp = [0.5, 0.5]
        wb = [0.5, 0.5]
        n_periods = 6
        rp_g = [[0.01, 0.02]] * n_periods
        rb_g = [[0.005, 0.015]] * n_periods

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)
        cum  = calculate_cumulative_attribution(attr)

        assert len(cum) == n_periods, f"Expected {n_periods} rows, got {len(cum)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 12. IC proxy / allocation effectiveness
# ═══════════════════════════════════════════════════════════════════════════════

class TestICProxy:

    def test_ic_proxy_returns_dict_with_expected_keys(self):
        groups = ["G0", "G1", "G2"]
        active_w = pd.Series([0.1, -0.05, -0.05], index=groups)
        bench_group_rets = pd.DataFrame(
            {"G0": [0.04, 0.01, 0.03], "G1": [0.02, 0.02, 0.01], "G2": [0.01, 0.03, 0.02]},
            index=MONTHLY_DATES[:3],
        )
        bench_total_rets = pd.Series(
            [(0.04+0.02+0.01)/3, (0.01+0.02+0.03)/3, (0.03+0.01+0.02)/3],
            index=MONTHLY_DATES[:3],
        )
        result = calculate_ic_proxy(active_w, bench_group_rets, bench_total_rets)

        assert "ic_proxy"  in result
        assert "t_stat"    in result
        assert "n_periods" in result
        assert result["is_proxy"] is True, "IC proxy flag must always be True"

    def test_ic_proxy_is_marked_as_proxy_not_true_ic(self):
        """is_proxy must always be True regardless of data quality."""
        groups = ["G0", "G1"]
        active_w = pd.Series([0.2, -0.2], index=groups)
        bench_ret = pd.DataFrame(
            {"G0": [0.05, 0.02], "G1": [0.01, 0.04]}, index=MONTHLY_DATES[:2]
        )
        bench_total = pd.Series([0.03, 0.03], index=MONTHLY_DATES[:2])

        result = calculate_ic_proxy(active_w, bench_ret, bench_total)
        assert result["is_proxy"] is True

    def test_ic_proxy_returns_none_for_constant_active_weights(self):
        """
        If all active weights are identical, cross-sectional correlation is undefined.
        """
        groups = ["G0", "G1"]
        active_w = pd.Series([0.0, 0.0], index=groups)  # zero active weights, std=0
        bench_ret = pd.DataFrame(
            {"G0": [0.04, 0.02], "G1": [0.01, 0.03]}, index=MONTHLY_DATES[:2]
        )
        bench_total = pd.Series([0.025, 0.025], index=MONTHLY_DATES[:2])

        result = calculate_ic_proxy(active_w, bench_ret, bench_total)
        assert result["ic_proxy"] is None, "IC proxy should be None when active weights are all zero"

    def test_ic_proxy_returns_none_for_constant_group_returns(self):
        """
        If benchmark-relative group returns are identical every period,
        the cross-sectional correlation is undefined for those periods.
        """
        groups = ["G0", "G1"]
        active_w = pd.Series([0.1, -0.1], index=groups)
        # All periods: both groups earn exactly benchmark return → no dispersion
        bench_ret   = pd.DataFrame(
            {"G0": [0.03, 0.03, 0.03], "G1": [0.03, 0.03, 0.03]},
            index=MONTHLY_DATES[:3],
        )
        bench_total = pd.Series([0.03, 0.03, 0.03], index=MONTHLY_DATES[:3])

        result = calculate_ic_proxy(active_w, bench_ret, bench_total)
        # No valid periods (all group returns = benchmark total → std=0 every period)
        assert result["ic_proxy"] is None or result["n_periods"] == 0

    def test_ic_proxy_positive_when_overweighted_group_outperforms(self):
        """
        Overweight G0 (active_w > 0) and G0 outperforms benchmark total.
        Overweight G1 (active_w < 0) and G1 underperforms benchmark total.
        → Each period, positive correlation between active_w and group_active_return.
        → IC proxy should be positive.
        """
        groups = ["G0", "G1"]
        active_w = pd.Series([0.1, -0.1], index=groups)

        bench_ret = pd.DataFrame({
            "G0": [0.06, 0.05, 0.07],
            "G1": [0.01, 0.02, 0.00],
        }, index=MONTHLY_DATES[:3])

        bench_total = pd.Series([0.035, 0.035, 0.035], index=MONTHLY_DATES[:3])

        result = calculate_ic_proxy(active_w, bench_ret, bench_total)
        assert result["ic_proxy"] is not None
        assert result["ic_proxy"] > 0, "IC proxy should be positive when overweights align with outperformers"

    def test_ic_proxy_negative_when_overweighted_group_underperforms(self):
        """
        Overweight G0 but G0 consistently underperforms benchmark total.
        → IC proxy should be negative.
        """
        groups = ["G0", "G1"]
        active_w = pd.Series([0.1, -0.1], index=groups)

        bench_ret = pd.DataFrame({
            "G0": [0.01, 0.01, 0.01],
            "G1": [0.06, 0.07, 0.08],
        }, index=MONTHLY_DATES[:3])

        bench_total = pd.Series([0.035, 0.04, 0.045], index=MONTHLY_DATES[:3])

        result = calculate_ic_proxy(active_w, bench_ret, bench_total)
        assert result["ic_proxy"] is not None
        assert result["ic_proxy"] < 0, "IC proxy should be negative when overweights misalign with outperformers"


# ═══════════════════════════════════════════════════════════════════════════════
# 13. BHB vs BF — different allocation effects, same total
# ═══════════════════════════════════════════════════════════════════════════════

class TestBHBvsBF:

    def test_bhb_and_bf_give_same_total_attribution(self):
        rng = np.random.default_rng(13)
        n_groups  = 3
        n_periods = 4

        wp = rng.dirichlet(np.ones(n_groups)).tolist()
        wb = rng.dirichlet(np.ones(n_groups)).tolist()
        rp_g = rng.normal(0.005, 0.02, (n_periods, n_groups)).tolist()
        rb_g = rng.normal(0.004, 0.02, (n_periods, n_groups)).tolist()

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)

        attr_bf  = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")
        attr_bhb = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="bhb")

        # Same total
        assert_allclose(
            attr_bf.groupby("Period")["Total Effect"].sum().values,
            attr_bhb.groupby("Period")["Total Effect"].sum().values,
            atol=1e-12,
            err_msg="BHB and BF must give the same per-period total attribution",
        )

    def test_bhb_and_bf_give_different_allocation_effects(self):
        """Allocation effects should differ between BF and BHB (unless benchmark groups all return same)."""
        wp = [0.7, 0.3]
        wb = [0.5, 0.5]
        rp_g = [[0.05, 0.01]]   # different group returns → BF != BHB
        rb_g = [[0.06, 0.02]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr_bf  = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")
        attr_bhb = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="bhb")

        alloc_bf_g0  = float(attr_bf[attr_bf["Group"] == "G0"]["Alloc Effect"].iloc[0])
        alloc_bhb_g0 = float(attr_bhb[attr_bhb["Group"] == "G0"]["Alloc Effect"].iloc[0])

        assert abs(alloc_bf_g0 - alloc_bhb_g0) > 1e-8, (
            "BF and BHB should give different allocation effects when group returns differ"
        )

    def test_bhb_and_bf_share_identical_selection_effects(self):
        """Selection formula is the same in both methods."""
        wp = [0.6, 0.4]
        wb = [0.5, 0.5]
        rp_g = [[0.05, 0.02]]
        rb_g = [[0.03, 0.01]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr_bf  = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="brinson_fachler")
        attr_bhb = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="bhb")

        assert_allclose(
            attr_bf["Select Effect"].values,
            attr_bhb["Select Effect"].values,
            atol=1e-12,
            err_msg="Selection effects must be identical between BF and BHB",
        )

    def test_invalid_method_raises(self):
        wp = [0.5, 0.5]
        wb = [0.5, 0.5]
        rp_g = [[0.04, 0.02]]
        rb_g = [[0.03, 0.02]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        with pytest.raises(ValueError, match="brinson_fachler"):
            calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret, method="unknown")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Zero-weight group in portfolio (missing group)
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroWeightGroup:

    def test_zero_portfolio_weight_group_has_zero_selection_and_interaction(self):
        """
        If portfolio holds nothing in a group (w_p,g = 0), portfolio group return
        is undefined.  calculate_group_returns sets it equal to bench return,
        making selection and interaction zero.  Only allocation effect applies.
        """
        # Build daily returns for 3 assets: 2 in G0, 1 in G1
        dates = pd.date_range("2023-01-01", periods=31, freq="B")
        daily_rets = pd.DataFrame({
            "A": np.full(len(dates), 0.001),   # G0
            "B": np.full(len(dates), 0.001),   # G0
            "C": np.full(len(dates), 0.002),   # G1  — portfolio holds nothing here
        }, index=dates)

        classification = {"A": "G0", "B": "G0", "C": "G1"}

        # Portfolio: 60% A, 40% B, 0% C
        port_w  = pd.Series({"A": 0.6, "B": 0.4, "C": 0.0})
        bench_w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})

        port_ret, bench_ret = calculate_group_returns(
            daily_rets, port_w, bench_w, classification, freq="ME"
        )

        # Group weights
        port_gw  = calculate_group_weights(port_w,  classification)
        bench_gw = calculate_group_weights(bench_w, classification)

        attr = calculate_brinson_attribution(
            port_gw, bench_gw, port_ret, bench_ret, method="brinson_fachler"
        )

        g1_rows = attr[attr["Group"] == "G1"]
        assert_allclose(g1_rows["Select Effect"].values, 0.0, atol=1e-10,
                        err_msg="Zero portfolio weight group should have zero selection effect")
        assert_allclose(g1_rows["Inter Effect"].values, 0.0, atol=1e-10,
                        err_msg="Zero portfolio weight group should have zero interaction effect")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Validate reconciliation helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateReconciliation:

    def test_reconciliation_passes_for_correct_attribution(self):
        wp = [0.6, 0.4]
        wb = [0.5, 0.5]
        rp_g = [[0.04, 0.01], [0.03, -0.01]]
        rb_g = [[0.03, 0.02], [0.01, 0.00]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr   = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        result = validate_brinson_reconciliation(attr, active)
        assert result["pass"] is True, f"Reconciliation failed: max residual={result['max_residual']}"
        assert result["max_residual"] < 1e-8

    def test_reconciliation_fails_for_deliberately_corrupted_attribution(self):
        """Manually inject a residual and confirm the validator catches it."""
        wp = [0.6, 0.4]
        wb = [0.5, 0.5]
        rp_g = [[0.04, 0.01]]
        rb_g = [[0.03, 0.02]]

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr   = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        # Corrupt the active return to introduce a residual
        corrupted_active = active + 0.05

        result = validate_brinson_reconciliation(attr, corrupted_active, tolerance=1e-8)
        assert result["pass"] is False, "Validator should detect a 5% residual"
        assert result["max_residual"] > 0.04

    def test_n_periods_matches_attribution_data(self):
        wp = [0.5, 0.5]
        wb = [0.5, 0.5]
        n_periods = 5
        rp_g = [[0.01, 0.02]] * n_periods
        rb_g = [[0.005, 0.015]] * n_periods

        w_p, w_b, port_ret, bench_ret = _make_attribution_inputs(wp, wb, rp_g, rb_g)
        attr   = calculate_brinson_attribution(w_p, w_b, port_ret, bench_ret)
        active = calculate_period_active_return(w_p, w_b, port_ret, bench_ret)

        result = validate_brinson_reconciliation(attr, active)
        assert result["n_periods"] == n_periods


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: calculate_group_returns + attribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroupReturnsIntegration:

    def _make_daily_prices(self, assets_returns: dict[str, float], n_days: int = 63) -> pd.DataFrame:
        """Build price series with constant daily returns."""
        dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
        data = {a: (1 + r) ** np.arange(n_days) * 100.0 for a, r in assets_returns.items()}
        return pd.DataFrame(data, index=dates)

    def test_group_returns_sum_weighted_correctly(self):
        """
        Two assets A (G0) and B (G0) with constant daily returns 0.1% and 0.2%.
        Port weights: A=0.6, B=0.4.  Both in G0.

        We use exactly 22 prices starting 2023-01-02 with freq="B".
        January 2023 has exactly 22 weekdays (Mon 2 – Tue 31), so after
        pct_change().dropna() we get 21 daily returns all within January.
        resample("ME") yields one monthly period.

        Oracle: G0 monthly portfolio return = 0.6*(1.001^21 - 1) + 0.4*(1.002^21 - 1)
        """
        r_A, r_B = 0.001, 0.002
        # 22 prices → 21 return days, all within January 2023 → exactly one ME period
        prices = self._make_daily_prices({"A": r_A, "B": r_B}, n_days=22)
        daily_rets = prices.pct_change().dropna()

        classification = {"A": "G0", "B": "G0"}
        port_w  = pd.Series({"A": 0.6, "B": 0.4})
        bench_w = pd.Series({"A": 0.5, "B": 0.5})

        port_ret, bench_ret = calculate_group_returns(daily_rets, port_w, bench_w, classification)

        # oracle: 21 constant-return days compounded, then weighted
        n_ret_days = len(daily_rets)   # = 21
        monthly_A      = (1 + r_A) ** n_ret_days - 1
        monthly_B      = (1 + r_B) ** n_ret_days - 1
        oracle_port_G0 = 0.6 * monthly_A + 0.4 * monthly_B

        assert len(port_ret) == 1, (
            f"Expected exactly 1 monthly period, got {len(port_ret)}. "
            "Check that dates all fall within a single calendar month."
        )
        actual_port_G0 = float(port_ret["G0"].iloc[0])
        assert_allclose(actual_port_G0, oracle_port_G0, rtol=1e-6,
                        err_msg="Portfolio group return does not match weighted oracle")

    def test_reconciliation_end_to_end(self):
        """
        Full pipeline: daily returns → group returns → attribution → reconciliation.
        Two groups, constant daily returns.
        """
        prices = self._make_daily_prices({"A": 0.001, "B": 0.002, "C": 0.0015})
        daily_rets = prices.pct_change().dropna()

        classification = {"A": "G0", "B": "G0", "C": "G1"}
        port_w  = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
        bench_w = pd.Series({"A": 0.33, "B": 0.33, "C": 0.34})

        port_gw  = calculate_group_weights(port_w,  classification)
        bench_gw = calculate_group_weights(bench_w, classification)
        port_ret, bench_ret = calculate_group_returns(daily_rets, port_w, bench_w, classification)

        attr   = calculate_brinson_attribution(port_gw, bench_gw, port_ret, bench_ret)
        active = calculate_period_active_return(port_gw, bench_gw, port_ret, bench_ret)

        validation = validate_brinson_reconciliation(attr, active)
        assert validation["pass"], (
            f"End-to-end reconciliation failed: max residual = {validation['max_residual']}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 16. normalize_asset_name — handles hidden / invisible characters
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeAssetName:
    """
    normalize_asset_name must produce identical output for strings that are
    visually identical to a human but differ at the byte level.
    This is the root cause of the production overlap bug discovered 2026-06-08:
    `for a in pd.Series` iterates over VALUES (floats) not INDEX (strings),
    so the old check always returned empty overlap when weights was a pd.Series.
    """

    def test_clean_ascii_unchanged(self):
        assert normalize_asset_name("US Equities (S&P 500)") == "US Equities (S&P 500)"
        assert normalize_asset_name("US Aggregate Bonds") == "US Aggregate Bonds"

    def test_trailing_space_stripped(self):
        assert normalize_asset_name("US Aggregate Bonds ") == "US Aggregate Bonds"
        assert normalize_asset_name("  Gold  ") == "Gold"

    def test_leading_space_stripped(self):
        assert normalize_asset_name(" Cash Proxy (T-Bills)") == "Cash Proxy (T-Bills)"

    def test_internal_multiple_spaces_collapsed(self):
        assert normalize_asset_name("US  Aggregate  Bonds") == "US Aggregate Bonds"

    def test_non_breaking_space_normalised(self):
        nbspace = " "
        assert normalize_asset_name(f"Cash Proxy{nbspace}(T-Bills)") == "Cash Proxy (T-Bills)"

    def test_en_dash_normalised_to_hyphen(self):
        # en dash U+2013 — looks like hyphen in T-Bills
        assert normalize_asset_name("Cash Proxy (T–Bills)") == "Cash Proxy (T-Bills)"

    def test_em_dash_normalised_to_hyphen(self):
        assert normalize_asset_name("Cash Proxy (T—Bills)") == "Cash Proxy (T-Bills)"

    def test_non_breaking_hyphen_normalised(self):
        # U+2011 non-breaking hyphen — visually identical to U+002D
        assert normalize_asset_name("Cash Proxy (T‑Bills)") == "Cash Proxy (T-Bills)"

    def test_figure_dash_normalised(self):
        # U+2012 figure dash
        assert normalize_asset_name("Cash Proxy (T‒Bills)") == "Cash Proxy (T-Bills)"

    def test_zero_width_space_removed(self):
        # U+200B — completely invisible
        zwsp = "​"
        assert normalize_asset_name(f"US{zwsp} Aggregate Bonds") == "US Aggregate Bonds"

    def test_zero_width_non_joiner_removed(self):
        # U+200C
        zwnj = "‌"
        assert normalize_asset_name(f"Gold{zwnj}") == "Gold"

    def test_non_str_input_converted(self):
        import numpy as np
        # numpy string (numpy.str_) — comes from DataFrame column names
        assert normalize_asset_name(np.str_("US Aggregate Bonds")) == "US Aggregate Bonds"

    def test_two_visually_identical_strings_normalise_equal(self):
        """
        Simulate the exact production case: asset name from UNIVERSE dict
        (clean ASCII) vs name that went through some encoding pipeline
        and gained a non-breaking hyphen.
        """
        clean    = "Cash Proxy (T-Bills)"       # from config/settings.py
        encoded  = "Cash Proxy (T‑Bills)"  # non-breaking hyphen from some pipeline
        assert normalize_asset_name(clean) == normalize_asset_name(encoded)

    def test_nfc_normalisation(self):
        # é as precomposed (U+00E9) vs decomposed (U+0065 + U+0301)
        precomposed  = "é"
        decomposed   = "é"
        # both should normalise to the same NFC form
        assert normalize_asset_name(precomposed) == normalize_asset_name(decomposed)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. align_asset_names — root cause of overlap = empty for pd.Series weights
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlignAssetNames:
    """
    The production bug was:
        for a in pd.Series  → iterates over VALUES (floats 0.1, 0.2, ...)
        not over INDEX (strings 'US Aggregate Bonds', ...)
    So 0.1 in pd.Index(['US Aggregate Bonds', ...]) → False for all, overlap = [].

    align_asset_names uses weight_source.index which is always correct.
    """

    _CLASSIFICATION = {
        "US Aggregate Bonds": "Fixed Income",
        "Gold":               "Alternatives",
        "Cash Proxy (T-Bills)": "Cash",
        "US Equities":        "Equities",
    }
    _RET_COLS = ["US Aggregate Bonds", "Gold", "Cash Proxy (T-Bills)", "US Equities"]

    # ── Core overlap tests ────────────────────────────────────────────────────

    def test_full_overlap_pd_series(self):
        """pd.Series weights — the exact production failure scenario."""
        weights = pd.Series({
            "US Aggregate Bonds":   0.25,
            "Gold":                 0.25,
            "Cash Proxy (T-Bills)": 0.25,
            "US Equities":          0.25,
        })
        result = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        assert sorted(result["available_assets"]) == sorted(self._RET_COLS), (
            "Full overlap pd.Series: all 4 assets should be matched"
        )

    def test_full_overlap_dict(self):
        """dict weights — should work (original dict iteration gives keys)."""
        weights = {
            "US Aggregate Bonds":   0.25,
            "Gold":                 0.25,
            "Cash Proxy (T-Bills)": 0.25,
            "US Equities":          0.25,
        }
        result = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        assert sorted(result["available_assets"]) == sorted(self._RET_COLS)

    def test_full_overlap_dataframe(self):
        """DataFrame weights (last row used)."""
        df = pd.DataFrame(
            [[0.25, 0.25, 0.25, 0.25]],
            columns=self._RET_COLS,
        )
        result = align_asset_names(df, self._RET_COLS, self._CLASSIFICATION)
        assert sorted(result["available_assets"]) == sorted(self._RET_COLS)

    def test_no_true_overlap_returns_empty(self):
        """Genuinely different names → available_assets empty."""
        weights = pd.Series({"AssetX": 0.5, "AssetY": 0.5})
        result  = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        assert result["available_assets"] == [], (
            "No overlap: available_assets should be empty"
        )

    def test_partial_overlap(self):
        """Weights for 2 of 4 return columns → only 2 matched."""
        weights = pd.Series({"Gold": 0.6, "Unknown": 0.4})
        result  = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        assert result["available_assets"] == ["Gold"]

    # ── Unicode normalisation ─────────────────────────────────────────────────

    def test_trailing_space_in_weights_matches_clean_column(self):
        weights = pd.Series({
            "US Aggregate Bonds ":   0.5,   # trailing space
            "Gold":                  0.5,
        })
        result = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        assert "US Aggregate Bonds" in result["available_assets"], (
            "Trailing space in weight key should match clean column name"
        )

    def test_trailing_space_in_column_matches_clean_weight(self):
        """
        Column has trailing space; weight key is clean.
        Both normalize to the same string so they match.
        The canonical name returned is the RAW return column name (with trailing space)
        — this is by design so that asset_returns[available_assets] still works.
        """
        cols_with_space = ["US Aggregate Bonds ", "Gold", "Cash Proxy (T-Bills)", "US Equities"]
        weights = pd.Series({
            "US Aggregate Bonds": 0.5,
            "Gold":               0.5,
        })
        result = align_asset_names(weights, cols_with_space, self._CLASSIFICATION)
        assert len(result["available_assets"]) == 2, (
            "Trailing space in column should match clean weight key after normalisation"
        )
        # Canonical name comes from return column as-is so DataFrame slicing still works.
        # "US Aggregate Bonds " (with space) — the raw column name — must be present.
        assert "US Aggregate Bonds " in result["available_assets"], (
            "Canonical name should be the raw return column name (with trailing space) "
            "so that asset_returns[available_assets] succeeds"
        )

    def test_en_dash_in_weight_matches_hyphen_in_column(self):
        weights = pd.Series({
            "Cash Proxy (T–Bills)": 0.5,   # en dash
            "Gold":                      0.5,
        })
        result = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        assert "Cash Proxy (T-Bills)" in result["available_assets"], (
            "En dash in weight key should match hyphen in column after normalisation"
        )

    def test_non_breaking_hyphen_in_column_matches_clean_weight(self):
        cols_nbh = ["Cash Proxy (T‑Bills)", "Gold", "US Aggregate Bonds", "US Equities"]
        weights  = pd.Series({"Cash Proxy (T-Bills)": 0.6, "Gold": 0.4})
        result   = align_asset_names(weights, cols_nbh, self._CLASSIFICATION)
        # canonical name comes from column (which has non-breaking hyphen)
        # but available_assets uses canonical (column) name
        assert len(result["available_assets"]) == 2, (
            "Non-breaking hyphen in column should match regular hyphen in weight"
        )

    def test_non_breaking_space_in_weight_matches_clean_column(self):
        weights = pd.Series({
            "Cash Proxy (T-Bills)": 0.5,   # NBSP
            "Gold":                       0.5,
        })
        result = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        assert "Cash Proxy (T-Bills)" in result["available_assets"]

    # ── Weight renormalisation ─────────────────────────────────────────────────

    def test_weights_renormalised_after_partial_drop(self):
        """If some assets dropped, remaining weights are renormalised to sum to 1."""
        weights = pd.Series({
            "Gold":    0.30,
            "Unknown": 0.70,   # not in returns → dropped
        })
        result = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        assert len(result["available_assets"]) == 1
        assert_allclose(result["port_weights"].sum(), 1.0, atol=1e-12)

    def test_full_overlap_weights_sum_to_one(self):
        weights = pd.Series({
            "US Aggregate Bonds":   0.25,
            "Gold":                 0.25,
            "Cash Proxy (T-Bills)": 0.25,
            "US Equities":          0.25,
        })
        result = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        assert_allclose(result["port_weights"].sum(), 1.0, atol=1e-12)

    # ── Classification mapping ────────────────────────────────────────────────

    def test_canonical_classification_keys_match_available_assets(self):
        """classification keys must exactly match available_assets values."""
        weights = pd.Series({k: 0.25 for k in self._RET_COLS})
        result  = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        for asset in result["available_assets"]:
            assert asset in result["canonical_classification"], (
                f"'{asset}' in available_assets but not in canonical_classification"
            )

    def test_classification_uses_canonical_names_not_normalised_keys(self):
        """Canonical names (from return columns) are used, not the raw weight keys."""
        weights = pd.Series({
            "Gold ": 0.4,   # trailing space
            "US Aggregate Bonds": 0.6,
        })
        result = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        # Canonical name should be "Gold" (from return column), not "Gold "
        assert "Gold" in result["canonical_classification"]
        assert "Gold " not in result["canonical_classification"]

    # ── Debug output ─────────────────────────────────────────────────────────

    def test_debug_contains_required_keys(self):
        weights = pd.Series({"Gold": 0.5, "Unknown": 0.5})
        result  = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        required = {
            "raw_weight_keys", "raw_ret_cols",
            "norm_weight_keys", "norm_ret_cols",
            "repr_weight_keys", "repr_ret_cols",
            "n_common", "n_weight_keys", "n_ret_cols",
            "only_in_returns", "only_in_weights", "common_assets",
        }
        assert required <= set(result["debug"].keys()), (
            f"debug missing: {required - set(result['debug'].keys())}"
        )

    def test_debug_repr_shows_actual_hidden_characters(self):
        """repr() of a string with trailing space shows the space explicitly."""
        weights = pd.Series({"Gold ": 0.5, "US Aggregate Bonds": 0.5})
        result  = align_asset_names(weights, self._RET_COLS, self._CLASSIFICATION)
        repr_keys = result["debug"]["repr_weight_keys"]
        # One of the repr strings should contain a trailing space inside quotes
        assert any("'Gold '" in r for r in repr_keys), (
            "repr_weight_keys should expose trailing space in 'Gold '"
        )
