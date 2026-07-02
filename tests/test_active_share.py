"""
Tests for analytics.active_share.

Test coverage:
 1.  Active Share = 0 when portfolio is identical to benchmark
 2.  Active Share = 1 when portfolio has no asset overlap with benchmark
 3.  Active Share formula — specific known value (single-asset benchmark)
 4.  Active Share formula — multi-asset benchmark
 5.  Active Share is in [0, 1] for any valid portfolio
 6.  Active Share handles the union of assets correctly (missing → 0)
 7.  Active Share is symmetric: AS(A vs B) = AS(B vs A)
 8.  Normalisation: unnormalised weights give same AS as normalised weights
 9.  Tracking Error = 0 when portfolio returns equal benchmark returns
10.  Tracking Error is correctly annualised (std × sqrt(252))
11.  Tracking Error > 0 for distinct return series
12.  Tracking Error handles misaligned dates via reindex
13.  Active weight breakdown sums to approximately 0
14.  Active weight breakdown has required columns
15.  Active weight breakdown sorted descending by active weight
16.  build_benchmark_weights finds SPY as 'US Equities (S&P 500)' = 1.0
17.  build_benchmark_weights returns Series summing to 1.0
18.  Classification labels map to correct buckets
19.  Quadrant labels cover all four cases
20.  TE classification covers all three bands
21.  Active Share = 1 - w_benchmark_asset for single-asset benchmark (analytic check)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analytics.active_share import (
    calculate_active_share,
    calculate_tracking_error,
    active_weight_breakdown,
    build_benchmark_weights,
    active_share_classification,
    te_classification,
    quadrant_label,
    AS_CLOSET_INDEX,
    AS_MODERATE,
    AS_GENUINE,
    TE_THRESHOLD_LOW,
    TE_THRESHOLD_HIGH,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0003, 0.01, n), index=dates, name="Portfolio")


PORT_RETS  = make_returns(seed=42)
BENCH_RETS = make_returns(seed=99)   # independent benchmark returns


# ---------------------------------------------------------------------------
# 1. Active Share = 0 when portfolio == benchmark
# ---------------------------------------------------------------------------

def test_active_share_zero_identical():
    w = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    np.testing.assert_allclose(calculate_active_share(w, w), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Active Share = 1 when no overlap
# ---------------------------------------------------------------------------

def test_active_share_one_no_overlap():
    portfolio  = pd.Series({"A": 0.5, "B": 0.5})
    benchmark  = pd.Series({"C": 0.7, "D": 0.3})
    np.testing.assert_allclose(calculate_active_share(portfolio, benchmark), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Formula check — single-asset benchmark (analytic)
#    When benchmark is 100% in asset X:
#    AS = 1 - w_X  (since Σ_{others} w_i = 1 - w_X = the other half)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w_spy", [0.0, 0.10, 0.30, 0.50, 0.80, 1.00])
def test_active_share_single_asset_benchmark_analytic(w_spy):
    """AS = 1 - w_benchmark_asset for any single-asset benchmark."""
    n_others = 5
    w_others = (1 - w_spy) / n_others if n_others else 0
    labels_others = [f"Asset{i}" for i in range(n_others)]
    portfolio = pd.Series(
        {**{"Benchmark": w_spy}, **{lbl: w_others for lbl in labels_others}}
    )
    benchmark = pd.Series({"Benchmark": 1.0})
    expected  = 1.0 - w_spy
    np.testing.assert_allclose(
        calculate_active_share(portfolio, benchmark), expected, atol=1e-10,
        err_msg=f"AS should be {expected:.2f} when w_spy={w_spy}",
    )


# ---------------------------------------------------------------------------
# 4. Formula check — multi-asset benchmark
# ---------------------------------------------------------------------------

def test_active_share_multi_asset_benchmark():
    portfolio = pd.Series({"A": 0.50, "B": 0.30, "C": 0.20})
    benchmark = pd.Series({"A": 0.40, "B": 0.40, "C": 0.20})
    # |0.50-0.40| + |0.30-0.40| + |0.20-0.20| = 0.10 + 0.10 + 0 = 0.20 → AS = 0.10
    expected = 0.5 * (0.10 + 0.10 + 0.0)
    np.testing.assert_allclose(
        calculate_active_share(portfolio, benchmark), expected, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 5. Active Share is always in [0, 1]
# ---------------------------------------------------------------------------

def test_active_share_bounds():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(2, 10)
        raw_p = rng.dirichlet(np.ones(n))
        raw_b = rng.dirichlet(np.ones(n))
        labels = [f"A{i}" for i in range(n)]
        as_val = calculate_active_share(
            pd.Series(raw_p, index=labels),
            pd.Series(raw_b, index=labels),
        )
        assert 0.0 <= as_val <= 1.0 + 1e-10, f"AS = {as_val} is outside [0, 1]"


# ---------------------------------------------------------------------------
# 6. Union of assets — assets missing from one series default to 0
# ---------------------------------------------------------------------------

def test_active_share_union_handling():
    portfolio = pd.Series({"A": 0.6, "B": 0.4})
    benchmark = pd.Series({"A": 0.5, "C": 0.5})   # C not in portfolio
    # w: A=0.6, B=0.4, C=0    bench: A=0.5, B=0, C=0.5
    # |0.6-0.5| + |0.4-0| + |0-0.5| = 0.1 + 0.4 + 0.5 = 1.0 → AS = 0.5
    expected = 0.5 * (0.1 + 0.4 + 0.5)
    np.testing.assert_allclose(
        calculate_active_share(portfolio, benchmark), expected, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 7. Active Share is symmetric
# ---------------------------------------------------------------------------

def test_active_share_symmetric():
    p = pd.Series({"A": 0.3, "B": 0.5, "C": 0.2})
    b = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    np.testing.assert_allclose(
        calculate_active_share(p, b),
        calculate_active_share(b, p),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 8. Normalisation: double the weights → same AS
# ---------------------------------------------------------------------------

def test_active_share_normalisation():
    p = pd.Series({"A": 0.4, "B": 0.6})
    b = pd.Series({"A": 0.7, "B": 0.3})
    as_normalised   = calculate_active_share(p, b)
    as_unnormalised = calculate_active_share(p * 2, b * 3)   # both get renormalised
    np.testing.assert_allclose(as_normalised, as_unnormalised, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. Tracking Error = 0 when returns are identical
# ---------------------------------------------------------------------------

def test_tracking_error_zero_identical():
    te = calculate_tracking_error(PORT_RETS, PORT_RETS)
    np.testing.assert_allclose(te, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 10. Tracking Error correctly annualised
# ---------------------------------------------------------------------------

def test_tracking_error_annualisation():
    active_daily = PORT_RETS - BENCH_RETS.reindex(PORT_RETS.index).fillna(0)
    expected_te = float(active_daily.std(ddof=1) * np.sqrt(252))
    computed_te = calculate_tracking_error(PORT_RETS, BENCH_RETS)
    np.testing.assert_allclose(computed_te, expected_te, rtol=1e-10)


# ---------------------------------------------------------------------------
# 11. Tracking Error > 0 for distinct return series
# ---------------------------------------------------------------------------

def test_tracking_error_positive_for_distinct_series():
    te = calculate_tracking_error(PORT_RETS, BENCH_RETS)
    assert te > 0.0, "TE must be positive when portfolio and benchmark differ"


# ---------------------------------------------------------------------------
# 12. Tracking Error handles misaligned dates
# ---------------------------------------------------------------------------

def test_tracking_error_misaligned_dates():
    port  = PORT_RETS.iloc[:300]
    bench = BENCH_RETS.iloc[100:]   # overlap: rows 100-299
    te = calculate_tracking_error(port, bench)
    assert te >= 0.0
    assert np.isfinite(te)


# ---------------------------------------------------------------------------
# 13. Active weight breakdown sums to 0
# ---------------------------------------------------------------------------

def test_active_weight_breakdown_sums_to_zero():
    p = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
    b = pd.Series({"A": 1.0})
    df = active_weight_breakdown(p, b)
    np.testing.assert_allclose(
        df["Active Weight"].sum(), 0.0, atol=1e-12,
        err_msg="Active weights must sum to 0 (overweights = underweights)",
    )


# ---------------------------------------------------------------------------
# 14. Active weight breakdown has required columns
# ---------------------------------------------------------------------------

def test_active_weight_breakdown_columns():
    p = pd.Series({"A": 0.6, "B": 0.4})
    b = pd.Series({"A": 1.0})
    df = active_weight_breakdown(p, b)
    required = {"Asset", "Portfolio Weight", "Benchmark Weight", "Active Weight"}
    assert required.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# 15. Active weight breakdown sorted descending
# ---------------------------------------------------------------------------

def test_active_weight_breakdown_sorted():
    p = pd.Series({"A": 0.2, "B": 0.5, "C": 0.3})
    b = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    df = active_weight_breakdown(p, b)
    aw = df["Active Weight"].values
    assert (aw[:-1] >= aw[1:]).all(), (
        "Active weights must be sorted descending"
    )


# ---------------------------------------------------------------------------
# 16. build_benchmark_weights identifies SPY correctly
# ---------------------------------------------------------------------------

def test_build_benchmark_weights_finds_spy():
    portfolio = pd.Series({
        "US Equities (S&P 500)": 0.30,
        "US Aggregate Bonds":    0.40,
        "Gold":                  0.30,
    })
    bench_w = build_benchmark_weights(portfolio)
    assert "US Equities (S&P 500)" in bench_w.index, (
        "Benchmark weights must include the SPY label"
    )
    assert bench_w["US Equities (S&P 500)"] == 1.0, (
        "SPY should have weight 1.0 in the benchmark"
    )


# ---------------------------------------------------------------------------
# 17. build_benchmark_weights sums to 1.0
# ---------------------------------------------------------------------------

def test_build_benchmark_weights_sums_to_one():
    portfolio = pd.Series({
        "US Equities (S&P 500)": 0.5,
        "Gold": 0.3,
        "US Aggregate Bonds": 0.2,
    })
    bench_w = build_benchmark_weights(portfolio)
    np.testing.assert_allclose(
        bench_w.sum(), 1.0, atol=1e-12,
        err_msg="Benchmark weights must sum to 1.0",
    )


# ---------------------------------------------------------------------------
# 18. Classification labels at the correct bucket boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("as_val,expected_label", [
    (0.05,  "Closet Indexer"),
    (0.19,  "Closet Indexer"),
    (0.20,  "Moderately Active"),
    (0.50,  "Moderately Active"),
    (0.60,  "Genuinely Active"),
    (0.79,  "Genuinely Active"),
    (0.80,  "High Conviction"),
    (0.95,  "High Conviction"),
    (1.00,  "High Conviction"),
])
def test_classification_labels(as_val, expected_label):
    result = active_share_classification(as_val)
    assert result["label"] == expected_label, (
        f"AS={as_val}: expected '{expected_label}', got '{result['label']}'"
    )


def test_classification_has_required_keys():
    for as_val in [0.05, 0.40, 0.70, 0.90]:
        result = active_share_classification(as_val)
        assert "label"       in result
        assert "description" in result
        assert "color"       in result


# ---------------------------------------------------------------------------
# 19. Quadrant labels cover all four cases
# ---------------------------------------------------------------------------

def test_quadrant_labels_all_cases():
    # High AS, High TE
    q1 = quadrant_label(0.70, 0.10)
    assert isinstance(q1, str) and len(q1) > 0

    # High AS, Low TE
    q2 = quadrant_label(0.70, 0.02)
    assert isinstance(q2, str) and len(q2) > 0

    # Low AS, Low TE
    q3 = quadrant_label(0.10, 0.02)
    assert isinstance(q3, str) and len(q3) > 0

    # Low AS, High TE
    q4 = quadrant_label(0.10, 0.10)
    assert isinstance(q4, str) and len(q4) > 0

    # The high-AS + high-TE quadrant should be the "active" one
    assert "Active" in q1 or "active" in q1.lower()
    # The low-AS + low-TE quadrant should mention closet indexing
    assert "Closet" in q3 or "closet" in q3.lower()


# ---------------------------------------------------------------------------
# 20. TE classification
# ---------------------------------------------------------------------------

def test_te_classification_bands():
    assert te_classification(0.02) == "Low TE"
    assert te_classification(TE_THRESHOLD_LOW - 0.001) == "Low TE"
    assert te_classification(TE_THRESHOLD_LOW) == "Moderate TE"
    assert te_classification(0.06) == "Moderate TE"
    assert te_classification(TE_THRESHOLD_HIGH) == "High TE"
    assert te_classification(0.15) == "High TE"


# ---------------------------------------------------------------------------
# 21. Active Share + Tracking Error integration — typical multi-asset portfolio
# ---------------------------------------------------------------------------

def test_active_share_typical_multi_asset():
    """
    Typical multi-asset portfolio vs SPY benchmark:
    expected Active Share > 0.5 because most assets are not SPY.
    """
    portfolio = pd.Series({
        "US Equities (S&P 500)":       0.25,
        "UK Equities (FTSE 100)":      0.10,
        "US Aggregate Bonds":          0.25,
        "Global Bonds (Hedged)":       0.15,
        "Gold":                        0.15,
        "Emerging Markets":            0.10,
    })
    bench_w = build_benchmark_weights(portfolio)
    as_val  = calculate_active_share(portfolio, bench_w)

    # 25% is in SPY → AS = 1 - 0.25 = 0.75
    np.testing.assert_allclose(as_val, 0.75, atol=1e-10)
    assert active_share_classification(as_val)["label"] == "Genuinely Active"
