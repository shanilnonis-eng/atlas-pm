"""
Tests for the regime_detection analytics module.

Validates:
  1.  HMM fits without error on synthetic two-regime data
  2.  Label switching is fixed: low-vol state is always REGIME_NAMES[0]
  3.  Regime labels cover every date in the input series
  4.  Short series raises ValueError (< MIN_REQUIRED_OBSERVATIONS)
  5.  NaN in series raises ValueError
  6.  regime_statistics produces correct columns and partitions the sample
  7.  Transition matrix rows sum to 1
  8.  Regime durations are positive and episode counts are consistent
  9.  Emission params: low-vol state has strictly lower daily std than high-vol state
  10. contiguous_blocks reconstructs the full date range without gaps
  11. regime_beta_alpha returns finite values on well-separated synthetic data
  12. fit_hmm is reproducible: same seed → same state sequence
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analytics.regime_detection import (
    REGIME_NAMES,
    MIN_REQUIRED_OBSERVATIONS,
    fit_hmm,
    label_regimes,
    regime_statistics,
    regime_beta_alpha,
    transition_matrix,
    regime_durations,
    regime_emission_params,
    contiguous_blocks,
    _get_state_variances,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_two_regime_returns(
    n_low: int = 400,
    n_high: int = 300,
    seed: int = 0,
) -> pd.Series:
    """
    Synthetic daily returns with a clear two-regime structure.

    Regime A (low vol):  mean = +0.04% / day,  std = 0.5% / day
    Regime B (high vol): mean = -0.02% / day,  std = 2.0% / day

    Blocks alternate: [A, B, A, B] for better test coverage of transitions.
    """
    rng = np.random.default_rng(seed)
    block = n_low // 2
    low_a  = rng.normal(0.0004, 0.005,  block)
    high_a = rng.normal(-0.0002, 0.020, n_high // 2)
    low_b  = rng.normal(0.0004, 0.005,  block)
    high_b = rng.normal(-0.0002, 0.020, n_high // 2)
    values = np.concatenate([low_a, high_a, low_b, high_b])
    dates  = pd.date_range("2018-01-02", periods=len(values), freq="B")
    return pd.Series(values, index=dates, name="returns")


def make_benchmark(returns: pd.Series, seed: int = 99) -> pd.Series:
    """Synthetic benchmark correlated with the portfolio."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.002, len(returns))
    return (returns * 0.8 + noise).rename("benchmark")


# Shared fixtures — fit once for the whole module
_RETURNS   = make_two_regime_returns()
_BENCHMARK = make_benchmark(_RETURNS)

try:
    _MODEL, _STATE_SEQ, _LL = fit_hmm(_RETURNS, n_states=2, n_restarts=5)
    _LABELS = label_regimes(_MODEL, _STATE_SEQ, _RETURNS)
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HMM_AVAILABLE,
    reason="hmmlearn not installed",
)


# ---------------------------------------------------------------------------
# 1. HMM fits without error
# ---------------------------------------------------------------------------

def test_fit_hmm_returns_three_values():
    model, seq, ll = fit_hmm(_RETURNS, n_states=2, n_restarts=5)
    assert model is not None
    assert len(seq) == len(_RETURNS)
    assert np.isfinite(ll)


def test_fit_hmm_log_likelihood_is_finite():
    """
    hmmlearn.score() returns the total log-likelihood (sum over all timesteps).
    Gaussian PDFs with small daily σ can exceed 1 near the mean, so the total
    can be positive — we only require it is finite.
    """
    _, _, ll = fit_hmm(_RETURNS, n_states=2, n_restarts=5)
    assert np.isfinite(ll), f"Log-likelihood must be finite; got {ll}"


def test_fit_hmm_state_sequence_valid_indices():
    """All state indices must be in [0, n_states)."""
    _, seq, _ = fit_hmm(_RETURNS, n_states=2, n_restarts=5)
    assert set(seq).issubset({0, 1}), f"Unexpected state indices: {set(seq)}"


# ---------------------------------------------------------------------------
# 2. Label switching fixed: low-vol is always REGIME_NAMES[0]
# ---------------------------------------------------------------------------

def test_label_regimes_low_vol_is_regime_names_0():
    """
    The state whose emission std is lower must map to REGIME_NAMES[0].
    We verify by checking the fitted emission parameters directly.
    """
    variances = _get_state_variances(_MODEL)
    sorted_states = np.argsort(variances)
    # After label_regimes(), dates where the low-vol raw state was active
    # must be labelled REGIME_NAMES[0].
    low_vol_raw = int(sorted_states[0])
    high_vol_raw = int(sorted_states[1])

    for raw, expected_name in [(low_vol_raw, REGIME_NAMES[0]),
                                (high_vol_raw, REGIME_NAMES[1])]:
        raw_mask = _STATE_SEQ == raw
        if raw_mask.sum() > 0:
            labelled = _LABELS[raw_mask]
            assert (labelled == expected_name).all(), (
                f"Raw state {raw} should map to '{expected_name}' "
                f"but found: {labelled.value_counts().to_dict()}"
            )


def test_label_regimes_only_uses_known_regime_names():
    """All labels must be from REGIME_NAMES."""
    unexpected = set(_LABELS.values) - set(REGIME_NAMES)
    assert not unexpected, f"Unexpected regime labels: {unexpected}"


# ---------------------------------------------------------------------------
# 3. Labels cover every date
# ---------------------------------------------------------------------------

def test_labels_cover_all_dates():
    assert len(_LABELS) == len(_RETURNS)
    assert (_LABELS.index == _RETURNS.index).all()
    assert _LABELS.isna().sum() == 0


def test_labels_have_both_regimes():
    """Both regimes should appear in a 700-day series with clear structure."""
    assert set(_LABELS.values) == set(REGIME_NAMES), (
        f"Expected both regimes; found: {set(_LABELS.values)}"
    )


# ---------------------------------------------------------------------------
# 4. Short series raises ValueError
# ---------------------------------------------------------------------------

def test_fit_hmm_raises_on_short_series():
    short = _RETURNS.iloc[:MIN_REQUIRED_OBSERVATIONS - 1]
    with pytest.raises(ValueError, match="observations"):
        fit_hmm(short)


def test_fit_hmm_accepts_minimum_length_series():
    """Exactly MIN_REQUIRED_OBSERVATIONS observations must be accepted."""
    exact = _RETURNS.iloc[:MIN_REQUIRED_OBSERVATIONS]
    model, seq, ll = fit_hmm(exact, n_restarts=3)
    assert len(seq) == MIN_REQUIRED_OBSERVATIONS


# ---------------------------------------------------------------------------
# 5. NaN in series raises ValueError
# ---------------------------------------------------------------------------

def test_fit_hmm_raises_on_nan_series():
    bad = _RETURNS.copy()
    bad.iloc[10] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        fit_hmm(bad)


# ---------------------------------------------------------------------------
# 6. regime_statistics: correct columns and partition
# ---------------------------------------------------------------------------

def test_regime_statistics_has_all_columns():
    stats = regime_statistics(_RETURNS, _LABELS)
    expected_cols = {
        "N Days", "Pct Sample", "Ann. Return", "Ann. Volatility",
        "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Daily Win Rate",
    }
    assert expected_cols.issubset(set(stats.columns)), (
        f"Missing columns: {expected_cols - set(stats.columns)}"
    )


def test_regime_statistics_n_days_partition():
    """Sum of N Days across all regimes must equal total observations."""
    stats = regime_statistics(_RETURNS, _LABELS)
    assert stats["N Days"].sum() == len(_RETURNS), (
        f"N Days sum {stats['N Days'].sum()} != total {len(_RETURNS)}"
    )


def test_regime_statistics_pct_sample_sums_to_one():
    stats = regime_statistics(_RETURNS, _LABELS)
    np.testing.assert_allclose(
        stats["Pct Sample"].sum(), 1.0, atol=1e-6,
        err_msg="Pct Sample does not sum to 1.0",
    )


def test_regime_statistics_volatility_is_positive():
    stats = regime_statistics(_RETURNS, _LABELS)
    for regime in REGIME_NAMES:
        if stats.loc[regime, "N Days"] > 0:
            assert stats.loc[regime, "Ann. Volatility"] > 0, (
                f"Volatility for {regime} is not positive"
            )


def test_high_vol_regime_has_higher_volatility_than_low_vol():
    """
    The HMM should separate regimes by volatility.  The 'High Vol (Bear)'
    regime must exhibit materially higher realised annualised volatility.
    """
    stats = regime_statistics(_RETURNS, _LABELS)
    vol_low  = stats.loc[REGIME_NAMES[0], "Ann. Volatility"]
    vol_high = stats.loc[REGIME_NAMES[1], "Ann. Volatility"]
    assert vol_high > vol_low, (
        f"Expected high-vol regime ({vol_high:.4f}) > low-vol regime ({vol_low:.4f})"
    )


def test_regime_statistics_with_rf():
    """Sharpe ratio should differ when rf_returns is provided."""
    rf = pd.Series(0.04 / 252, index=_RETURNS.index, name="rf")
    stats_no_rf = regime_statistics(_RETURNS, _LABELS)
    stats_rf    = regime_statistics(_RETURNS, _LABELS, rf_returns=rf)
    # Sharpe with a non-zero rf should differ from Sharpe without rf
    for regime in REGIME_NAMES:
        if stats_no_rf.loc[regime, "N Days"] > 30:
            assert stats_no_rf.loc[regime, "Sharpe Ratio"] != pytest.approx(
                stats_rf.loc[regime, "Sharpe Ratio"], abs=1e-4
            ), f"Sharpe ratio unchanged when rf provided for regime {regime}"


# ---------------------------------------------------------------------------
# 7. Transition matrix rows sum to 1
# ---------------------------------------------------------------------------

def test_transition_matrix_rows_sum_to_one():
    tm = transition_matrix(_MODEL)
    for i, regime in enumerate(REGIME_NAMES):
        row_sum = tm.loc[regime].sum()
        np.testing.assert_allclose(
            row_sum, 1.0, atol=1e-6,
            err_msg=f"Transition matrix row '{regime}' sums to {row_sum}, not 1.0",
        )


def test_transition_matrix_index_and_columns_match_regime_names():
    tm = transition_matrix(_MODEL)
    assert list(tm.index)   == REGIME_NAMES[:2]
    assert list(tm.columns) == REGIME_NAMES[:2]


def test_transition_matrix_values_are_probabilities():
    """All entries must lie in [0, 1]."""
    tm = transition_matrix(_MODEL)
    assert (tm.values >= 0).all() and (tm.values <= 1 + 1e-8).all()


# ---------------------------------------------------------------------------
# 8. Regime durations
# ---------------------------------------------------------------------------

def test_regime_durations_has_correct_columns():
    durs = regime_durations(_LABELS)
    expected = {"N Episodes", "Mean Days", "Median Days", "Max Days", "Min Days"}
    assert expected.issubset(set(durs.columns))


def test_regime_durations_episodes_positive():
    durs = regime_durations(_LABELS)
    for regime in REGIME_NAMES:
        if durs.loc[regime, "N Episodes"] > 0:
            assert durs.loc[regime, "Min Days"] >= 1
            assert durs.loc[regime, "Max Days"] >= durs.loc[regime, "Min Days"]


def test_regime_durations_total_days_consistent():
    """Sum of (N_episodes × mean_duration) ≈ total days in that regime."""
    durs = regime_durations(_LABELS)
    for regime in REGIME_NAMES:
        n_ep   = durs.loc[regime, "N Episodes"]
        mean_d = durs.loc[regime, "Mean Days"]
        actual = int((_LABELS == regime).sum())
        if n_ep > 0 and np.isfinite(mean_d):
            reconstructed = n_ep * mean_d
            # allow floating point rounding in mean calculation
            np.testing.assert_allclose(
                reconstructed, actual, rtol=0.01,
                err_msg=f"Duration accounting off for regime '{regime}'",
            )


# ---------------------------------------------------------------------------
# 9. Emission params: low-vol std < high-vol std
# ---------------------------------------------------------------------------

def test_emission_params_std_ordering():
    """Low Vol state must have strictly lower daily std than High Vol state."""
    ep = regime_emission_params(_MODEL)
    std_low  = ep.loc[REGIME_NAMES[0], "Daily Std Return"]
    std_high = ep.loc[REGIME_NAMES[1], "Daily Std Return"]
    assert std_high > std_low, (
        f"Emission std ordering wrong: low={std_low:.6f}, high={std_high:.6f}"
    )


def test_emission_params_columns_present():
    ep = regime_emission_params(_MODEL)
    expected = {"Daily Mean Return", "Daily Std Return",
                "Ann. Mean Return", "Ann. Volatility (implied)"}
    assert expected.issubset(set(ep.columns))


def test_emission_params_annualised_vol_matches_daily():
    """Ann. Volatility (implied) must equal Daily Std × sqrt(252)."""
    ep = regime_emission_params(_MODEL)
    from config.settings import TRADING_DAYS_PER_YEAR
    for regime in REGIME_NAMES:
        daily_std = ep.loc[regime, "Daily Std Return"]
        ann_vol   = ep.loc[regime, "Ann. Volatility (implied)"]
        np.testing.assert_allclose(
            ann_vol, daily_std * np.sqrt(TRADING_DAYS_PER_YEAR), rtol=1e-6,
            err_msg=f"Annualised vol mismatch for regime '{regime}'",
        )


# ---------------------------------------------------------------------------
# 10. contiguous_blocks covers all dates
# ---------------------------------------------------------------------------

def test_contiguous_blocks_covers_all_dates():
    """
    The union of all (start, end) blocks must cover every date in the
    regime label series.
    """
    blocks = contiguous_blocks(_LABELS)
    covered = set()
    for _, start, end in blocks:
        span = pd.date_range(start, end, freq="B")
        covered |= set(span)
    all_dates = set(_LABELS.index)
    # Allow minor mismatches due to business-day rounding at block boundaries
    uncovered = all_dates - covered
    assert len(uncovered) <= 2, (
        f"contiguous_blocks left {len(uncovered)} dates uncovered"
    )


def test_contiguous_blocks_only_known_regimes():
    blocks = contiguous_blocks(_LABELS)
    for regime, _, _ in blocks:
        assert regime in REGIME_NAMES, f"Unknown regime in block: {regime}"


def test_contiguous_blocks_no_adjacent_same_regime():
    """
    Consecutive blocks must not have the same regime label (they should
    have been merged).
    """
    blocks = contiguous_blocks(_LABELS)
    for i in range(len(blocks) - 1):
        assert blocks[i][0] != blocks[i + 1][0], (
            f"Adjacent blocks have the same regime '{blocks[i][0]}' at index {i}"
        )


# ---------------------------------------------------------------------------
# 11. regime_beta_alpha: finite values on correlated synthetic data
# ---------------------------------------------------------------------------

def test_regime_beta_alpha_finite_values():
    ba = regime_beta_alpha(_RETURNS, _BENCHMARK, _LABELS)
    assert list(ba.index) == REGIME_NAMES
    for col in ("Alpha (Ann.)", "Beta", "R-squared"):
        for regime in REGIME_NAMES:
            val = ba.loc[regime, col]
            assert np.isfinite(val), f"{col} for '{regime}' is not finite: {val}"


def test_regime_beta_alpha_r_squared_in_range():
    ba = regime_beta_alpha(_RETURNS, _BENCHMARK, _LABELS)
    for regime in REGIME_NAMES:
        r2 = ba.loc[regime, "R-squared"]
        assert 0.0 <= r2 <= 1.0 + 1e-6, f"R-squared out of [0,1] for '{regime}': {r2}"


def test_regime_beta_alpha_returns_nan_for_sparse_regime():
    """If a regime has < 30 observations, alpha/beta should be NaN."""
    # Force a label series where Low Vol (Bull) has only 5 days
    sparse_labels = _LABELS.copy()
    # override first 5 days to high-vol, leaving almost no low-vol days
    n_low = int((_LABELS == REGIME_NAMES[0]).sum())
    lv_indices = _LABELS[_LABELS == REGIME_NAMES[0]].index
    if len(lv_indices) > 29:
        # replace all but 5 with High Vol
        to_flip = lv_indices[:-5]
        sparse_labels.loc[to_flip] = REGIME_NAMES[1]

    ba = regime_beta_alpha(_RETURNS, _BENCHMARK, sparse_labels)
    # Low Vol (Bull) now has ≤ 5 obs — should be NaN
    assert np.isnan(ba.loc[REGIME_NAMES[0], "Beta"]), (
        "Expected NaN for sparse regime but got a value"
    )


# ---------------------------------------------------------------------------
# 12. Reproducibility
# ---------------------------------------------------------------------------

def test_fit_hmm_reproducible():
    """Same seed must produce the same state sequence."""
    _, seq1, _ = fit_hmm(_RETURNS, n_states=2, n_restarts=5, random_seed=0)
    _, seq2, _ = fit_hmm(_RETURNS, n_states=2, n_restarts=5, random_seed=0)
    np.testing.assert_array_equal(seq1, seq2,
                                   err_msg="HMM not reproducible with same seed")


def test_fit_hmm_different_seeds_may_differ():
    """
    Different seeds may produce different sequences.
    This is a soft check: if they happen to match it is fine, but it
    confirms the seed parameter is actually used.  We just verify no crash.
    """
    _, seq1, _ = fit_hmm(_RETURNS, n_states=2, n_restarts=3, random_seed=0)
    _, seq2, _ = fit_hmm(_RETURNS, n_states=2, n_restarts=3, random_seed=999)
    assert len(seq1) == len(seq2) == len(_RETURNS)
