"""
Regime Detection module for Atlas PM.

Implements a two-state Gaussian Hidden Markov Model (HMM) to identify
market regimes from return data.

The HMM assumes returns are drawn from one of N Gaussian distributions
(states).  The Viterbi algorithm decodes the most likely state sequence.
States are labelled deterministically by their emission variance: the
lower-variance state is always "Low Vol (Bull)"; the higher-variance
state is always "High Vol (Bear)".

Fixing the labelling this way resolves the label-switching problem: HMM
states are otherwise arbitrary integers, so running the model twice on
the same data can swap the labels.

Reference:
  Hamilton, J.D. (1989) "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle", Econometrica 57(2).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False

from analytics.returns import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
)
from config.settings import TRADING_DAYS_PER_YEAR

# Regime names — always ordered low-vol first, high-vol second.
REGIME_NAMES: list[str] = ["Low Vol (Bull)", "High Vol (Bear)"]
REGIME_COLORS: dict[str, str] = {
    "Low Vol (Bull)":  "rgba(46, 134, 171, 0.15)",
    "High Vol (Bear)": "rgba(232, 72, 85, 0.15)",
}
REGIME_LINE_COLORS: dict[str, str] = {
    "Low Vol (Bull)":  "#2e86ab",
    "High Vol (Bear)": "#e84855",
}

MIN_REQUIRED_OBSERVATIONS: int = 126  # ~6 months minimum


def _check_hmmlearn() -> None:
    if not _HMMLEARN_AVAILABLE:
        raise ImportError(
            "hmmlearn is required for regime detection. "
            "Install it with: pip install hmmlearn>=0.3"
        )


def _get_state_variances(model: "_GaussianHMM") -> np.ndarray:
    """Extract scalar emission variance for each HMM state."""
    if model.covariance_type == "full":
        return np.array([float(model.covars_[i][0, 0])
                         for i in range(model.n_components)])
    elif model.covariance_type == "diag":
        return model.covars_[:, 0].astype(float)
    elif model.covariance_type == "spherical":
        return np.asarray(model.covars_, dtype=float)
    else:  # tied
        return np.full(model.n_components, float(model.covars_[0, 0]))


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_hmm(
    returns: pd.Series,
    n_states: int = 2,
    n_iter: int = 200,
    n_restarts: int = 15,
    random_seed: int = 42,
) -> tuple["_GaussianHMM", np.ndarray, float]:
    """
    Fit a Gaussian HMM to the return series.

    Multiple random restarts are used and the model with the highest
    log-likelihood is returned, reducing sensitivity to EM initialisation.

    Parameters
    ----------
    returns      : daily simple returns with a DatetimeIndex
    n_states     : number of hidden states (default 2)
    n_iter       : EM iterations per restart
    n_restarts   : independent restarts (guards against local optima)
    random_seed  : base seed for reproducibility

    Returns
    -------
    (model, state_sequence, log_likelihood)
    model          : best-fit GaussianHMM
    state_sequence : np.ndarray of raw state indices (not yet regime-labelled)
    log_likelihood : total log-likelihood on the training data
    """
    _check_hmmlearn()

    if len(returns) < MIN_REQUIRED_OBSERVATIONS:
        raise ValueError(
            f"At least {MIN_REQUIRED_OBSERVATIONS} observations are required to fit "
            f"the HMM; got {len(returns)}. Load a longer return history."
        )
    if returns.isna().any():
        raise ValueError(
            "Return series contains NaN values. "
            "Drop or forward-fill missing dates before fitting."
        )

    X = returns.values.reshape(-1, 1)

    best_model: Optional["_GaussianHMM"] = None
    best_score = -np.inf

    for i in range(n_restarts):
        model = _GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_seed + i,
            tol=1e-5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X)
                score = model.score(X)
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError(
            "HMM failed to converge across all random restarts. "
            "Try a longer return series or reduce n_states."
        )

    state_sequence = best_model.predict(X)
    return best_model, state_sequence, float(best_score)


# ---------------------------------------------------------------------------
# Regime labelling (fixes label-switching)
# ---------------------------------------------------------------------------

def label_regimes(
    model: "_GaussianHMM",
    state_sequence: np.ndarray,
    returns: pd.Series,
) -> pd.Series:
    """
    Map raw HMM state indices to named regimes, sorted by emission variance.

    The state with the LOWEST emission variance → REGIME_NAMES[0] ("Low Vol (Bull)").
    The state with the HIGHEST emission variance → REGIME_NAMES[1] ("High Vol (Bear)").

    This is deterministic regardless of how the HMM labelled the states
    internally, completely resolving the label-switching problem.

    Parameters
    ----------
    model          : fitted GaussianHMM
    state_sequence : raw state-index array from model.predict()
    returns        : original return series (provides the DatetimeIndex)

    Returns
    -------
    pd.Series of regime names with a DatetimeIndex, dtype category.
    """
    variances = _get_state_variances(model)
    sorted_states = np.argsort(variances)  # ascending: [low-vol-idx, high-vol-idx]
    raw_to_regime = {
        int(sorted_states[i]): REGIME_NAMES[i]
        for i in range(model.n_components)
    }
    labels = pd.Series(
        [raw_to_regime[int(s)] for s in state_sequence],
        index=returns.index,
        name="regime",
        dtype="category",
    )
    return labels


# ---------------------------------------------------------------------------
# Conditional statistics
# ---------------------------------------------------------------------------

def regime_statistics(
    portfolio_returns: pd.Series,
    regime_labels: pd.Series,
    rf_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute annualised performance statistics conditional on each regime.

    Parameters
    ----------
    portfolio_returns : daily simple returns
    regime_labels     : output of label_regimes()
    rf_returns        : optional daily risk-free returns (same index)

    Returns
    -------
    DataFrame indexed by regime name with columns:
      N Days, Pct Sample, Ann. Return, Ann. Volatility,
      Sharpe Ratio, Sortino Ratio, Max Drawdown, Daily Win Rate.
    """
    total_days = len(portfolio_returns)
    rows = []

    for regime in REGIME_NAMES:
        mask = regime_labels == regime
        ret_r = portfolio_returns[mask]

        if len(ret_r) == 0:
            rows.append({
                "Regime": regime,
                "N Days": 0,
                "Pct Sample": 0.0,
                "Ann. Return": float("nan"),
                "Ann. Volatility": float("nan"),
                "Sharpe Ratio": float("nan"),
                "Sortino Ratio": float("nan"),
                "Max Drawdown": float("nan"),
                "Daily Win Rate": float("nan"),
            })
            continue

        rf_r = (rf_returns.reindex(ret_r.index).fillna(0)
                if rf_returns is not None else None)

        rows.append({
            "Regime": regime,
            "N Days": int(mask.sum()),
            "Pct Sample": float(mask.sum() / total_days),
            "Ann. Return": annualised_return(ret_r),
            "Ann. Volatility": annualised_volatility(ret_r),
            "Sharpe Ratio": sharpe_ratio(ret_r, rf_r),
            "Sortino Ratio": sortino_ratio(ret_r, rf_r),
            "Max Drawdown": max_drawdown(ret_r),
            "Daily Win Rate": float((ret_r > 0).mean()),
        })

    return pd.DataFrame(rows).set_index("Regime")


def regime_beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    OLS regression of portfolio returns on benchmark per regime.

    Returns DataFrame indexed by regime with columns:
      Alpha (Ann.), Beta, R-squared, p-value (beta).
    """
    from scipy.stats import linregress

    rows = []
    for regime in REGIME_NAMES:
        mask = regime_labels == regime
        port_r  = portfolio_returns[mask]
        bench_r = benchmark_returns.reindex(port_r.index).fillna(0)

        if len(port_r) < 30:
            rows.append({
                "Regime": regime,
                "Alpha (Ann.)": float("nan"),
                "Beta": float("nan"),
                "R-squared": float("nan"),
                "p-value (beta)": float("nan"),
            })
            continue

        slope, intercept, r_value, p_value, _ = linregress(bench_r.values, port_r.values)
        rows.append({
            "Regime": regime,
            "Alpha (Ann.)": float(intercept * TRADING_DAYS_PER_YEAR),
            "Beta": float(slope),
            "R-squared": float(r_value ** 2),
            "p-value (beta)": float(p_value),
        })

    return pd.DataFrame(rows).set_index("Regime")


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def transition_matrix(model: "_GaussianHMM") -> pd.DataFrame:
    """
    Return the HMM transition probability matrix as a DataFrame.

    Rows = "from" regime, columns = "to" regime, ordered by volatility so
    the labelling is consistent with label_regimes().
    """
    n = model.n_components
    variances = _get_state_variances(model)
    sorted_states = np.argsort(variances)
    idx_order = [int(sorted_states[i]) for i in range(n)]
    reordered = model.transmat_[np.ix_(idx_order, idx_order)]
    return pd.DataFrame(
        reordered,
        index=REGIME_NAMES[:n],
        columns=REGIME_NAMES[:n],
    )


# ---------------------------------------------------------------------------
# Regime duration statistics
# ---------------------------------------------------------------------------

def regime_durations(regime_labels: pd.Series) -> pd.DataFrame:
    """
    Duration statistics for consecutive regime runs (episodes).

    Returns DataFrame indexed by regime with columns:
      N Episodes, Mean Days, Median Days, Max Days, Min Days.
    """
    rows = []
    for regime in REGIME_NAMES:
        durations: list[int] = []
        count = 0
        for label in regime_labels:
            if label == regime:
                count += 1
            else:
                if count > 0:
                    durations.append(count)
                    count = 0
        if count > 0:
            durations.append(count)

        if durations:
            rows.append({
                "Regime": regime,
                "N Episodes": len(durations),
                "Mean Days": float(np.mean(durations)),
                "Median Days": float(np.median(durations)),
                "Max Days": int(np.max(durations)),
                "Min Days": int(np.min(durations)),
            })
        else:
            rows.append({
                "Regime": regime,
                "N Episodes": 0,
                "Mean Days": float("nan"),
                "Median Days": float("nan"),
                "Max Days": 0,
                "Min Days": 0,
            })

    return pd.DataFrame(rows).set_index("Regime")


# ---------------------------------------------------------------------------
# Emission parameter summary
# ---------------------------------------------------------------------------

def regime_emission_params(model: "_GaussianHMM") -> pd.DataFrame:
    """
    Summary of the fitted emission distributions per regime.

    Useful for validating that the two regimes are genuinely different.
    Returns annualised mean return and volatility implied by each state.
    """
    variances = _get_state_variances(model)
    sorted_states = np.argsort(variances)

    rows = []
    for i, raw_idx in enumerate(sorted_states):
        mean_daily = float(model.means_[raw_idx][0])
        std_daily  = float(np.sqrt(variances[raw_idx]))
        rows.append({
            "Regime": REGIME_NAMES[i],
            "Daily Mean Return": mean_daily,
            "Daily Std Return": std_daily,
            "Ann. Mean Return": mean_daily * TRADING_DAYS_PER_YEAR,
            "Ann. Volatility (implied)": std_daily * np.sqrt(TRADING_DAYS_PER_YEAR),
        })

    return pd.DataFrame(rows).set_index("Regime")


# ---------------------------------------------------------------------------
# Helper: contiguous regime blocks (for chart shading)
# ---------------------------------------------------------------------------

def contiguous_blocks(
    regime_labels: pd.Series,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Return a list of (regime_name, start_date, end_date) for each
    contiguous run of the same regime.  Used to add vrect shapes to charts.
    """
    blocks: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    current: str | None = None
    start: pd.Timestamp | None = None
    prev: pd.Timestamp | None = None

    for date, label in regime_labels.items():
        if label != current:
            if current is not None and start is not None and prev is not None:
                blocks.append((current, start, prev))
            current = label
            start = date
        prev = date

    if current is not None and start is not None and prev is not None:
        blocks.append((current, start, prev))

    return blocks
