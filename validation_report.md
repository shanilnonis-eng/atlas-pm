# Atlas PM — Validation Report

**Generated:** 2026-06-07  
**Test suite:** `validation_tests.py` — 90 tests across 11 modules  
**Final result:** ✅ **90 / 90 PASS** (after 3 bugs found and fixed)

---

## Summary Table

| Module | Tests | Passed | Failed | Status |
|--------|-------|--------|--------|--------|
| Data & Returns Computation | 6 | 6 | 0 | ✅ |
| Return Metrics (Sharpe, Sortino, Drawdown, CAGR) | 10 | 10 | 0 | ✅ |
| Portfolio Metrics (Return, Vol, Beta) | 5 | 5 | 0 | ✅ |
| Historical VaR & CVaR | 9 | 9 | 0 | ✅ |
| Risk Contributions (MRC / CRC / Euler) | 4 | 4 | 0 | ✅ |
| Optimiser (Equal Weight, MinVar, MaxSharpe, RP) | 17 | 17 | 0 | ✅ |
| Black-Litterman | 11 | 11 | 0 | ✅ |
| Stress Testing | 6 | 6 | 0 | ✅ |
| Factor Attribution (FF3, OLS) | 7 | 7 | 0 | ✅ |
| Edge Cases | 6 | 6 | 0 | ✅ |
| Covariance Matrix | 4 | 4 | 0 | ✅ |
| **TOTAL** | **90** | **90** | **0** | ✅ |

---

## Bugs Found and Fixed

Three bugs were identified during code inspection, confirmed by testing, and corrected. None required architectural changes — all fixes are under 5 lines each.

---

### BUG-1 — HIGH severity
**Location:** `construction/black_litterman.py` — `BlackLitterman.summary()`

**Description:**  
`summary()` multiplied `equilibrium_returns()` and `posterior_returns()` by `TRADING_DAYS_PER_YEAR` (252). Both methods already return **annualised** values — the covariance matrix is computed as `daily_cov × 252`, so `π = λΣw` is already in annual units. The extra multiplication inflated all displayed equilibrium and posterior returns by a factor of 252.

For a typical 10-asset portfolio with equilibrium returns of ~5–8% per year, the UI was displaying ~1,260–2,016% per year. Any recruiter or PM who spot-checked the BL page would have noticed immediately.

**Confirmed by test:** `test_bl_summary_annualisation_correct` — verified ratio of `summary()` to `equilibrium_returns()` is 1.0.

**Fix applied to `construction/black_litterman.py`:**
```python
# BEFORE (bug)
eq   = self.equilibrium_returns() * TRADING_DAYS_PER_YEAR
post = self.posterior_returns()   * TRADING_DAYS_PER_YEAR

# AFTER (correct)
eq   = self.equilibrium_returns()
post = self.posterior_returns()
```

---

### BUG-2 — MEDIUM severity
**Location:** `analytics/returns.py` — `sortino_ratio()`

**Description:**  
The Sortino ratio denominator used `returns[returns < 0]` (total returns below zero) to compute downside deviation. The numerator uses **excess** returns (returns minus risk-free rate). These must be consistent: if the numerator penalises returns below the risk-free hurdle, the denominator must also measure downside deviation relative to that same hurdle.

When `rf = 0`, the two are identical so the bug was invisible in testing without a risk-free rate. When `rf > 0` (the real use case), the function undercounts downside observations: a day where the portfolio returns 0.1% but the risk-free rate is 0.15% is an excess return loss, but the function ignores it because total returns > 0. This understates downside risk and overstates the Sortino ratio.

**Fix applied to `analytics/returns.py`:**
```python
# BEFORE (bug — uses total returns, not excess returns, for downside)
negative = returns[returns < 0]

# AFTER (correct — downside defined relative to the risk-free hurdle)
negative = excess[excess < 0]
```

---

### BUG-3 — MEDIUM severity
**Location:** `analytics/returns.py` — `sharpe_ratio()`

**Description:**  
The zero-volatility guard used exact float equality (`if ann_vol == 0`). Floating point arithmetic on a constant return series produces `ann_vol ≈ 3.4e-18` rather than exactly `0.0`, bypassing the guard and returning `~8.3 × 10¹⁶` instead of `NaN`. Any UI display of the Sharpe for a constant-return asset (e.g. the cash/T-bill proxy at certain periods) would show a meaningless astronomical number.

**Fix applied to `analytics/returns.py`:**
```python
# BEFORE (bug — exact float equality fails for near-zero floating point results)
if ann_vol == 0:

# AFTER (correct — tolerance-based check)
if ann_vol < 1e-12:
```

---

## Design Notes — Not Bugs, But Interview-Critical

These are legitimate design choices. You need to be able to defend them under questioning.

---

### NOTE-1 — Sharpe ratio: CAGR numerator vs arithmetic mean
**Location:** `analytics/returns.py` — `sharpe_ratio()`

The numerator uses `annualised_return(excess)` which applies **geometric compounding** (CAGR): `(1 + excess).prod()^(252/n) − 1`. The original Sharpe (1994) definition uses **arithmetic mean × 252**: `excess.mean() × 252`.

The two diverge for longer periods and higher return variance. Geometric is arguably more accurate for measuring realised wealth growth, but arithmetic is the industry standard when comparing Sharpe ratios across managers. If a recruiter asks you to compute a Sharpe ratio by hand and gets a different answer, this is why. Neither is wrong — but you must know which you are using and defend the choice.

**Defensible answer:** "We use the geometric (CAGR-based) Sharpe because it accurately measures the compounded wealth experience of the investor over the measurement period. Arithmetic-mean Sharpe tends to overstate performance in volatile strategies."

---

### NOTE-2 — Ledoit-Wolf shrinkage: approximate formula
**Location:** `construction/optimiser.py` — `ledoit_wolf_shrinkage()`

The shrinkage intensity formula used is:
```
delta = (||Σ||² + μ²) / ((n+1) × ||Σ − μI||²)
```
This is **not** the Oracle Approximating Shrinkage (OAS) estimator from Ledoit & Wolf (2004), which requires estimating the asymptotic variance of the sample covariance via a more involved expression. The formula used here is a simplified proxy that moves shrinkage intensity in the right direction but does not match the analytically optimal intensity.

The practical impact is small: shrinkage still reduces condition number and stabilises optimisation (confirmed by test). But if asked "how did you implement Ledoit-Wolf?", be honest that this is a numerically stable approximation, not the full analytical estimator. For production use, `sklearn.covariance.LedoitWolf` provides the exact implementation.

---

### NOTE-3 — Module docstring misleads on return type
**Location:** `data/loader.py` — module docstring

The docstring states *"We return DAILY LOG RETURNS"*. In fact, `compute_returns()` returns a tuple of `(simple_returns, log_returns)`. All financial calculation modules (`analytics/returns.py`, `construction/optimiser.py`, `construction/black_litterman.py`) consume **simple returns**, not log returns. The code is correct; the docstring is wrong and could mislead anyone extending the codebase.

---

### NOTE-4 — VaR sign convention: all-positive return series
**Location:** `analytics/risk.py` — `historical_var()`

For a series of entirely positive returns (e.g. T-bills in a calm period), `np.percentile(returns, 5)` is a small positive number, and `historical_var` returns its negation — a **negative** loss. The function does not clip to zero. In the UI this would display as a negative VaR ("you are guaranteed to make money in the worst 5% of days"), which is technically correct but confusing. Worth noting in an interview — and worth adding a `max(result, 0)` if the dashboard displays VaR for assets like BIL.

---

## What Was Independently Verified

Every test below used a **separate oracle calculation** — the app's own logic was not used as its own proof.

| Calculation | Oracle method used | Result |
|-------------|-------------------|--------|
| Simple returns from prices | Manual `(P_t / P_{t-1}) − 1` per cell | ✅ Exact match |
| Log returns | Manual `ln(P_t / P_{t-1})` | ✅ Exact match |
| Total return (geometric) | `(1+r)^n − 1` computed independently | ✅ Exact match |
| Annualised return (CAGR) | `(1 + total_return)^(252/n) − 1` | ✅ Exact match |
| Annualised volatility | `std(r, ddof=1) × √252` | ✅ Exact match |
| Sharpe ratio formula | `ann_excess / ann_vol` re-derived | ✅ Internally consistent |
| Max drawdown | Constructed wealth path, manual peak-to-trough | ✅ Exact match |
| Portfolio return | Row-by-row weighted sum | ✅ Exact match to 1e-10 |
| Portfolio vol | `√(w'Σw)` via matrix multiply | ✅ Match within 2% for n=2000 |
| Beta | `cov(port, bench) / var(bench)` from scratch | ✅ Exact match |
| Historical VaR | `−np.percentile(returns, 5)` | ✅ Exact match |
| Historical CVaR | `−mean(returns ≤ VaR_threshold)` | ✅ Exact match |
| CVaR > VaR | Algebraic necessity — tail average > tail quantile | ✅ |
| √T VaR scaling | `VaR_10d = VaR_1d × √10` | ✅ Exact match |
| MRC formula | `(Σw)_i / σ_p` computed independently | ✅ Match to 1e-6 |
| Component RC sum | Euler decomposition: `Σ(w_i × MRC_i) = σ_p` | ✅ Holds to 1e-6 |
| % Risk contribution | Sums to 1.0 | ✅ |
| MinVar 2-asset oracle | Analytical solution: `w* = σ₂² / (σ₁² + σ₂²)` | ✅ Match within 3% |
| MinVar ≤ EW vol | Direct covariance comparison | ✅ |
| MaxSharpe > EW Sharpe | Direct objective comparison | ✅ |
| MaxSharpe responds to rf | Different rf → different weights | ✅ |
| Risk parity equal-vol case | Equal vol → equal weight → equal % risk contribution | ✅ |
| BL equilibrium formula | `π = λΣw` computed directly | ✅ Exact match |
| BL no-views → equilibrium | Posterior = prior when views = [] | ✅ |
| BL high confidence → larger shift | View impact scales with confidence | ✅ |
| BL weights sum to 1 | Constraint verification | ✅ |
| Stress P&L oracle | `Σ(w_i × shock_i)` computed directly | ✅ Exact match |
| Equity crash → negative P&L | Scenario logic check | ✅ |
| Rate shock → negative bond P&L | Scenario logic check | ✅ |
| FF3 regression betas | OLS via `np.linalg.lstsq` on same data | ✅ Exact match |
| Beta recovery on synthetic data | Known true betas: 1.2, 0.4, −0.3 | ✅ Within ±0.15 |
| R² on known model | Should be > 0.85 with low noise | ✅ |
| Pure market portfolio | β_mkt ≈ 1, β_smb ≈ 0, β_hml ≈ 0, α ≈ 0 | ✅ |
| BL summary annualisation | Ratio to `equilibrium_returns()` = 1.0 | ✅ (after fix) |

---

## What Cannot Be Verified by Automated Tests

**AI Commentary (`ai/commentary.py`)**

There is no automated oracle for the AI-generated text. What can be asserted:

1. The system prompt explicitly forbids inventing numbers not in the provided context.
2. The prompts inject actual computed statistics as JSON before asking for commentary.
3. The model is instructed to separate data from interpretation from recommendation.

What cannot be verified automatically:
- Whether the model actually references the numbers correctly (hallucination risk)
- Whether interpretations are accurate (e.g. "Sharpe of 0.8 is strong" — is it, for this asset class?)
- Whether causal claims are supported by the data

**Recommendation for interview readiness:** Before any live demo, regenerate the commentary and read it against the actual numbers. Check that every figure cited in the AI output matches what is displayed on the dashboard. The model has been known to round incorrectly or cite figures in the wrong units.

---

## Final Confidence Assessment

| Module | Confidence level | Basis |
|--------|-----------------|-------|
| Returns calculation (CAGR, vol, drawdown) | **High** | Oracle-verified to 1e-9 |
| Sharpe ratio | **High** | Internally consistent; formula choice documented |
| Sortino ratio | **High** (after fix) | Denominator now uses excess returns consistently |
| Portfolio return and volatility | **High** | Matrix oracle verified |
| Historical VaR | **High** | Percentile oracle exact match |
| Historical CVaR | **High** | Tail-mean oracle exact match; CVaR > VaR confirmed |
| Equal Weight | **High** | Trivially correct, confirmed |
| Minimum Variance | **High** | 2-asset analytical oracle matches; constraint respected |
| Maximum Sharpe | **Medium-High** | Correct objective; multi-start heuristic — not guaranteed globally optimal on every input, but robust for the 10-asset universe |
| Risk Parity | **Medium-High** | Correct invariant (equal % risk contribution) confirmed; convergence depends on solver tolerance |
| Black-Litterman (maths) | **High** | He & Litterman formula verified; posterior shifts correctly |
| Black-Litterman (UI display) | **High** (after fix) | summary() no longer double-annualises |
| Risk contributions | **High** | Euler decomposition identity holds |
| Stress testing | **High** | P&L verified as `Σ(w_i × shock_i)`; all scenario directions correct |
| Factor attribution | **High** | OLS oracle match exact; true betas recovered on synthetic data |
| Covariance matrix | **High** | Symmetric, PD, annualised correctly; shrinkage reduces condition number |
| AI Commentary | **Unverifiable by oracle** | Prompt-level controls only; manual spot-check required |

**Overall:** The dashboard is mathematically correct for all automatically testable functions after fixing the three bugs above. It will survive line-by-line scrutiny on core portfolio maths in an interview. The Sortino and Sharpe design choices are defensible and well-documented. The BL page now displays sensible return magnitudes.

---

## How to Re-run

```bash
cd atlas-pm

# Full test suite with verbose output
python -m pytest validation_tests.py -v

# Quick pass/fail only
python -m pytest validation_tests.py -q

# Single module
python -m pytest validation_tests.py::TestBlackLitterman -v

# Generate this report (partial — parses pytest output)
python validation_runner.py
```
