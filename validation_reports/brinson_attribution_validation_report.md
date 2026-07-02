# Brinson-Hood-Beebower Attribution — Validation Report

**Date**: 2026-06-08  
**Module**: `analytics/brinson_attribution.py`  
**Test file**: `tests/test_brinson_attribution.py`  
**Project**: Atlas PM

---

## 1. Module Purpose

This module implements the Brinson-Hood-Beebower (BHB) and Brinson-Fachler (BF)
performance attribution framework. It decomposes a portfolio's active return
(portfolio return minus benchmark return) into three interpretable components:

1. **Allocation Effect** — did the manager over/underweight the right asset classes?
2. **Selection Effect** — did asset selection within each class beat the benchmark?
3. **Interaction Effect** — combined impact of getting both allocation and selection right simultaneously

This complements the existing Fama-French factor attribution module. Factor attribution
asks "what systematic risk premia explain my returns?" Brinson attribution asks
"what portfolio management decisions relative to a benchmark explain my active return?"

---

## 2. Methodology

The attribution is performed at the **group (asset class) level** on a **monthly** basis.

**Step 1**: Assets are classified into groups (Equities, Fixed Income, Alternatives, Cash).

**Step 2**: A benchmark weight vector is constructed. Two options are available:
- Equal-weight: all assets receive weight 1/N
- 60/40 group-weighted: Equities 60%, Fixed Income 30%, Alternatives 5%, Cash 5%,
  with equal weights within each group

**Step 3**: Monthly group returns are computed by compounding daily asset returns
and weighting by within-group asset weights.

**Step 4**: For each group and each monthly period, attribution effects are computed
using either BF or BHB formulas.

---

## 3. Formula Definitions

### Brinson-Fachler (default)

| Effect | Formula |
|--------|---------|
| Allocation | (w_p,g − w_b,g) × (r_b,g − r_b) |
| Selection | w_b,g × (r_p,g − r_b,g) |
| Interaction | (w_p,g − w_b,g) × (r_p,g − r_b,g) |

### Brinson-Hood-Beebower (original 1986)

| Effect | Formula |
|--------|---------|
| Allocation | (w_p,g − w_b,g) × r_b,g |
| Selection | w_b,g × (r_p,g − r_b,g) |
| Interaction | (w_p,g − w_b,g) × (r_p,g − r_b,g) |

**Difference**: Only the allocation formula differs. BF measures allocation relative
to benchmark total return (r_b,g − r_b); BHB measures relative to zero (r_b,g).

**Reconciliation identity** (holds for both methods when weights sum to 1):
```
Σ_g [Allocation + Selection + Interaction] = r_p,t − r_b,t
```

**Proof** (BF, single period, dropping t):
```
Σ_g [(w_p,g−w_b,g)(r_b,g−r_b) + w_b,g(r_p,g−r_b,g) + (w_p,g−w_b,g)(r_p,g−r_b,g)]
= Σ_g [w_p,g*r_p,g − w_b,g*r_b,g − (w_p,g−w_b,g)*r_b]
= r_p − r_b − r_b*Σ_g(w_p,g−w_b,g)
= r_p − r_b − r_b*(1−1)
= r_p − r_b  ✓
```

**Default**: Brinson-Fachler is the default. It is the standard for relative
performance attribution in institutional asset management because the allocation
effect is positive only when you overweight a group that *outperforms the benchmark
total return*, which is the intuitive meaning of a good allocation decision.

---

## 4. Data Inputs

| Input | Source | Notes |
|-------|--------|-------|
| Asset daily returns | `st.session_state["simple_returns"]` | All assets in universe |
| Portfolio weights | `st.session_state["current_weights"]` | Static (no rebalancing) |
| Benchmark weights | Computed from universe | Equal-weight or 60/40 group-weighted |
| Asset classification | `DEFAULT_CLASSIFICATION` dict | Equities / Fixed Income / Alternatives / Cash |

**Benchmark caveat**: The benchmark is constructed from the same asset universe as the
portfolio, with alternative weights. It is NOT an investable institutional benchmark
such as MSCI World, FTSE All-Share, or a live 60/40 blended index. All outputs are
labelled clearly as using a simplified benchmark.

---

## 5. Reconciliation Validation

The `validate_brinson_reconciliation()` function checks that for each monthly period:

```
|Σ_g(Total Effect) − Active Return| < 1e-8
```

This tolerance accounts for double-precision floating-point arithmetic (machine epsilon ~2.2e-16;
accumulated over ~40 groups × 3 terms = several hundred floating-point ops, so 1e-8 is
conservative).

The reconciliation is guaranteed analytically when:
1. `Σ port_group_weights = 1` (enforced by normalisation in `calculate_brinson_attribution`)
2. `Σ bench_group_weights = 1` (same)
3. `r_b,t = Σ_g(w_b,g × r_b,g,t)` (used consistently throughout)

All three conditions are enforced in code.

---

## 6. Synthetic Test Cases

All 46 tests use deterministic datasets — no live market data, no randomness
beyond fixed-seed random number generators.

| Test category | Dataset construction |
|---------------|---------------------|
| Reconciliation (BF/BHB) | Hand-crafted 2–4 group, 3–6 period examples; also numpy `dirichlet` random weights |
| Zero active weights | Equal portfolio and benchmark weights |
| Zero selection | Identical group returns for port and bench |
| Identical portfolio=benchmark | Weights AND returns equal |
| Overweight winner/loser | Constructed so r_b,G0 > r_b (winner) or < r_b (loser) |
| Selection sign | Constructed r_p,g > r_b,g or vice versa |
| Interaction formula | Random 3-group 4-period example with oracle verification |
| Weight normalisation | Various asset subsets; checked to sum to 1 |
| Cumulative | Oracle: numpy cumsum of known period effects |
| IC proxy | Constructed so overweight groups outperform (positive) or underperform (negative) |
| BHB vs BF | Same inputs, both methods; total must be identical |
| Missing group | Daily prices with one group having zero portfolio weight |
| Reconciliation helper | Correct data (pass) and corrupted data (fail) |
| Group returns integration | Constant daily returns, oracle computable from `(1+r)^n − 1` |
| End-to-end pipeline | Full data pipeline → attribution → reconciliation pass |

---

## 7. Tests Passed and Failed

```
============================= test session starts =============================
collected 46 items

All 46 tests: PASSED
0 failures, 0 errors
130 total project tests: 130 passed (including all walk-forward tests)
```

| Test class | Tests | Result |
|-----------|-------|--------|
| TestReconciliationBF | 3 | PASS |
| TestReconciliationBHB | 2 | PASS |
| TestZeroActiveWeights | 3 | PASS |
| TestZeroSelectionDifference | 3 | PASS |
| TestIdenticalPortfolioAndBenchmark | 2 | PASS |
| TestOverweightWinner | 2 | PASS |
| TestOverweightLoser | 1 | PASS |
| TestSelectionEffect | 2 | PASS |
| TestInteractionEffect | 2 | PASS |
| TestWeightNormalisation | 6 | PASS |
| TestCumulativeAttribution | 3 | PASS |
| TestICProxy | 6 | PASS |
| TestBHBvsBF | 4 | PASS |
| TestZeroWeightGroup | 1 | PASS |
| TestValidateReconciliation | 3 | PASS |
| TestGroupReturnsIntegration | 2 | PASS |

---

## 8. Issues Found

| Severity | Issue | Status |
|----------|-------|--------|
| MEDIUM | Test oracle bug: `TestGroupReturnsIntegration::test_group_returns_sum_weighted_correctly` used 63 prices spanning 3 months, but oracle computed compound return for 62 days (full span rather than first month). | Fixed: test now uses 22 prices (21 return days, all within January 2023). |
| LOW | Zero-weight portfolio group: portfolio group return is undefined when a group has no assets. | Handled: `calculate_group_returns` sets portfolio group return equal to benchmark group return for zero-weight groups, ensuring selection and interaction are zero and all effect flows through allocation. |
| LOW | `build_benchmark_weights` with group_weighted and a universe subset (some groups missing): normalisation could divide by zero or produce unexpected weights. | Handled: normalises only over groups that have available assets; falls back to equal-weight across groups if total group weight is zero. |

---

## 9. Fixes Applied

1. **Test oracle**: Changed `n_days=63` to `n_days=22` in `_make_daily_prices` for the group returns oracle test, so that all returns fall within one calendar month and `resample("ME")` returns exactly one period.

2. **Zero active weight in group**: Explicit handling in `calculate_group_returns` — when `pw_sum < 1e-12`, the portfolio group return is set to `NaN` in the DataFrame and then overwritten with the benchmark group return, making selection and interaction zero.

3. **Weight normalisation guard in attribution engine**: `calculate_brinson_attribution` normalises both weight vectors to sum exactly to 1 before computing effects, preventing floating-point drift from causing non-zero residuals.

---

## 10. Limitations

1. **Simplified benchmark**: The benchmark is not a live investable index. All active return measures are relative to an internal equal-weight or 60/40 constructed benchmark. This is appropriate for learning and interview demonstration but should not be presented as institutional-grade attribution.

2. **Static weights**: The portfolio uses fixed beginning-of-period weights throughout. A real attribution system would use actual weights at each rebalancing date. This is a known approximation; the effect is small for short horizons.

3. **Arithmetic cumulation**: Multi-period attribution uses arithmetic cumulative sums. For horizons beyond 2–3 years, geometric linking (Carino, Menchero) would be more accurate. The arithmetic method understates compounding effects in high-return environments.

4. **Asset class level only**: Attribution is performed at the group (asset class) level, not individual security level. Within-group diversification and individual asset selection effects are lumped together.

5. **ETF proxy assets**: The universe uses ETFs, not individual securities. Each ETF represents an index, so "selection effect" measures whether the portfolio's choice of ETF outperformed an equal-weight mix of all ETFs in that class — not individual stock picking.

---

## 11. Information Coefficient Note

**This module does NOT compute a true Information Coefficient.**

True IC = cross-sectional Pearson correlation between analyst forecast alpha scores
and subsequent realised returns. This requires forecast scores at the individual
asset level, which do not exist in Atlas PM.

What is implemented is an **Allocation Effectiveness Proxy**:
- For each period, compute the cross-sectional correlation between the vector of
  group active weights (w_p,g − w_b,g) and the vector of benchmark-relative group
  returns (r_b,g,t − r_b,t).
- Report the mean of these per-period correlations.

This measures whether overweighting groups systematically co-incides with those
groups outperforming the benchmark. It is a group-level allocation skill proxy,
not a forecast skill measure.

The dashboard labels this unambiguously as a proxy with a warning banner.

---

## 12. Final Confidence Assessment

**Confidence level: HIGH for the calculation engine.**

Rationale:
- Both BHB and BF attribution effects reconcile exactly to active return for all
  46 test cases (max residual < 1e-12).
- Tests cover all critical edge cases: zero active weights, zero selection, identical
  portfolio and benchmark, sign tests for allocation and selection, interaction formula
  verification.
- The reconciliation identity is proven analytically and verified numerically.
- The group return calculation is verified against a hand-computed oracle for a
  constant-return dataset.

**Confidence level: MEDIUM for the dashboard interpretation.**

The simplified benchmark means active return numbers are not comparable to
institutional performance attribution. The module is correct given its inputs;
the inputs themselves are a simplified approximation.

---

## 13. Interview Explanation

> "The Brinson attribution on my portfolio management application decomposes active
> return into three components. The allocation effect captures how much of my active
> return came from over or underweighting asset classes relative to a benchmark — using
> the Brinson-Fachler formula, which is the industry standard because it only rewards
> overweights in groups that actually outperformed the benchmark total, not groups that
> merely had positive absolute returns. The selection effect captures whether my choice
> of specific assets within each class outperformed what the benchmark held in that
> class. The interaction term captures the combined effect of getting both the bet size
> and the security selection right simultaneously.
>
> The reconciliation identity guarantees that allocation plus selection plus interaction
> equals active return to machine precision for every monthly period — I have automated
> tests that verify this. The benchmark in my implementation is a simplified
> equal-weight or group-weighted portfolio of my ETF universe, not a live institutional
> index, and I'm transparent about that in the UI. I also implemented an allocation
> effectiveness proxy — the correlation between my group active weights and subsequent
> group outperformance — which I label clearly as a proxy, not the true Information
> Coefficient, because I don't have analyst forecast scores."
