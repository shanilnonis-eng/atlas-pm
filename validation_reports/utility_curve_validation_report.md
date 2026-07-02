# Utility Curve Validation Report — Atlas PM

**Module:** `analytics/investor_utility.py`  
**Page integration:** `pages/2_Portfolio_Construction.py` — Efficient Frontier expander  
**Date:** 2026-06-09  
**Status:** ✅ All tests passing (27/27 new + 235/235 pre-existing)

---

## 1. Module Purpose

Implements mean-variance investor utility for the efficient frontier risk-preference overlay.
Exposes four functions and two constants used by the Portfolio Construction page.

| Symbol | Role |
|--------|------|
| `calculate_mean_variance_utility` | Evaluate U = E(r) − 0.5·A·σ² |
| `calculate_indifference_curve` | Evaluate E(r) = U + 0.5·A·σ² over a volatility grid |
| `find_utility_optimal_portfolio` | Select the frontier point with the highest U |
| `map_profile_to_risk_aversion` | Translate a named profile to a coefficient A |
| `PROFILE_NAMES` | Ordered list of profile labels (inc. "Custom") |
| `PROFILE_RISK_AVERSION` | Canonical profile → A mapping |

---

## 2. Formula

### Utility function

```
U = E(r) - 0.5 × A × σ²
```

| Variable | Definition |
|----------|------------|
| U | Mean-variance utility (higher is better) |
| E(r) | Annualised expected return (arithmetic, decimal fraction) |
| σ | Annualised portfolio volatility (decimal fraction) |
| A | Risk-aversion coefficient (positive scalar, typically 1–10) |

### Indifference curve (fixed U)

```
E(r) = U + 0.5 × A × σ²
```

- Upward sloping: as σ increases, the investor requires higher E(r) to maintain the same utility.
- Convex (quadratic in σ): the curve steepens at higher volatility.
- Higher A makes the curve steeper at every point — the investor demands more return per unit of extra risk.

---

## 3. Investor Profile → Risk-Aversion Mapping

| Profile | A |
|---------|---|
| Very Conservative | 8.0 |
| Conservative | 6.0 |
| Balanced | 4.0 (textbook reference) |
| Growth | 2.0 |
| Aggressive | 1.0 |
| Custom | Slider 0.5 – 10.0 |

---

## 4. Interpretation

**Utility-optimal portfolio:**  
The frontier point with the highest U for the selected A. Found by evaluating the utility formula
at every discrete frontier point (N = 40) and selecting argmax.

**Comparison with Max Sharpe:**  
- Max Sharpe maximises (E(r) − rf) / σ — a ratio, independent of risk tolerance.
- Utility-optimal maximises E(r) − 0.5·A·σ² — absolute, dependent on A.
- For a low-risk-aversion investor (A ≈ 1), the utility-optimal portfolio will be near Max Sharpe
  or on the high-return end of the frontier.
- For a high-risk-aversion investor (A ≈ 8), it will be near the minimum-variance point.

**Indifference curves on the chart:**  
- The optimal indifference curve (solid purple, U*) passes through the utility-optimal point and
  is tangent to the efficient frontier from below.
- Two lower curves (dotted, U* − 0.01 and U* − 0.025) are shown for visual context.
- All curves are clipped to the chart's visible return range ± 5 pp to avoid visual distortion.

---

## 5. Tests Performed

| # | Test | Result |
|---|------|--------|
| 1 | Utility increases when expected return increases (vol fixed) | ✅ Pass |
| 2 | Utility decreases when volatility increases (return fixed) | ✅ Pass |
| 3 | Higher risk aversion penalises volatility more | ✅ Pass |
| 4 | Indifference curve is strictly upward sloping | ✅ Pass |
| 5 | Higher risk aversion creates a steeper indifference curve | ✅ Pass |
| 6 | Utility-optimal portfolio is correctly selected (argmax) | ✅ Pass |
| 7 | Utility-optimal point lies on one of the frontier points | ✅ Pass |
| 8 | Custom risk aversion values (0.5, 1.0, 3.5, 7.0, 10.0) | ✅ Pass (parametrised × 5) |
| 9a | risk_aversion = 0 raises ValueError | ✅ Pass |
| 9b | risk_aversion < 0 raises ValueError | ✅ Pass |
| 9c | Indifference curve with negative A raises ValueError | ✅ Pass |
| 9d | find_utility_optimal with A = 0 raises ValueError | ✅ Pass |
| 9e | find_utility_optimal with A < 0 raises ValueError | ✅ Pass |
| 9f | Empty frontier arrays raise ValueError | ✅ Pass |
| 9g | Mismatched array shapes raise ValueError | ✅ Pass |
| 9h | Unknown profile string raises ValueError | ✅ Pass |
| 10 | Chart data: utility-optimal trace + utility curve trace present | ✅ Pass |
| 11a | Utility formula exact (rtol = 1e-12) | ✅ Pass |
| 11b | Indifference curve formula exact (rtol = 1e-12) | ✅ Pass |
| 11c | Vectorised utility matches loop of scalar calls | ✅ Pass |
| 12a | All 5 named profiles map to canonical A values | ✅ Pass |
| 12b | PROFILE_NAMES includes "Custom" | ✅ Pass |
| 12c | Very Conservative has lower utility than Aggressive at same (return, vol) | ✅ Pass |

**Total new tests:** 27 / 27 passed  
**Pre-existing tests:** 235 / 235 passed (no regressions)

---

## 6. Utility-Optimal Point — Frontier Membership

The utility-optimal portfolio is one of the 40 discrete points on the efficient frontier
(computed by `efficient_frontier()` using minimum-variance at each target return level).
The point is not interpolated — it is exactly one of the frontier solutions.

Test `test_utility_optimal_lies_on_frontier` confirms this:
```
result["expected_return"] == rets[result["index"]]  # exact equality
result["volatility"]      == vols[result["index"]]  # exact equality
```

---

## 7. Limitations

1. **Discrete approximation.** The utility-optimal point is the best of the 40 frontier points,
   not a continuous optimum. The true utility-maximum on the continuous frontier may differ by
   a small amount depending on frontier resolution.

2. **Arithmetic mean returns.** The frontier uses arithmetic mean returns scaled by 252 as
   expected returns. These overstate geometric (compound) returns, especially for high-volatility
   portfolios. The utility score should not be compared across different return periods.

3. **Historical inputs.** Expected returns are estimated from historical data. Future returns
   may differ materially. The utility-optimal portfolio is model-implied and sensitive to the
   return-estimation period.

4. **Single-period, quadratic utility.** The mean-variance utility function is a second-order
   approximation. It does not account for skewness, fat tails, or multi-period compounding.

5. **No liability matching.** The framework does not incorporate the investor's specific
   liabilities, constraints, time horizon, or tax situation.

6. **Illustrative only.** The output is labelled "model-implied preference" and includes
   explicit disclaimers. It does not constitute financial advice or a regulated recommendation.

---

## 8. Interview-Ready Explanation

**"Walk me through the utility curve overlay."**

> The mean-variance utility function is U = E(r) − 0.5 × A × σ², where A is the risk-aversion
> coefficient. A higher A means the investor penalises variance more heavily. The indifference
> curve E(r) = U + 0.5 × A × σ² shows all (volatility, return) pairs that give the same utility
> — it's upward sloping and convex because you need progressively more return to accept each
> additional unit of risk.
>
> On the efficient frontier, I evaluate utility at every frontier point and pick the argmax.
> That's the utility-optimal portfolio — the one the investor should theoretically prefer for
> their stated risk tolerance. It differs from the max Sharpe portfolio because Sharpe is a
> ratio (return per unit of risk) and is independent of how risk-averse the investor is, whereas
> the utility-optimal point directly incorporates A.
>
> For an aggressive investor (A = 1), the utility-optimal portfolio tends to be near max Sharpe
> or the high-return end; for a very conservative investor (A = 8), it shifts toward the
> minimum-variance point. I've implemented it so the chart updates immediately when the user
> changes their risk profile — the frontier computation is cached and only the overlay rerenders.
>
> The output is clearly labelled illustrative — it's a model-implied preference based on
> historical returns, not financial advice.

---

## 9. Files Created / Modified

| File | Action |
|------|--------|
| `analytics/investor_utility.py` | **Created** — 4 utility functions + profile constants |
| `tests/test_utility_curves.py` | **Created** — 27 tests |
| `pages/2_Portfolio_Construction.py` | **Modified** — added import + restructured Efficient Frontier expander |
| `validation_reports/utility_curve_validation_report.md` | **Created** — this file |
