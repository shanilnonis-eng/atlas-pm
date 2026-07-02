# GARCH Volatility Validation Report — Atlas PM

**Module:** `analytics/garch_volatility.py`  
**Page integration:** `pages/4_Risk_Management.py` — new "GARCH Volatility" tab  
**Library:** `arch` v8.0.0 (Kevin Sheppard)  
**Date:** 2026-06-09  
**Status:** ✅ All tests passing (26/26 new + 262/262 pre-existing)

---

## 1. Module Purpose

Adds conditional volatility estimation to the Risk Management page, extending
the existing static Historical VaR and rolling-vol estimates with two GARCH models.

| Aspect | Historical / Rolling | GARCH / GJR-GARCH |
|--------|---------------------|-------------------|
| Volatility estimate | Equal-weight trailing window | Conditional: weighted toward recent observations |
| Responds to shocks | Slowly (window dilutes) | Rapidly (exponential decay of shocks) |
| Leverage effect | Not captured | Captured by GJR-GARCH γ term |
| Multi-step forecast | Not available | GARCH recursion reverts to long-run vol |

---

## 2. Models Implemented

### GARCH(1,1) — Bollerslev (1986)

```
sigma²_t = omega + alpha * epsilon²_{t-1} + beta * sigma²_{t-1}
```

- **omega (ω):** long-run variance floor; must be positive
- **alpha (α):** reaction coefficient — how much a shock today affects tomorrow's variance
- **beta (β):** persistence — how slowly yesterday's variance decays
- **Persistence = α + β** — must be < 1 for covariance stationarity (mean-reversion in vol)

### GJR-GARCH(1,1,1) — Glosten, Jagannathan & Runkle (1993)

```
sigma²_t = omega + alpha * epsilon²_{t-1}
                 + gamma * epsilon²_{t-1} * I(epsilon_{t-1} < 0)
                 + beta * sigma²_{t-1}
```

- Adds **gamma (γ):** the asymmetry (leverage) parameter
- **I(ε < 0) = 1** when the previous return shock was negative
- **Interpretation:** if γ > 0 and statistically significant, bad news causes a disproportionately larger increase in volatility than good news of equal magnitude
- **Persistence = α + β + γ/2** (the γ/2 factor arises because E[I(ε<0)] = 0.5 under symmetric distributions)

### Innovation distribution

Both models use **Student-t innovations** (`dist='studentst'` in arch). The t-distribution has heavier tails than Gaussian, providing a better fit for daily equity returns which exhibit excess kurtosis (fat tails).

The degrees-of-freedom parameter **ν** is estimated by MLE:
- Low ν (e.g. 4–6): very fat tails — typical for volatile asset classes
- High ν (e.g. > 30): effectively Gaussian

---

## 3. GARCH-VaR Formula

```
VaR_1d = -(mu + z_alpha * sigma_T)
```

where:
- `sigma_T` = last conditional volatility (daily, fraction)
- `mu` = conditional mean (daily, fraction)
- `z_alpha` = quantile of the standardised Student-t at level (1 − confidence)
  - Standardised t has unit variance: `z = t.ppf(alpha, df=nu) × sqrt((nu−2)/nu)`
  - Falls back to Normal quantile when `nu` is not available

**Important note on t vs Normal VaR:**  
At 95% confidence, the standardised t quantile is *less* extreme than the Normal quantile
(e.g. for ν=6: ~−1.58 vs ~−1.645). The t-distribution's heavy-tail benefit manifests at
*very high* confidence (99%+): at 99%, standardised t(6) ≈ −2.78 vs Normal −2.326. This
is intentional and correct — the t-distribution's extra probability mass in the extreme tails
shifts the 5th percentile slightly rightward (less extreme) but the 1st percentile leftward
(more extreme).

---

## 4. Units and Annualisation

| Step | Detail |
|------|--------|
| Input | Daily simple returns as fractions (e.g. 0.01 = 1%) |
| arch internal | Multiplied by 100 → percent; `conditional_volatility` is in percent |
| Output conversion | Divided by 100 → fractions |
| Annualisation | × √252 where `annualise=True` (default) |

All public functions follow the same fraction convention as the rest of Atlas PM.

---

## 5. Tests Performed

| # | Test | Result |
|---|------|--------|
| 1 | `_validate_returns` raises for < 100 observations | ✅ Pass |
| 2 | `_validate_returns` raises for constant returns | ✅ Pass |
| 3 | `fit_garch` produces a valid result with required attributes | ✅ Pass |
| 4 | `fit_gjr_garch` produces a valid result | ✅ Pass |
| 5 | GJR-GARCH result contains gamma parameter | ✅ Pass |
| 6 | Conditional volatility is strictly positive everywhere | ✅ Pass |
| 7 | Conditional volatility has same length as input returns | ✅ Pass |
| 8 | Annualised cond. vol = daily cond. vol × √252 (rtol=1e-10) | ✅ Pass |
| 9 | GARCH persistence in (0, 1] | ✅ Pass |
| 10 | GJR-GARCH persistence in (0, 1] | ✅ Pass |
| 11 | GARCH-VaR and GJR-VaR are positive and finite | ✅ Pass |
| 12 | VaR(90%) < VaR(95%) < VaR(99%) — monotone in confidence | ✅ Pass |
| 13 | GJR-GARCH VaR is positive | ✅ Pass |
| 14 | Forecast length = requested horizon (1, 5, 21) | ✅ Pass × 3 |
| 15 | All forecast values positive and finite | ✅ Pass |
| 16 | `get_garch_params` returns DataFrame with required columns | ✅ Pass |
| 17 | `get_garch_params` contains omega, alpha, beta for both models | ✅ Pass |
| 18 | `has_leverage_effect` returns (bool, float); (False, nan) for plain GARCH | ✅ Pass |
| 19 | Leverage effect p-value in [0, 1] when not NaN | ✅ Pass |
| 20 | GARCH-VaR within 10× of Historical VaR (sanity check) | ✅ Pass |
| 21 | `garch_persistence` returns finite float | ✅ Pass |
| 22 | `fit_garch` raises ValueError for short series (end-to-end) | ✅ Pass |
| 23 | `fit_gjr_garch` raises ValueError for short series | ✅ Pass |
| 24 | 21-day forecast values are finite and < 100% daily | ✅ Pass |

**Total new tests: 26 / 26 passed**  
**Pre-existing tests: 262 / 262 passed — 0 regressions**  
**Total: 288 / 288**

---

## 6. Page Integration

A new **"GARCH Volatility"** tab is added to `pages/4_Risk_Management.py`.

**Rendering order inside the tab:**

1. Model description table (GARCH vs GJR-GARCH)
2. "Fit GARCH Models" button — triggers MLE fitting (~2–5s), results cached in `st.session_state["_garch_cache"]`
3. Key metrics row: current annualised conditional vol × 2 + persistence × 2
4. Leverage effect callout (significant or not, with p-value)
5. Conditional volatility chart: rolling historical vol (grey, dotted) + GARCH (blue) + GJR (purple)
6. VaR comparison table: Historical | GARCH | GJR-GARCH — updates when sidebar confidence changes
7. 21-day forecast chart (reverts to long-run vol; slope reflects persistence)
8. Model parameters (collapsible): parameter estimates + std errors + t-stats + p-values for both models
9. Interpretation expander: full narrative explanation of each result element
10. Model limitations info box

**Session state key:** `_garch_cache`  
**Confidence level:** read from the existing sidebar slider — VaR comparison updates automatically when changed.

---

## 7. Limitations

1. **Stationarity assumption:** GARCH requires persistence < 1 for covariance stationarity. Near-integrated GARCH (α+β ≈ 1) gives very slow vol mean-reversion and may produce extreme long-horizon forecasts.

2. **Univariate only:** Each model is fitted to the *portfolio* returns series, not to individual assets. For covariance matrix estimation, a multivariate DCC-GARCH would be required (not implemented — Ledoit-Wolf shrinkage remains the optimizer input).

3. **Parameter estimation error:** MLE estimates have standard errors (shown in the parameter table). For short histories (< 300 observations), parameters may be unreliable.

4. **Structural breaks:** GARCH cannot distinguish a genuine structural change in volatility from a large shock within a stationary regime. A single outlier day can distort α and β for several months.

5. **Distribution assumption:** Student-t innovations with constant degrees of freedom ν may not capture time-varying tail behaviour across different market regimes.

6. **Not a forecast of directional returns:** GARCH models variance, not the sign or magnitude of returns.

---

## 8. Interview-Ready Explanation

**"Why did you add GARCH to the risk module?"**

> Your existing Historical VaR gives equal weight to all observations in the trailing window.
> If you had a volatile period 6 months ago, it still affects your VaR today with the same weight
> as yesterday. GARCH conditions on the *current* volatility regime — the VaR estimate tightens
> in calm markets and widens rapidly when a shock occurs. That gives more timely risk signals.
>
> I implemented both GARCH(1,1) and GJR-GARCH. The difference is the γ (gamma) parameter in
> GJR. In equity markets, bad news — a negative return — tends to raise volatility more than
> good news of the same size. That's the leverage effect. GARCH can't capture it; GJR can.
> I test whether γ is statistically significant, which tells you whether the asymmetry
> actually exists in this portfolio's history.
>
> Both models are fitted with Student-t innovations rather than Normal because daily returns
> have fat tails — the Normal distribution underestimates extreme events, which is exactly what
> you care about in a VaR context.
>
> The GARCH-VaR is the VaR conditional on *today's* volatility estimate. Using the t-distribution
> quantile rather than Normal is technically correct given the fitting distribution, though the
> difference matters most at very high confidence levels (99%+) where the t has heavier tails
> than Normal for the same estimated variance.

---

## 9. Files Created / Modified

| File | Action |
|------|--------|
| `analytics/garch_volatility.py` | **Created** — 8 public functions |
| `tests/test_garch_volatility.py` | **Created** — 26 tests |
| `pages/4_Risk_Management.py` | **Modified** — added import + 6th tab |
| `validation_reports/garch_validation_report.md` | **Created** — this file |
