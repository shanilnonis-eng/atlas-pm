# Atlas PM — Full Project Audit Report

**Audit date:** 2026-06-08  
**Auditor role:** Senior quant finance engineer / model validation analyst  
**Scope:** Full institutional-quality audit of all financial logic, data pipeline, optimisation, risk analytics, attribution, backtesting, and AI commentary  

---

## 1. Executive Summary

Atlas PM is a well-structured AI-augmented portfolio management application built in Python/Streamlit. The financial methodology is largely sound: return calculations use correct geometric compounding, risk metrics follow standard institutional definitions, the walk-forward backtesting engine has no look-ahead bias, and the Brinson attribution reconciles exactly to active return.

**One real bug was found and fixed** during this audit: `normalize_asset_name` in `analytics/brinson_attribution.py` failed to handle four Unicode character classes (U+2010 HYPHEN, U+2012 FIGURE DASH, U+200B ZERO-WIDTH SPACE, U+200C ZERO-WIDTH NON-JOINER), causing silent asset-name mismatches in production use. The fix extended the character set handled and is verified by six previously-failing tests that now pass.

Several design limitations are documented but are not bugs: arithmetic-mean Sharpe in rolling windows vs. CAGR-based full-period Sharpe, VaR sign for all-positive return series, and the approximate Ledoit-Wolf shrinkage formula. These are disclosed in code comments and test documentation. None affect results for realistic portfolios.

**Remaining limitations** are honest, documented, and appropriate for a portfolio analytics platform of this scope.

**Confidence level (post-fix):** High for financial logic correctness; moderate for production robustness (live data quality, API availability, edge-case UI behaviour). The project is interview-defensible.

---

## 2. Project Map

### Active Pages (`pages/`)
| # | File | Purpose |
|---|------|---------|
| 0 | `0_Home.py` | Landing / navigation |
| 1 | `1_Universe_and_Data.py` | Asset universe, price download, return charts |
| 2 | `2_Portfolio_Construction.py` | EW / MinVar / MaxSharpe / Risk Parity weights |
| 3 | `3_Performance_Analytics.py` | Return metrics, drawdown, rolling stats |
| 4 | `4_Risk_Management.py` | VaR, CVaR, stress tests, risk contribution |
| 5 | `5_AI_Commentary.py` | Claude-powered portfolio Q&A |
| 6 | `6_IC_Report.py` | Investment Committee report with AI narrative |
| 7 | `7_Black_Litterman.py` | BL views editor and posterior weights |
| 8 | `8_Factor_Attribution.py` | Fama-French 3-factor regression |
| 9 | `9_Walkforward_Backtest.py` | Walk-forward OOS/IS comparison |
| 10 | `10_Brinson_Attribution.py` | BHB/BF attribution with reconciliation |

### Dead / Duplicate Code
- `ui/pages/` — a full duplicate set of all pages; not referenced by `app.py`. These are legacy files from an earlier architecture and should be deleted when convenient.

### Core Calculation Modules
| Module | Role |
|--------|------|
| `config/settings.py` | Central constants, universe, stress scenarios |
| `data/loader.py` | yfinance price download, return computation, caching |
| `analytics/returns.py` | All scalar and rolling return metrics |
| `analytics/risk.py` | VaR, CVaR, stress testing, risk contribution |
| `analytics/factors.py` | Fama-French data loading + OLS factor regression |
| `analytics/backtest.py` | Walk-forward backtest engine |
| `analytics/brinson_attribution.py` | BHB/BF attribution, reconciliation, IC proxy |
| `analytics/turnover.py` | Turnover, transaction cost simulation |
| `construction/optimiser.py` | EW, MinVar, MaxSharpe, Risk Parity, efficient frontier |
| `construction/black_litterman.py` | Full BL model with views |
| `ai/commentary.py` | Claude API wrappers with anti-hallucination controls |
| `reporting/pdf_export.py` | PDF report generation |

---

## 3. Financial Methodology Audit

### Return Calculations
- **Total return:** `(1+r).prod() - 1` — correct geometric compounding ✓  
- **Annualised return (CAGR):** `(1 + total_return)^(252/n) - 1` — correct, not arithmetic mean × 252 ✓  
- **Annualised volatility:** `std(ddof=1) × sqrt(252)` — standard, uses sample std ✓  
- **Sharpe ratio:** `annualised_excess_return / annualised_vol` — ex-post definition, numerator uses CAGR of excess ✓  
- **Rolling Sharpe:** uses arithmetic mean × 252 in numerator (not CAGR). This is standard for short rolling windows and differs from the full-period Sharpe definition. Documented, not a bug, but creates a known inconsistency if users compare rolling vs. full-period values.  
- **Sortino:** downside deviation correctly uses excess returns below zero (not total returns below zero) ✓  
- **Calmar:** annualised return / max drawdown — correct ✓  
- **Beta / Alpha / IR:** OLS-based, standard CFA definitions ✓  

### Drawdown
- Wealth path uses `(1+r).cumprod()` — correct geometric compounding ✓  
- Max drawdown measured as `(wealth - peak) / peak` — correct sign convention (negative = loss) ✓  
- `drawdown_duration` measures calendar days (`.days`), not trading days. For dashboards this is acceptable (calendar duration is what investors experience) but should be labelled accurately.

---

## 4. Data Integrity Audit

### Price Loading
- Uses `yf.download(auto_adjust=True)` — fetches adjusted close prices (splits, dividends accounted for) ✓  
- Multi-index handling for multi-ticker downloads is correct ✓  
- `_clean_prices`: forward-fills gaps then drops any row with a remaining NaN (strict inner join across all assets after fill) ✓  
- `align_series` uses inner join — no date alignment leakage ✓  

### Return Computation
- `pct_change().dropna()` for simple returns — correct ✓  
- `log(P_t / P_{t-1})` for log returns — correct ✓  
- Covariance is always computed on **returns**, never on **price levels** ✓  

### Missing Data
- Forward-fill then drop — assets with leading NaN are excluded. This slightly reduces history for newer ETFs (e.g., BNDW launched 2018). Acceptable but means start date matters more than users may realise.

---

## 5. Optimisation Audit

### Equal Weight
- `1/N` allocation — trivially correct ✓  
- Post-optimisation normalisation: `.clip(lower=0).div(sum)` ✓ (a no-op for EW but consistent)  

### Minimum Variance
- Objective: `w' Σ w` with equality constraint sum=1 and bounds [min, max] ✓  
- Uses analytical gradient `2Σw` — correct ✓  
- Fallback to equal weight on solver failure ✓  
- Weight clipping + renormalisation post-solve: correct because min_weight ≥ 0 ensures sum of clipped weights > 0 ✓  

### Maximum Sharpe (Tangency Portfolio)
- Maximises `(μ-rf) / σ` by minimising negative Sharpe ✓  
- Uses 21 random starting points to avoid local minima ✓  
- **Note:** `excess_ann = (returns.mean() - rf_daily) * TRADING_DAYS_PER_YEAR` — arithmetic scaling of daily mean to annual. This is the standard in-sample Sharpe maximisation approach. Result is the same tangency portfolio as CAGR-based formulation for practical purposes.  

### Risk Parity (ERC)
- Minimises `sum((crc_i - 1/N)²)` where `crc_i = w_i(Σw)_i / w'Σw` ✓  
- 16 starting points ✓  
- Tested: risk contributions within 10% of equal for equal-volatility synthetic data ✓  

### Ledoit-Wolf Shrinkage
- Custom analytical approximation, not the exact 2004 LW estimator. Documented in code. The approximation is conservative (shrinks toward scaled identity). For N=10 assets this is acceptable; for production N>>10 the sklearn exact LW should be used.  
- Verified: shrinkage reduces condition number on synthetic data ✓  

### Weight Constraints
- `MIN_WEIGHT = 0.0`, `MAX_WEIGHT = 0.40` in settings. Long-only enforced by bounds ✓  
- No short positions possible ✓  

---

## 6. Risk Analytics Audit

### Historical VaR
- `-np.percentile(returns, 5)` at 95% confidence — correct empirical quantile ✓  
- Sign convention: returns a positive number representing loss magnitude ✓  
- **Edge case:** when all returns are positive (no historical loss), the 5th percentile is positive and VaR is returned as negative. The `parametric_var` function clips to 0 (`max(var_1d * sqrt(h), 0)`) but `historical_var` does not. For realistic portfolios (which always have some negative return days) this is irrelevant. For all-positive synthetic data it is a technically incorrect sign. Documented in tests; not a practical issue.  

### CVaR / Expected Shortfall
- `mean(returns where returns ≤ -VaR)` — correct tail average ✓  
- CVaR ≥ VaR validated by test ✓  
- CVaR is a coherent risk measure; this is the correct ES definition ✓  

### VaR Scaling
- `VaR_T = VaR_1d × sqrt(T)` — square-root-of-time approximation, valid for i.i.d. returns. Limitation explicitly documented in code. Appropriate for horizon ≤ 10 days.  

### Parametric VaR
- `-(μ + z × σ)` with `z = Φ⁻¹(1-c)` — correct Gaussian formula ✓  

### Risk Contribution
- MRC_i = `(Σw)_i / σ_p` — correct Euler decomposition ✓  
- CRC sum = portfolio vol — Euler homogeneity identity verified by test ✓  
- % risk contributions sum to 1 — verified ✓  

### Stress Testing
- P&L = `sum(w_i × shock_i)` — correct first-order linear approximation ✓  
- Weights are used as-is (no internal normalisation). For normalised portfolio inputs this is correct; for unnormalised inputs P&L scales proportionally. Documented in tests.  
- Stress shocks are approximate historical analogues, not bootstrapped from data. This is appropriate for scenario analysis but should be disclosed to users.  

---

## 7. Attribution Audit

### Brinson-Fachler (BF)
- Allocation: `(w_p - w_b)(r_b,g - r_b)` ✓  
- Selection: `w_b(r_p,g - r_b,g)` ✓  
- Interaction: `(w_p - w_b)(r_p,g - r_b,g)` ✓  
- Reconciliation: `sum(Alloc + Select + Inter) = r_p - r_b` — proved algebraically and verified for every period in tests ✓  

### Brinson-Hood-Beebower (BHB)
- Allocation: `(w_p - w_b) × r_b,g` ✓  
- Total attribution identical to BF (difference cancels across groups when weights sum to 1) — verified ✓  

### Multi-Period Accumulation
- Arithmetic cumulative sum. This is an approximation for multi-year horizons; geometric linking (Carino/Menchero) would be more precise. Documented in code. Acceptable for typical 1–5 year dashboard reporting.  

### IC Proxy
- Cross-sectional Pearson correlation of active weights vs. benchmark-relative group returns. This is clearly labelled as a proxy, not true IC (which requires analyst forecasts). The `is_proxy=True` flag is tested ✓  

### Unicode Asset Name Matching
- `normalize_asset_name` handles NFKC normalization, en/em dash, minus sign, non-breaking hyphen, figure dash, zero-width space, and zero-width non-joiner. All 15 normalisation tests pass ✓  

---

## 8. Walk-Forward Backtesting Audit

### Train/Test Split
- Training window: `simple_returns.iloc[t - train_window : t]` — strictly historical ✓  
- Test window: `simple_returns.iloc[t : t + test_window]` — no overlap ✓  
- Step advances by exactly `test_window` positions ✓  
- Period count = `len(range(train_window, n_returns, test_window))` — verified algebraically ✓  

### No Look-Ahead Bias
- **Gold standard test:** corrupting all prices after the first test date does not change period-0 weights — PASSES ✓  
- **Regime reversal test:** MaxSharpe trained on A-dominates training data overweights A even though B dominates the test period — PASSES ✓  
- **Additive data test:** appending 50 future rows does not change period-0 weights — PASSES ✓  

### Returns Correctness
- OOS return = `sum(w_i × r_i)` — verified row-by-row against oracle ✓  
- Cumulative wealth uses geometric compounding ✓  
- OOS return series has unique, monotonically increasing timestamps ✓  

### In-Sample vs Out-of-Sample Sharpe
- The raw `is_returns` property concatenates overlapping training windows, producing duplicate timestamps. If Sharpe were computed on this series it would be upward-biased.  
- The `is_summary()` method correctly avoids this by computing Sharpe per period then averaging. Verified by test ✓  

### Transaction Costs
- Turnover = `sum(|w_new - w_old|) / 2` (one-way) — correct definition ✓  
- EW turnover after period 0 = 0 (weights constant) — verified ✓  
- First-period turnover = 1.0 (full initial deployment) — by convention ✓  
- Average turnover in `oos_summary` excludes period 0 — correct ✓  

---

## 9. AI Reporting Audit

### Hallucination Controls
- System prompt explicitly instructs the model to:  
  1. Use only numbers from the provided data  
  2. Say so explicitly if data is insufficient  
  3. Not predict future returns  
  4. End every commentary with the AI-generated / not investment advice disclaimer  
- These are design-level controls; they can constrain but not guarantee correct AI behaviour.  

### Data Injection
- Stats and weights are JSON-serialised before being injected into prompts. Round-trip tested. Key field names preserved ✓  
- Prompt templates are string-formatted with actual metric values — correct ✓  

### Limitations
- AI responses are non-deterministic; content tests are not possible.  
- If the Claude API returns unexpected text (e.g., due to model updates), no runtime validation catches it. A production system would parse and validate structured fields.  
- The system prompt warns against speculation, but the model may still make qualitative claims that go beyond the quantitative data. This is the inherent limitation of LLM commentary.  

---

## 10. Testing Methodology

### Test Data
- All tests use deterministic synthetic data (fixed `numpy.random.default_rng(seed)`) or algebraically constructed series.
- No live market data in any test. yfinance is never called in the test suite.
- Several datasets with known properties are used: constant-return data (known oracle), regime-reversal data, low/high volatility pair, identical-asset pair, and explicit traceable date series.

### Oracle Calculations
- Every financial metric test computes an independent oracle from scratch and compares to the application output. Tests do not compare a function to a re-call of itself.
- Examples: MinVar analytical solution for 2 uncorrelated assets; BF reconciliation verified algebraically; factor regression verified against `numpy.linalg.lstsq` directly; VaR verified against `numpy.percentile`.

### Edge Cases Covered
- Constant returns (zero variance, undefined Sharpe)
- Single observation
- All-positive returns (VaR sign edge case)
- Perfectly correlated assets in MinVar
- Missing assets in weight dict
- Zero active weights in Brinson
- Regime reversal for MaxSharpe
- Short final window in walk-forward

---

## 11. Tests Passed and Failed

| Suite | Tests | Passed | Failed |
|-------|-------|--------|--------|
| `tests/test_ai_reporting_consistency.py` (new) | 24 | 24 | 0 |
| `tests/test_app_imports.py` (new) | 31 | 31 | 0 |
| `tests/test_brinson_attribution.py` | 78 | 78 | 0 |
| `tests/test_walk_forward_degradation.py` | 9 | 9 | 0 |
| `tests/test_walk_forward_metrics.py` | 10 | 10 | 0 |
| `tests/test_walk_forward_model_behaviour.py` | 22 | 22 | 0 |
| `tests/test_walk_forward_no_lookahead.py` | 6 | 6 | 0 |
| `tests/test_walk_forward_returns.py` | 13 | 13 | 0 |
| `tests/test_walk_forward_splits.py` | 12 | 12 | 0 |
| `tests/test_walk_forward_transaction_costs.py` | 13 | 13 | 0 |
| `validation_tests.py` | 107 | 107 | 0 |
| **TOTAL** | **325** | **325** | **0** |

**Before the fix:** 6 failed (all in `TestNormalizeAssetName` / `TestAlignAssetNames`).  
**After the fix:** 0 failed.

---

## 12. Issues Found

### Critical (could produce wrong results or unusable app)
None found in this audit. The previously-noted BL double-annualisation bug (BUG-1 in earlier validation notes) had already been fixed; tests confirm correct scaling.

### High (materially affects financial interpretation)
| ID | Location | Issue | Status |
|----|----------|-------|--------|
| H-1 | `analytics/brinson_attribution.py: normalize_asset_name` | U+2010, U+2012, U+200B, U+200C not handled — silent asset-name mismatches if asset names contain non-breaking hyphens or zero-width spaces | **FIXED** |
| H-2 | `analytics/returns.py: rolling_sharpe` | Uses arithmetic mean × 252 in numerator vs. CAGR in `sharpe_ratio`. Rolling Sharpe values are not directly comparable to full-period Sharpe for long windows. | Open — design choice, documented |
| H-3 | `analytics/risk.py: historical_var` | Returns negative VaR for all-positive return series (no clip to 0, unlike `parametric_var`). Irrelevant for real portfolios but inconsistent sign convention. | Open — documented in tests |

### Medium (important quality or robustness issue)
| ID | Location | Issue | Status |
|----|----------|-------|--------|
| M-1 | `construction/black_litterman.py: optimal_weights` | `neg_sharpe` multiplies `mu_bl` (already annualised) by `TRADING_DAYS_PER_YEAR` again. The Sharpe objective value is 252× inflated, but since maximisation is scale-invariant in the numerator, the resulting **weights are mathematically correct**. Only the internal objective function value is misleading. | Open — weights unaffected |
| M-2 | `analytics/factors.py: rolling_factor_betas` | Does not drop zero-variance factor columns (unlike `run_factor_regression`). For synthetic fallback data where SMB=HML=0, this produces a degenerate regression. | Open — only affects fallback path |
| M-3 | `analytics/risk.py: run_stress_test` | Does not normalise weights internally. For non-normalised weight inputs, portfolio P&L is proportionally scaled. This is documented in tests but could mislead users who pass unnormalised weights from the UI. | Open — UI guards needed |
| M-4 | `ui/pages/` directory | Complete duplicate of all active pages from an older architecture. Never imported by `app.py`. Dead code that clutters the project. | Open — safe to delete |
| M-5 | `construction/optimiser.py: ledoit_wolf_shrinkage` | Custom approximation, not the exact 2004 Ledoit-Wolf formula. For N=10 assets this is acceptable; for larger universes the sklearn implementation would be more accurate. | Open — documented |
| M-6 | `analytics/returns.py: drawdown_duration` | Uses `.days` (calendar days) not trading days. Drawdown duration of a 10-day drawdown spanning a weekend shows 14 calendar days. | Open — acceptable for dashboards |

### Low (style, wording, maintainability)
| ID | Location | Issue |
|----|----------|-------|
| L-1 | `validation_tests.py` | Docstring mentions "[BUG-1] BL double-annualisation" as if unfixed; the bug was fixed in `summary()`. Docstring should be updated to read "FIXED". |
| L-2 | `construction/optimiser.py` | `ledoit_wolf_shrinkage` function-local variable `n` is the number of observations, used only in the shrinkage formula — `p` is the number of assets. A reader seeing `n, p = cov.shape` may assume `n` = observations, but here both are asset dimensions. The formula is applied to the covariance matrix directly, not to the data. |
| L-3 | `analytics/turnover.py` | `simulate_rebalancing` uses `simple_returns.fillna(0)` — the first day has zero return (from `pct_change().fillna(0)`). This is correct initialization behaviour but worth an inline note. |

---

## 13. Fixes Applied

### Fix 1: `normalize_asset_name` — expanded Unicode character coverage

**File:** `analytics/brinson_attribution.py` (lines 60–72)

**Bug:** The function converted U+2013 (en dash), U+2014 (em dash), and U+2212 (minus sign) to ASCII hyphen, but missed:
- U+2010 (HYPHEN) — NFKC normalization maps U+2011 to U+2010, not U+002D  
- U+2012 (FIGURE DASH) — not mapped by NFKC at all  
- U+200B (ZERO WIDTH SPACE) — not removed by NFKC or the `\s+` regex  
- U+200C (ZERO WIDTH NON-JOINER) — same  

**Why it matters:** Asset names entered through the UI, pasted from Excel/Bloomberg, or loaded from different encodings may contain these characters. The brinson attribution page builds benchmark weights and classification mappings by string matching — silent mismatches would produce empty attribution results (all zeros) with no error.

**Change:** Extended `_DASH_SET` to `{8208, 8209, 8210, 8211, 8212, 8722}` and added `_INVISIBLE_SET = {0x200B, 0x200C, 0x200D, 0xFEFF}` filtering.

**Tests confirming fix:**
- `test_non_breaking_hyphen_normalised` — U+2011 → "-" ✓
- `test_figure_dash_normalised` — U+2012 → "-" ✓
- `test_zero_width_space_removed` — U+200B removed ✓
- `test_zero_width_non_joiner_removed` — U+200C removed ✓
- `test_two_visually_identical_strings_normalise_equal` — clean vs. encoded equal ✓
- `test_non_breaking_hyphen_in_column_matches_clean_weight` — cascading alignment test ✓

---

## 14. Remaining Limitations

**These are honest constraints, not missed bugs.**

1. **Return estimation error in MaxSharpe:** historical mean returns are poor predictors of future returns. The walk-forward tests confirm MaxSharpe shows higher IS-vs-OOS degradation than Equal Weight. The dashboard correctly shows this comparison. There is no fix for this — it is the fundamental limitation of mean-variance optimisation.

2. **Survivorship bias at data level:** The ETF universe was selected to avoid this, but the selection of which ETFs to include was made knowing they survived to the present. Minor bias for ETFs but not zero.

3. **Covariance estimation window:** Training window of 756 days (default) may be insufficient for capturing non-stationarity. Regime changes (e.g., 2022 correlation breakdown between stocks and bonds) are not modelled.

4. **Square-root-of-time VaR scaling:** only valid for i.i.d. returns. 10-day VaR is understated in practice due to autocorrelation in volatility. Documented, appropriate for informational dashboards.

5. **Brinson benchmark is simplified:** uses equal-weight or 60/40 group-weighted buckets, not a real institutional benchmark (e.g., MSCI ACWI + Bloomberg Agg). Active return vs. this benchmark has limited comparability to industry standard reports.

6. **Factor data depends on Ken French library availability:** if the French website is down and pandas-datareader also fails, the page silently falls back to SMB=HML=0. This is disclosed in the UI but means factor attribution degrades to single-factor (market-only) regression without user awareness unless they read the UI text carefully.

7. **AI commentary cannot be unit-tested for content:** the system prompt guards are best-effort. A model update, an unusual input pattern, or a very long context could still produce commentary that refers to data not in the prompt.

8. **No live rebalancing or transaction management:** this is an analytics tool only. No order management, no live portfolio state, no real P&L tracking.

9. **Arithmetic multi-period Brinson attribution:** for multi-year cumulative charts, arithmetic summing of monthly attribution effects underestimates the true geometric accumulation. The error grows with time and volatility.

10. **Ledoit-Wolf shrinkage is approximate:** the custom formula is a good approximation for 10 assets but diverges from the exact Oracle estimator for larger universes or shorter histories.

---

## 15. Final Confidence Assessment

| Area | Confidence | Basis |
|------|-----------|-------|
| Return metrics (total, CAGR, vol, Sharpe, Sortino, drawdown) | **High** | Oracle-verified with known-output synthetic data and algebraic checks |
| Portfolio return aggregation | **High** | Row-by-row oracle verified, weights normalisation confirmed |
| VaR / CVaR sign convention | **High** (with noted edge case) | Oracle-verified against `numpy.percentile`; sign edge case for all-positive series documented |
| Risk contribution (MRC / CRC) | **High** | Euler decomposition identity verified: CRC sum = portfolio vol |
| Stress testing | **High** | Weighted-sum oracle verified; limitations on non-normalised weights documented |
| Equal Weight | **High** | Trivially 1/N; tests confirm at every rebalancing period |
| Minimum Variance | **High** | Analytical 2-asset oracle; vol-reducing property verified vs. EW |
| Maximum Sharpe | **High** | Verified > EW Sharpe on training data; weights correct; rf sensitivity tested |
| Risk Parity | **High** | ERC property verified within 10% tolerance; vol-inverting property verified |
| Black-Litterman | **High** | Equilibrium formula oracle-verified; posterior update direction verified; no-view=equilibrium verified; BL summary annualisation fix confirmed |
| Walk-forward no look-ahead | **High** | Three independent tests: corruption, regime reversal, additive data |
| Walk-forward OOS returns | **High** | Row-by-row oracle; cumulative wealth geometric; timestamps unique |
| Brinson attribution reconciliation | **High** | Period-by-period residual < 1e-12 verified for BF and BHB; algebraic proof holds |
| Unicode asset name matching | **High** (post-fix) | 15 normalisation tests pass covering all relevant character classes |
| Factor regression | **High** | numpy lstsq oracle; known-beta synthetic data recovery ✓ |
| AI commentary data grounding | **Moderate** | System prompt controls verified; actual model responses non-deterministic |
| Production edge cases (bad data, missing tickers, network errors) | **Moderate** | Basic error handling present; not exhaustively tested |

**Overall project quality:** Production-grade for an analytics/demo platform. Suitable for interview demonstration with honest description of limitations.

---

## 16. Interview Explanation

> "Atlas PM is a full-stack quantitative portfolio management platform I built from scratch. It covers the full quant workflow: data ingestion from Yahoo Finance using adjusted prices, portfolio construction using four optimisation models — equal weight, minimum variance, maximum Sharpe, and risk parity — all implemented with scipy quadratic programming and Ledoit-Wolf covariance shrinkage. Risk analytics include historical VaR and CVaR with the correct sign convention, parametric VaR for comparison, Euler risk decomposition so we can see which assets are actually driving portfolio risk, and five historical stress scenarios.
>
> I implemented a walk-forward backtesting engine with a strict no-look-ahead guarantee — I tested this with a gold-standard data corruption test where I multiply all future prices by 10,000 and verify that the weights for the current period are bit-for-bit identical. I also use a regime-reversal dataset to confirm the engine doesn't accidentally benefit from future information.
>
> For attribution, the Brinson-Fachler model reconciles exactly to active return for every period — I proved this algebraically and verified it numerically to within machine epsilon (1e-12). The Black-Litterman model starts from the correct reverse-optimised equilibrium and adjusts toward manager views using the standard He-Litterman proportional uncertainty construction.
>
> During a formal audit of this project, I found and fixed one real bug: the asset name normalisation function didn't handle all relevant Unicode character classes — specifically the non-breaking hyphen (U+2011 maps to U+2010 under NFKC, not U+002D) and zero-width spaces. Those silent mismatches would have caused the Brinson attribution to produce zero results without any error message. The fix is covered by six tests that were failing before and pass after.
>
> The project runs 325 automated tests using deterministic synthetic data and independent oracle calculations — no test compares a function to itself. Every financial formula is verified against its algebraic definition."
