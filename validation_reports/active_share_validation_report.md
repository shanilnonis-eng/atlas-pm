# Active Share Validation Report — Atlas PM

**Module:** `analytics/active_share.py`  
**Page integration:** `pages/3_Performance_Analytics.py` — headline metrics + new "Active Share" tab  
**Date:** 2026-06-09  
**Status:** ✅ All tests passing (35/35 new · 288/288 pre-existing · 323/323 total)

---

## 1. Module Purpose

Adds Active Share and Tracking Error to the Performance Analytics page, extending
the benchmark-relative analytics beyond returns and beta.

Active Share is the single most widely cited metric for evaluating whether a portfolio
manager is genuinely active or effectively replicating the benchmark while charging
active management fees.

---

## 2. Formula

### Active Share (Cremers & Petajisto 2009)

```
Active Share = 0.5 × Σ_i |w_i^portfolio − w_i^benchmark|
```

where the sum runs over the union of all assets in either portfolio or benchmark,
and assets present in one but not the other are treated as having weight zero.

**Mathematical properties:**
- AS ∈ [0, 1] always (when both weight vectors sum to 1)
- AS = 0 → portfolio identical to benchmark
- AS = 1 → portfolio has zero overlap with benchmark
- AS is **symmetric**: AS(A vs B) = AS(B vs A)
- For a single-asset benchmark with weight 1.0 in asset X:
  **AS = 1 − w_X** (analytically exact)

### Tracking Error

```
TE = std(r_portfolio − r_benchmark) × sqrt(252)
```

Annualised standard deviation of daily active returns (portfolio minus benchmark).

---

## 3. Benchmark Construction

The app benchmark is SPY (`BENCHMARK_TICKER` in config). The `build_benchmark_weights`
function maps this to the universe label `"US Equities (S&P 500)"` and sets that
asset's benchmark weight to 1.0.

**Implication for multi-asset portfolios:**

Since the benchmark is 100% SPY and the portfolio includes bonds, gold, commodities,
and international equities, the formula reduces to:

```
AS = 1 − w_SPY
```

For example, a balanced portfolio holding 25% US equities has AS = 75%.

This is correct and interpretable: 75% of the portfolio is in assets not held by the
benchmark. The benchmark context note in the UI clarifies that for equity-only mandates,
a multi-constituent index would be used instead.

---

## 4. Cremers & Petajisto 2×2 Framework

| | Low TE (< 4%) | High TE (≥ 4%) |
|---|---|---|
| **High Active Share (≥ 60%)** | Diversified Factor Bets | Active Allocator |
| **Low Active Share (< 60%)** | Closet Indexer | Selective Active |

Thresholds adapted for multi-asset portfolios. The original paper used equity fund data;
multi-asset portfolios naturally have higher AS vs equity-only benchmarks.

---

## 5. Tests Performed

| # | Test | Result |
|---|------|--------|
| 1 | AS = 0 when portfolio equals benchmark | ✅ Pass |
| 2 | AS = 1 when no asset overlap | ✅ Pass |
| 3 | AS = 1 − w_benchmark for single-asset benchmark (6 values of w) | ✅ Pass × 6 |
| 4 | Multi-asset benchmark formula check (manual calculation) | ✅ Pass |
| 5 | AS ∈ [0, 1] for 20 random portfolio/benchmark pairs | ✅ Pass |
| 6 | Union handling — missing assets default to 0 | ✅ Pass |
| 7 | AS is symmetric: AS(A,B) = AS(B,A) | ✅ Pass |
| 8 | Normalisation: doubled weights give same AS | ✅ Pass |
| 9 | TE = 0 when returns are identical | ✅ Pass |
| 10 | TE annualisation: std × √252 (rtol=1e-10) | ✅ Pass |
| 11 | TE > 0 for distinct return series | ✅ Pass |
| 12 | TE handles misaligned dates via reindex | ✅ Pass |
| 13 | Active weight breakdown sums to 0 | ✅ Pass |
| 14 | Active weight breakdown has required columns | ✅ Pass |
| 15 | Active weight breakdown sorted descending | ✅ Pass |
| 16 | build_benchmark_weights identifies SPY correctly | ✅ Pass |
| 17 | build_benchmark_weights sums to 1.0 | ✅ Pass |
| 18 | Classification labels at all bucket boundaries (9 values) | ✅ Pass × 9 |
| 19 | Classification dict has required keys (label, description, color) | ✅ Pass |
| 20 | Quadrant labels for all 4 cases — correct semantics | ✅ Pass |
| 21 | TE classification: Low / Moderate / High bands | ✅ Pass |
| 22 | Integration test: typical multi-asset portfolio → AS = 0.75 | ✅ Pass |

**Total new tests: 35 / 35**  
**Pre-existing tests: 288 / 288 — 0 regressions**

---

## 6. Page Integration

### Headline row (always visible)

A second `render_metric_row` is added below the existing six metrics:

| Metric | Definition |
|--------|-----------|
| Active Share | `0.5 × Σ|w_p − w_b|` |
| Tracking Error (Ann.) | `std(active returns) × √252` |
| Active Classification | Cremers & Petajisto label |
| Information Ratio | Already in stats; surfaced here for context |

### Active Share tab

1. **4 headline metrics** — AS, TE, classification, manager quadrant
2. **Active weight bar chart** — horizontal, green for overweights, red for underweights, sorted descending
3. **Weight detail table** — collapsible, shows exact portfolio/benchmark/active weights
4. **2×2 quadrant chart** — scatter plot with coloured background quadrants; portfolio shown as a diamond at its (TE, AS) coordinates
5. **Interpretation expander** — definition, 2×2 table, multi-asset context note
6. **Benchmark context info box** — clarifies the single-asset benchmark limitation

---

## 7. Limitations

1. **Single-asset benchmark:** The benchmark is 100% SPY. For equity-only comparisons
   (e.g., comparing an equity portfolio vs a diversified index like Russell 1000),
   the benchmark should have constituent weights. This would require index weight data
   not currently in the app.

2. **AS is weight-based, not return-based:** A manager could have AS = 0 (identical to
   index) but take on leverage or derivatives exposure that is not captured by weights.
   TE complements AS precisely for this reason.

3. **Multi-asset interpretation:** High AS vs SPY (e.g. 80%) is expected and normal
   for a diversified multi-asset portfolio — it does not by itself indicate active skill.
   TE is the return-based counterpart that confirms whether the different weights
   actually produce different returns.

4. **Attribution of Active Share:** The module shows *how much* the portfolio differs
   from the benchmark but not *why* — whether the difference comes from a deliberate
   bond allocation or from a sector tilt. Brinson attribution (already in the app)
   addresses this separately.

---

## 8. Interview-Ready Explanation

**"What's Active Share and why does it matter?"**

> Active Share measures what fraction of the portfolio looks nothing like the benchmark.
> If my benchmark is 100% SPY and I hold 25% US equities with the rest in bonds and gold,
> my Active Share is 75% — three-quarters of my portfolio is in positions the benchmark
> doesn't have.
>
> The reason it matters is the Cremers and Petajisto (2009) finding: managers with high
> Active Share — genuinely different from the benchmark — significantly outperformed on
> average, while "closet indexers" (high fees, low Active Share) underperformed. It became
> a standard tool for LPs evaluating whether they're paying for genuine active management.
>
> But Active Share alone isn't enough. You need Tracking Error alongside it. High Active Share
> with low Tracking Error means you have different weights but the positions are all
> correlated with the benchmark anyway — "factor bets" rather than stock picking. You want
> High AS *and* High TE for genuine alpha. I plot both on the 2×2 quadrant to show exactly
> where the portfolio sits.
>
> For multi-asset portfolios vs an equity-only benchmark like SPY, you'd expect Active
> Share to naturally be high — that's just what diversification looks like. The more
> interesting comparison would be against a 60/40 or a global multi-asset index.

---

## 9. Files Created / Modified

| File | Action |
|------|--------|
| `analytics/active_share.py` | **Created** — 7 public functions + 5 constants |
| `tests/test_active_share.py` | **Created** — 35 tests |
| `pages/3_Performance_Analytics.py` | **Modified** — second headline row + new "Active Share" tab |
| `validation_reports/active_share_validation_report.md` | **Created** — this file |
