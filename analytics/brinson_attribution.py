"""
Brinson-Hood-Beebower (BHB) and Brinson-Fachler (BF) Performance Attribution.

Decomposes active portfolio return versus a benchmark into three effects:

  1. Allocation Effect  — was capital over/under-weighted in the right groups?
  2. Selection Effect   — did within-group security/asset selection add value?
  3. Interaction Effect — combined effect of over/underweighting AND outperforming

═══════════════════════════════════════════════════════════════════════════════
Formulas  (group g, period t)
═══════════════════════════════════════════════════════════════════════════════

  w_p,g  : portfolio weight in group g
  w_b,g  : benchmark weight in group g
  r_p,g  : portfolio return attributable to group g (weighted average of assets)
  r_b,g  : benchmark return attributable to group g
  r_b    : total benchmark return = Σ_g(w_b,g × r_b,g)
  r_p    : total portfolio return  = Σ_g(w_p,g × r_p,g)

Brinson-Fachler (default — recommended for relative attribution):
  Allocation  = (w_p,g - w_b,g) × (r_b,g - r_b)
  Selection   = w_b,g × (r_p,g - r_b,g)
  Interaction = (w_p,g - w_b,g) × (r_p,g - r_b,g)

Brinson-Hood-Beebower (original 1986 paper):
  Allocation  = (w_p,g - w_b,g) × r_b,g
  Selection   = w_b,g × (r_p,g - r_b,g)
  Interaction = (w_p,g - w_b,g) × (r_p,g - r_b,g)

Reconciliation — holds for BOTH methods when Σ w_p,g = 1 and Σ w_b,g = 1:
  Σ_g [Allocation + Selection + Interaction] = r_p - r_b   (active return)

Difference between methods:
  BF allocation is relative to benchmark total return (r_b,g - r_b).
  BHB allocation is relative to zero (r_b,g).
  Their difference per group sums to zero across all groups, so the
  TOTAL attribution is identical for both methods.

References:
  Brinson, G.P., Hood, L.R. & Beebower, G.L. (1986)
    'Determinants of Portfolio Performance', Financial Analysts Journal, 42(4), 39-44.
  Brinson, G.P. & Fachler, N. (1985)
    'Measuring Non-US Equity Portfolio Performance',
    Journal of Portfolio Management, 11(3), 73-76.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional
import numpy as np
import pandas as pd


# ─── Asset name normalisation ────────────────────────────────────────────────


def normalize_asset_name(name):
    """
    Normalise asset names so portfolio weights, return columns,
    benchmark weights, and classification mappings can be matched reliably.
    """
    if name is None:
        return ""
    text = str(name)
    text = unicodedata.normalize("NFKC", text)
    # Replace dash/hyphen family with ASCII hyphen-minus (U+002D).
    # NFKC maps U+2011 (non-breaking hyphen) → U+2010 (hyphen), so U+2010
    # must be included.  U+2012 (figure dash) is not mapped by NFKC at all.
    # Codepoints: 8208=U+2010 HYPHEN, 8209=U+2011 NON-BREAKING HYPHEN,
    #             8210=U+2012 FIGURE DASH, 8211=U+2013 EN DASH,
    #             8212=U+2014 EM DASH,    8722=U+2212 MINUS SIGN
    _DASH_SET = frozenset({8208, 8209, 8210, 8211, 8212, 8722})
    text = ''.join('-' if ord(c) in _DASH_SET else c for c in text)
    # Remove zero-width and other invisible Unicode characters that are
    # visually indistinguishable from nothing but break string equality.
    # U+200B ZWSP, U+200C ZWNJ, U+200D ZWJ, U+FEFF BOM/ZWNBSP
    _INVISIBLE_SET = frozenset({0x200B, 0x200C, 0x200D, 0xFEFF})
    text = ''.join(c for c in text if ord(c) not in _INVISIBLE_SET)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def align_asset_names(
    weight_source: "pd.Series | dict",
    return_columns: "pd.Index | list[str]",
    classification: "dict[str, str]",
) -> dict:
    """
    Normalise and align asset names across portfolio weights, return data, and
    the asset-class classification mapping.

    Returns a dict with:
      'available_assets'         : list[str] — canonical names (from return_columns)
                                   for assets that have portfolio weights
      'port_weights'             : pd.Series — weights re-indexed by canonical name
      'canonical_classification' : dict[str, str] — {canonical_name: group}
      'debug'                    : dict — diagnostic counters and name lists
    """
    # ── Normalise weights ──────────────────────────────────────────────────────
    if isinstance(weight_source, pd.Series):
        raw_weight_keys = list(weight_source.index)
        weight_values   = {k: float(weight_source[k]) for k in raw_weight_keys}
    elif isinstance(weight_source, pd.DataFrame):
        # Take the first row (or last row — the latest computed weights)
        s = weight_source.iloc[-1] if len(weight_source) > 0 else weight_source.iloc[0]
        raw_weight_keys = list(s.index)
        weight_values   = {k: float(s[k]) for k in raw_weight_keys}
    else:
        # plain dict
        raw_weight_keys = list(weight_source.keys())
        weight_values   = {k: float(weight_source[k]) for k in raw_weight_keys}

    # ── Normalise return columns ───────────────────────────────────────────────
    raw_ret_cols      = list(return_columns)
    norm_to_ret_col   = {normalize_asset_name(c): c for c in raw_ret_cols}
    norm_to_weight_key = {normalize_asset_name(k): k for k in raw_weight_keys}

    # ── Normalise classification ───────────────────────────────────────────────
    norm_to_class = {normalize_asset_name(k): v for k, v in classification.items()}

    # ── Intersect ─────────────────────────────────────────────────────────────
    common_norm     = sorted(set(norm_to_ret_col) & set(norm_to_weight_key))
    only_in_returns = sorted(set(norm_to_ret_col) - set(norm_to_weight_key))
    only_in_weights = sorted(set(norm_to_weight_key) - set(norm_to_ret_col))

    # Canonical names come from return_columns (the DataFrame we'll slice)
    available_assets = [norm_to_ret_col[n] for n in common_norm]

    # Rebuild weight Series with canonical names
    port_weights = pd.Series(
        {norm_to_ret_col[n]: weight_values[norm_to_weight_key[n]] for n in common_norm}
    )
    if port_weights.sum() > 0:
        port_weights = port_weights / port_weights.sum()

    # Rebuild classification with canonical names
    canonical_classification = {
        norm_to_ret_col[n]: norm_to_class.get(n, "Other")
        for n in common_norm
    }

    return {
        "available_assets":          available_assets,
        "port_weights":               port_weights,
        "canonical_classification":   canonical_classification,
        "debug": {
            "raw_weight_keys":   raw_weight_keys,
            "raw_ret_cols":      raw_ret_cols,
            "norm_weight_keys":  [normalize_asset_name(k) for k in raw_weight_keys],
            "norm_ret_cols":     [normalize_asset_name(c) for c in raw_ret_cols],
            "repr_weight_keys":  [repr(k) for k in raw_weight_keys],
            "repr_ret_cols":     [repr(c) for c in raw_ret_cols],
            "n_common":          len(common_norm),
            "n_weight_keys":     len(raw_weight_keys),
            "n_ret_cols":        len(raw_ret_cols),
            "only_in_returns":   [norm_to_ret_col[n] for n in only_in_returns],
            "only_in_weights":   [norm_to_weight_key[n] for n in only_in_weights],
            "common_assets":     available_assets,
        },
    }


# ─── Default asset classification ────────────────────────────────────────────

DEFAULT_CLASSIFICATION: dict[str, str] = {
    "US Equities (S&P 500)":           "Equities",
    "UK Equities (FTSE 100)":          "Equities",
    "European Equities (Euro Stoxx)":  "Equities",
    "Emerging Markets":                "Equities",
    "US Aggregate Bonds":              "Fixed Income",
    "Global Bonds (Hedged)":           "Fixed Income",
    "Gold":                            "Alternatives",
    "Commodities (Broad)":             "Alternatives",
    "REITs (Global)":                  "Alternatives",
    "Cash Proxy (T-Bills)":            "Cash",
}

# Group weights for a 60/40-style simplified benchmark
BENCHMARK_60_40_GROUP_WEIGHTS: dict[str, float] = {
    "Equities":     0.60,
    "Fixed Income": 0.30,
    "Alternatives": 0.05,
    "Cash":         0.05,
}


# ─── Weight helpers ───────────────────────────────────────────────────────────

def calculate_group_weights(
    asset_weights: pd.Series,
    classification: dict[str, str],
) -> pd.Series:
    """
    Aggregate individual asset weights to group level.

    Assets not present in classification are assigned group "Other".
    Returns a Series indexed by group name, values summing to ≈ 1.
    """
    by_group: dict[str, float] = {}
    for asset, weight in asset_weights.items():
        group = classification.get(str(asset), "Other")
        by_group[group] = by_group.get(group, 0.0) + float(weight)
    return pd.Series(by_group)


def build_benchmark_weights(
    available_assets: list[str],
    classification: dict[str, str],
    method: str = "equal_weight",
    custom_group_weights: Optional[dict[str, float]] = None,
) -> pd.Series:
    """
    Construct benchmark asset weights.

    Parameters
    ----------
    available_assets    : assets actually present in the return series
    classification      : maps asset_label → group_name
    method              : "equal_weight" or "group_weighted"
    custom_group_weights: required when method="group_weighted"
                          e.g. {"Equities": 0.60, "Fixed Income": 0.30, ...}

    Returns
    -------
    pd.Series indexed by asset label, values sum to 1.

    Notes
    -----
    "equal_weight" : 1/n for every asset — the simplest possible benchmark.
    "group_weighted": equal weight within each group, with group totals
                      determined by custom_group_weights (normalised to 1
                      over groups that actually have available assets).
    """
    n = len(available_assets)
    if n == 0:
        raise ValueError("available_assets is empty")

    if method == "equal_weight":
        return pd.Series(1.0 / n, index=available_assets)

    if method == "group_weighted":
        if custom_group_weights is None:
            raise ValueError("custom_group_weights required for method='group_weighted'")

        # count how many available assets fall in each group
        group_counts: dict[str, int] = {}
        for a in available_assets:
            g = classification.get(a, "Other")
            group_counts[g] = group_counts.get(g, 0) + 1

        # normalise group weights to only the groups that have assets
        total_gw = sum(custom_group_weights.get(g, 0.0) for g in group_counts)
        if total_gw <= 0:
            # fallback: equal weight across groups
            total_gw = len(group_counts)
            gw_norm = {g: 1.0 / total_gw for g in group_counts}
        else:
            gw_norm = {
                g: custom_group_weights.get(g, 0.0) / total_gw
                for g in group_counts
            }

        weights: dict[str, float] = {}
        for a in available_assets:
            g = classification.get(a, "Other")
            weights[a] = gw_norm[g] / group_counts[g]

        return pd.Series(weights)

    raise ValueError(f"Unknown method: '{method}'. Use 'equal_weight' or 'group_weighted'.")


# ─── Return computation ───────────────────────────────────────────────────────

def calculate_group_returns(
    asset_returns: pd.DataFrame,
    port_asset_weights: pd.Series,
    bench_asset_weights: pd.Series,
    classification: dict[str, str],
    freq: str = "ME",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute group-level period returns for portfolio and benchmark.

    The portfolio group return for group g is the return you would earn
    on the *proportion* of the portfolio invested in g, i.e. a within-group
    weighted average using normalised portfolio weights.

    If a group has zero portfolio weight (portfolio holds nothing there),
    the portfolio group return is set equal to the benchmark group return
    so that the selection and interaction effects remain zero for that group.
    All active exposure for that group then flows through the allocation effect.

    Parameters
    ----------
    asset_returns       : daily simple returns, columns = asset labels
    port_asset_weights  : Series {asset_label: weight}, sums to ~1
    bench_asset_weights : Series {asset_label: weight}, sums to ~1
    classification      : {asset_label: group_name}
    freq                : pandas resample frequency, default "ME" (month-end)

    Returns
    -------
    (port_group_returns, bench_group_returns) — DataFrames, index=periods, cols=groups
    """
    # Identify groups present in the available assets
    available = [c for c in asset_returns.columns]
    groups = sorted({classification.get(a, "Other") for a in available})

    # Compound to period returns
    period_asset_rets = (1 + asset_returns).resample(freq).prod() - 1

    port_group: dict[str, pd.Series] = {}
    bench_group: dict[str, pd.Series] = {}

    for group in groups:
        group_assets = [
            a for a in available if classification.get(a, "Other") == group
            and a in period_asset_rets.columns
        ]
        if not group_assets:
            continue

        monthly_slice = period_asset_rets[group_assets]

        # Portfolio within-group weights
        pw = port_asset_weights.reindex(group_assets).fillna(0.0)
        pw_sum = pw.sum()
        if pw_sum > 1e-12:
            pw_norm = pw / pw_sum
            port_group[group] = (monthly_slice * pw_norm).sum(axis=1)
        else:
            # Zero portfolio weight in this group — placeholder; overwritten below
            port_group[group] = pd.Series(np.nan, index=monthly_slice.index)

        # Benchmark within-group weights
        bw = bench_asset_weights.reindex(group_assets).fillna(0.0)
        bw_sum = bw.sum()
        if bw_sum > 1e-12:
            bw_norm = bw / bw_sum
            bench_group[group] = (monthly_slice * bw_norm).sum(axis=1)
        else:
            bench_group[group] = monthly_slice.mean(axis=1)

    port_df  = pd.DataFrame(port_group)
    bench_df = pd.DataFrame(bench_group)

    # Where portfolio group return is NaN (zero weight group),
    # set it equal to benchmark to neutralise selection / interaction effects
    for col in port_df.columns:
        mask = port_df[col].isna()
        if mask.any() and col in bench_df.columns:
            port_df.loc[mask, col] = bench_df.loc[mask, col]

    return port_df, bench_df


# ─── Core attribution engine ──────────────────────────────────────────────────

def calculate_brinson_attribution(
    port_group_weights: pd.Series,
    bench_group_weights: pd.Series,
    port_group_returns: pd.DataFrame,
    bench_group_returns: pd.DataFrame,
    method: str = "brinson_fachler",
) -> pd.DataFrame:
    """
    Period-by-period Brinson attribution.

    Parameters
    ----------
    port_group_weights  : w_p,g — constant across periods (static portfolio)
    bench_group_weights : w_b,g — constant across periods
    port_group_returns  : rows=periods, cols=groups  (r_p,g,t)
    bench_group_returns : rows=periods, cols=groups  (r_b,g,t)
    method              : "brinson_fachler" (default) or "bhb"

    Returns
    -------
    Long-form DataFrame with one row per (period, group) containing:
      Period, Group, Port Weight, Bench Weight, Active Weight,
      Port Return, Bench Return, Bench Total, Active Return,
      Alloc Effect, Select Effect, Inter Effect, Total Effect
    """
    if method not in {"brinson_fachler", "bhb"}:
        raise ValueError(
            f"method must be 'brinson_fachler' or 'bhb', got '{method}'"
        )

    # Intersect groups across all inputs
    groups = sorted(
        set(port_group_weights.index)
        & set(bench_group_weights.index)
        & set(port_group_returns.columns)
        & set(bench_group_returns.columns)
    )
    if not groups:
        raise ValueError("No groups in common across weights and return DataFrames")

    w_p = port_group_weights[groups].copy()
    w_b = bench_group_weights[groups].copy()

    # Normalise to exactly 1 (guard against floating-point drift)
    w_p = w_p / w_p.sum()
    w_b = w_b / w_b.sum()

    records: list[dict] = []

    for period in port_group_returns.index:
        if period not in bench_group_returns.index:
            continue

        r_p_g = port_group_returns.loc[period, groups]
        r_b_g = bench_group_returns.loc[period, groups]

        # Total benchmark return for this period
        r_b_total = float((w_b * r_b_g).sum())

        for g in groups:
            wp = float(w_p[g])
            wb = float(w_b[g])
            rp = float(r_p_g[g])
            rb = float(r_b_g[g])

            aw = wp - wb        # active weight
            ar = rp - rb        # within-group active return

            if method == "brinson_fachler":
                alloc = aw * (rb - r_b_total)
            else:  # bhb
                alloc = aw * rb

            sel   = wb * ar
            inter = aw * ar
            total = alloc + sel + inter

            records.append({
                "Period":        period,
                "Group":         g,
                "Port Weight":   wp,
                "Bench Weight":  wb,
                "Active Weight": aw,
                "Port Return":   rp,
                "Bench Return":  rb,
                "Bench Total":   r_b_total,
                "Alloc Effect":  alloc,
                "Select Effect": sel,
                "Inter Effect":  inter,
                "Total Effect":  total,
            })

    return pd.DataFrame(records)


# ─── Active return from attribution-consistent weights ────────────────────────

def calculate_period_active_return(
    port_group_weights: pd.Series,
    bench_group_weights: pd.Series,
    port_group_returns: pd.DataFrame,
    bench_group_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute the per-period active return consistent with the attribution inputs.

    Uses the same weights and group returns as calculate_brinson_attribution.
    This will reconcile exactly with the sum of attribution effects.

    Note: this is the 'attribution-consistent' portfolio return, computed as
    Σ_g(w_p,g × r_p,g,t).  It may differ slightly from the actual compounded
    portfolio return in session state due to daily-vs-monthly weighting.
    """
    groups = sorted(
        set(port_group_weights.index)
        & set(bench_group_weights.index)
        & set(port_group_returns.columns)
        & set(bench_group_returns.columns)
    )
    w_p = port_group_weights[groups]
    w_b = bench_group_weights[groups]
    w_p = w_p / w_p.sum()
    w_b = w_b / w_b.sum()

    port_total  = (port_group_returns[groups]  * w_p).sum(axis=1)
    bench_total = (bench_group_returns[groups] * w_b).sum(axis=1)
    active = port_total - bench_total
    active.name = "Active Return"
    return active


# ─── Cumulative attribution ───────────────────────────────────────────────────

def calculate_cumulative_attribution(
    attribution_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Arithmetic cumulative sum of attribution effects over time.

    Sums Alloc, Select, Inter, Total across groups for each period,
    then takes running cumulative sum.

    Note: arithmetic cumulation is an approximation for multi-period
    attribution.  Geometric linking (Carino, Menchero) is more precise
    for horizons beyond two years but adds complexity.  Arithmetic is
    standard for short-to-medium horizons and for dashboard reporting.

    Returns
    -------
    DataFrame indexed by period, columns:
      Cum Alloc, Cum Select, Cum Inter, Cum Total, Cum Active Return
    """
    period_effects = attribution_df.groupby("Period")[
        ["Alloc Effect", "Select Effect", "Inter Effect", "Total Effect"]
    ].sum()

    cum = period_effects.cumsum()
    cum.columns = ["Cum Alloc", "Cum Select", "Cum Inter", "Cum Total"]

    # Also store period-level active return for the chart
    period_active = attribution_df.groupby("Period")["Total Effect"].sum()
    cum["Cum Active Return"] = period_active.cumsum()

    return cum


# ─── Reconciliation validator ────────────────────────────────────────────────

def validate_brinson_reconciliation(
    attribution_df: pd.DataFrame,
    active_returns: pd.Series,
    tolerance: float = 1e-8,
) -> dict:
    """
    Verify that sum of effects equals active return for every period.

    Parameters
    ----------
    attribution_df : output of calculate_brinson_attribution
    active_returns : output of calculate_period_active_return (same inputs)
    tolerance      : absolute tolerance for floating-point comparison

    Returns
    -------
    dict with keys:
      'pass'             : bool — True if max_residual <= tolerance
      'max_residual'     : float — worst absolute reconciliation error
      'mean_residual'    : float — mean absolute reconciliation error
      'periods_failing'  : list  — periods where |residual| > tolerance
      'residuals'        : pd.Series — per-period residual
      'n_periods'        : int
    """
    period_totals = attribution_df.groupby("Period")["Total Effect"].sum()

    aligned = pd.DataFrame({
        "Attribution": period_totals,
        "Active":      active_returns,
    }).dropna()

    residuals = aligned["Attribution"] - aligned["Active"]
    max_res   = float(residuals.abs().max()) if len(residuals) else 0.0
    mean_res  = float(residuals.abs().mean()) if len(residuals) else 0.0
    failing   = list(aligned.index[residuals.abs() > tolerance])

    return {
        "pass":            max_res <= tolerance,
        "max_residual":    max_res,
        "mean_residual":   mean_res,
        "periods_failing": failing,
        "residuals":       residuals,
        "n_periods":       len(aligned),
    }


# ─── Allocation effectiveness proxy (not true IC) ────────────────────────────

def calculate_ic_proxy(
    active_weights: pd.Series,
    bench_group_returns: pd.DataFrame,
    bench_total_returns: pd.Series,
) -> dict:
    """
    Compute the Allocation Effectiveness Proxy.

    WARNING: This is NOT the true Information Coefficient.

    True IC = cross-sectional correlation between analyst forecast scores
    and subsequent realised returns.  No forecast scores exist in this system.

    This proxy measures: for each monthly period, are the groups with
    higher active weight (overweight vs benchmark) also the groups that
    outperform the benchmark that month?

    Method: Pearson correlation between active_weights (constant, by group)
    and benchmark-relative group returns (r_b,g,t - r_b,t) for each period.
    The average of these cross-sectional correlations is reported.

    Interpretation:
      > 0  : overweighted groups tended to outperform — allocation skill proxy
      ≈ 0  : no relationship between active weights and subsequent group returns
      < 0  : overweighted groups tended to underperform

    Parameters
    ----------
    active_weights      : w_p,g - w_b,g, indexed by group (constant)
    bench_group_returns : r_b,g,t — rows=periods, cols=groups
    bench_total_returns : r_b,t — indexed by period

    Returns
    -------
    dict with:
      'ic_proxy'     : float (mean cross-sectional correlation) or None
      't_stat'       : float (t-stat under H0: IC=0) or None
      'n_periods'    : int
      'is_proxy'     : True (always — to remind caller this is not true IC)
    """
    groups = [g for g in active_weights.index if g in bench_group_returns.columns]

    if len(groups) < 2:
        return {"ic_proxy": None, "t_stat": None, "n_periods": 0, "is_proxy": True}

    aw = active_weights[groups]

    # If active weights are all identical (no variation), IC is undefined
    if float(aw.std()) < 1e-12:
        return {"ic_proxy": None, "t_stat": None, "n_periods": 0, "is_proxy": True}

    correlations: list[float] = []

    for period in bench_group_returns.index:
        if period not in bench_total_returns.index:
            continue

        rb_g = bench_group_returns.loc[period, groups]
        rb_t = float(bench_total_returns[period])

        group_active_ret = rb_g - rb_t

        if float(group_active_ret.std()) < 1e-12:
            # All groups had identical benchmark-relative return; skip period
            continue

        corr = float(np.corrcoef(aw.values, group_active_ret.values)[0, 1])
        if not np.isnan(corr):
            correlations.append(corr)

    n = len(correlations)
    if n == 0:
        return {"ic_proxy": None, "t_stat": None, "n_periods": 0, "is_proxy": True}

    ic = float(np.mean(correlations))

    # t-stat: t = IC * sqrt(n) / sqrt(1 - IC^2)
    # Uses n periods as the sample for the mean IC
    if abs(ic) < 1.0 and n > 1:
        t_stat = ic * np.sqrt(n) / np.sqrt(max(1.0 - ic**2, 1e-12))
    else:
        t_stat = None

    return {"ic_proxy": ic, "t_stat": t_stat, "n_periods": n, "is_proxy": True}


# ─── Interpretation text generator ───────────────────────────────────────────

def generate_interpretation(
    attribution_df: pd.DataFrame,
    cumulative_df: pd.DataFrame,
    validation: dict,
    benchmark_label: str = "Simplified Equal-Weight Benchmark",
    ic_result: Optional[dict] = None,
) -> str:
    """
    Generate plain-English institutional commentary on attribution results.
    """
    # Cumulative totals (final row of cumulative_df)
    final = cumulative_df.iloc[-1]
    cum_alloc  = float(final.get("Cum Alloc", 0.0))
    cum_sel    = float(final.get("Cum Select", 0.0))
    cum_inter  = float(final.get("Cum Inter", 0.0))
    cum_total  = float(final.get("Cum Total", 0.0))

    # Group contributions (sum over all periods)
    group_totals = attribution_df.groupby("Group")["Total Effect"].sum().sort_values(ascending=False)

    sign = lambda x: "positive" if x > 0 else "negative" if x < 0 else "negligible"
    pct  = lambda x: f"{x*100:+.2f}%"

    lines = []
    lines.append(f"**Benchmark**: {benchmark_label} (simplified — not an institutional index).\n")
    lines.append(f"Over the full period, cumulative active return was {pct(cum_total)}.")
    lines.append(
        f"Of this, allocation contributed {pct(cum_alloc)}, "
        f"selection contributed {pct(cum_sel)}, "
        f"and interaction contributed {pct(cum_inter)}."
    )

    # Dominant effect
    abs_vals = {"allocation": abs(cum_alloc), "selection": abs(cum_sel), "interaction": abs(cum_inter)}
    dominant = max(abs_vals, key=abs_vals.get)
    lines.append(
        f"The dominant driver was **{dominant}** — meaning that "
        + {
            "allocation": "the decision of how much capital to allocate to each asset class (relative to benchmark) was the primary source of active return.",
            "selection":  "within-group asset selection relative to the benchmark was the primary driver.",
            "interaction":"the combined effect of over/underweighting groups where selection also differed drove the result.",
        }[dominant]
    )

    # Allocation insight (BF context)
    if cum_alloc > 1e-4:
        lines.append(
            "Positive allocation effect indicates the portfolio was generally overweight "
            "groups that outperformed the benchmark and/or underweight groups that lagged."
        )
    elif cum_alloc < -1e-4:
        lines.append(
            "Negative allocation effect indicates the portfolio was overweight "
            "groups that underperformed the benchmark."
        )

    # Selection insight
    if cum_sel > 1e-4:
        lines.append(
            "Positive selection effect indicates that within each asset class, "
            "the chosen assets outperformed the benchmark's holdings of that class."
        )
    elif cum_sel < -1e-4:
        lines.append(
            "Negative selection effect indicates that within-group asset selection detracted from returns."
        )

    # Top / bottom groups
    if len(group_totals) >= 2:
        top_group   = group_totals.index[0]
        top_val     = group_totals.iloc[0]
        bot_group   = group_totals.index[-1]
        bot_val     = group_totals.iloc[-1]
        lines.append(
            f"By group, **{top_group}** was the largest contributor ({pct(top_val)}) "
            f"and **{bot_group}** was the largest detractor ({pct(bot_val)})."
        )

    # Interaction
    if abs(cum_inter) < 0.0005:
        lines.append("Interaction effects were immaterial.")
    elif cum_inter > 0:
        lines.append(
            "Positive interaction effects indicate the portfolio tended to overweight "
            "groups where it also outperformed on a security basis — a favourable alignment."
        )
    else:
        lines.append(
            "Negative interaction effects indicate the portfolio overweighted groups "
            "where within-group selection disappointed."
        )

    # IC proxy
    if ic_result and ic_result.get("ic_proxy") is not None:
        ic_val = ic_result["ic_proxy"]
        lines.append(
            f"\n**Allocation Effectiveness Proxy (not true IC)**: {ic_val:.3f}. "
            + ("This suggests group overweights were on average aligned with subsequent outperformance."
               if ic_val > 0.1 else
               "This suggests limited systematic alignment between overweights and subsequent group performance."
               if abs(ic_val) <= 0.1 else
               "This suggests group overweights were on average misaligned with subsequent group returns.")
        )

    # Reconciliation
    if validation["pass"]:
        lines.append(
            f"\n**Reconciliation**: PASS — attribution effects sum to active return "
            f"within {validation['max_residual']:.2e} for all {validation['n_periods']} periods."
        )
    else:
        lines.append(
            f"\n**Reconciliation**: WARNING — max residual {validation['max_residual']:.2e} "
            f"exceeds tolerance. Check input weight normalisation."
        )

    return "\n\n".join(lines)
