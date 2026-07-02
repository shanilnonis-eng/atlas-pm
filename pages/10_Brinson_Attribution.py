"""
Page 10 — Brinson-Hood-Beebower Performance Attribution

Decomposes active portfolio return into allocation, selection, and interaction
effects relative to a simplified benchmark.

IMPORTANT: The benchmark here is constructed from the same asset universe with
alternative weights (equal-weight or 60/40 group-weighted). It is NOT an
institutional index like MSCI World or a 60/40 blend of live market indices.
This is labelled clearly throughout the page.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Streamlit caches imported library modules across page navigations.
# Force-evict the analytics module so every page load gets a fresh import.
for _mod in [k for k in sys.modules if k.startswith("analytics")]:
    del sys.modules[_mod]

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.brinson_attribution import (
    DEFAULT_CLASSIFICATION,
    BENCHMARK_60_40_GROUP_WEIGHTS,
    normalize_asset_name,
    align_asset_names,
    calculate_group_weights,
    build_benchmark_weights,
    calculate_group_returns,
    calculate_brinson_attribution,
    calculate_period_active_return,
    calculate_cumulative_attribution,
    validate_brinson_reconciliation,
    calculate_ic_proxy,
    generate_interpretation,
)
from ui.components.metrics import fmt_pct, fmt_ratio
from config.settings import ACCENT_COLOR

# ─── Colour palette consistent with rest of app ───────────────────────────────
C_ALLOC  = "#1a3a5c"   # dark navy
C_SELECT = "#2e86ab"   # mid blue
C_INTER  = "#f4a261"   # orange
C_ACTIVE = "#e84855"   # red
C_BENCH  = "#57cc99"   # green
LAYOUT = dict(
    font=dict(family="Inter, Arial, sans-serif", size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1),
)

# ─── Guard ────────────────────────────────────────────────────────────────────

st.title("Brinson-Hood-Beebower Attribution")
st.markdown("---")

_missing = [k for k in ("portfolio_returns", "simple_returns", "current_weights")
            if k not in st.session_state]
if _missing:
    st.warning(
        "Please complete **Portfolio Construction** first — then return here.\n\n"
        "Steps: **Universe & Data** → **Portfolio Construction** (click Optimise) → this page.",
        icon="⚠️",
    )
    st.stop()

# ─── Pull data from session ───────────────────────────────────────────────────

asset_returns    = st.session_state["simple_returns"]        # daily, all assets
current_weights  = st.session_state["current_weights"]       # pd.Series or dict {label: weight}
bench_daily      = st.session_state["bench_returns"]         # daily SPY
model_label      = st.session_state.get("current_model", "Portfolio")

# ─── Align asset names (handles hidden unicode / whitespace mismatches) ───────
alignment = align_asset_names(
    weight_source    = current_weights,
    return_columns   = asset_returns.columns,
    classification   = DEFAULT_CLASSIFICATION,
)
dbg = alignment["debug"]

# Debug expander — always available, even on success
with st.expander("Asset name diagnostic (expand if page fails to load)", expanded=False):
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Raw weight keys** (`repr` — shows hidden chars)")
        for k in dbg["repr_weight_keys"]:
            st.code(k, language=None)
    with col_d2:
        st.markdown("**Raw return columns** (`repr`)")
        for k in dbg["repr_ret_cols"]:
            st.code(k, language=None)
    st.markdown(f"**Normalised overlap**: {dbg['n_common']} / {dbg['n_weight_keys']} weight keys "
                f"match {dbg['n_ret_cols']} return columns")
    if dbg["only_in_weights"]:
        st.warning(f"In weights but not returns: {dbg['only_in_weights']}")
    if dbg["only_in_returns"]:
        st.info(f"In returns but not weights (unallocated): {dbg['only_in_returns']}")
    if dbg["common_assets"]:
        st.success(f"Matched assets ({len(dbg['common_assets'])}): {dbg['common_assets']}")

    # Date / period diagnostic
    if not asset_returns.empty:
        st.markdown(f"**Return data period**: {asset_returns.index[0].date()} → {asset_returns.index[-1].date()} "
                    f"({len(asset_returns)} daily obs)")

# Hard stop if truly no overlap (after normalization)
if not alignment["available_assets"]:
    st.error(
        "**No asset name overlap after normalization.**\n\n"
        "Normalised weight keys and return columns share zero common names. "
        "Go to **Portfolio Construction**, click **Optimise Portfolio**, "
        "then return here. Use the diagnostic expander above to investigate.",
    )
    st.stop()

available_assets = alignment["available_assets"]
port_asset_w     = alignment["port_weights"]       # already normalised & sum-to-1
classification   = alignment["canonical_classification"]

# Warn if some assets were dropped
n_dropped = dbg["n_weight_keys"] - dbg["n_common"]
if n_dropped > 0:
    st.warning(
        f"{n_dropped} asset(s) in portfolio weights not found in return data after normalisation "
        f"and were dropped. Remaining weights have been renormalised to sum to 1. "
        f"Missing: {dbg['only_in_weights']}",
        icon="⚠️",
    )

asset_rets = asset_returns[available_assets]

# ─── Sidebar configuration ────────────────────────────────────────────────────

with st.sidebar:
    st.header("Attribution Settings")

    method_label = st.radio(
        "Attribution method",
        ["Brinson-Fachler (recommended)", "Brinson-Hood-Beebower (original)"],
        index=0,
        help=(
            "**Brinson-Fachler**: allocation effect is relative to benchmark total return. "
            "More intuitive for relative attribution — preferred by practitioners.\n\n"
            "**BHB (1986 paper)**: allocation effect is relative to zero. "
            "Allocation for a group can be positive even if that group underperformed "
            "the benchmark total, which can be misleading."
        ),
    )
    method = "brinson_fachler" if "Fachler" in method_label else "bhb"

    benchmark_choice = st.selectbox(
        "Benchmark weights",
        ["Equal Weight (1/N)", "60/40 Group-Weighted", ],
        index=0,
        help=(
            "**Equal Weight**: every asset in your universe gets weight 1/N. "
            "Simple and transparent.\n\n"
            "**60/40 Group-Weighted**: 60% Equities, 30% Fixed Income, "
            "5% Alternatives, 5% Cash — equal-weight within each group. "
            "A conventional institutional benchmark proxy."
        ),
    )
    bench_method = "equal_weight" if "Equal" in benchmark_choice else "group_weighted"

    if bench_method == "group_weighted":
        st.caption(
            "Group weights: Equities 60%, Fixed Income 30%, Alternatives 5%, Cash 5%"
        )

# ─── Build benchmark weights ─────────────────────────────────────────────────
# Use `classification` (already normalised via align_asset_names) for benchmark
# construction so the keys are guaranteed to match `available_assets`.

bench_asset_w = build_benchmark_weights(
    available_assets,
    classification,   # normalised already
    method=bench_method,
    custom_group_weights=BENCHMARK_60_40_GROUP_WEIGHTS if bench_method == "group_weighted" else None,
)

bench_label = (
    f"Equal-Weight Benchmark ({len(available_assets)} assets)"
    if bench_method == "equal_weight"
    else "60/40 Group-Weighted Benchmark"
)

# classification already set by align_asset_names above
groups_present = sorted(set(classification.values()))

# ─── Group weights ────────────────────────────────────────────────────────────

port_gw  = calculate_group_weights(port_asset_w,  classification)
bench_gw = calculate_group_weights(bench_asset_w, classification)

# ─── Group returns (monthly compounding) ─────────────────────────────────────

with st.spinner("Computing group returns…"):
    port_group_ret, bench_group_ret = calculate_group_returns(
        asset_rets, port_asset_w, bench_asset_w, classification, freq="ME"
    )

if port_group_ret.empty or len(port_group_ret) < 2:
    st.error("Insufficient data: need at least 2 monthly periods for attribution.")
    st.stop()

# ─── Core attribution ────────────────────────────────────────────────────────

attr_df = calculate_brinson_attribution(
    port_gw, bench_gw, port_group_ret, bench_group_ret, method=method
)
active_returns = calculate_period_active_return(
    port_gw, bench_gw, port_group_ret, bench_group_ret
)
cum_df     = calculate_cumulative_attribution(attr_df)
validation = validate_brinson_reconciliation(attr_df, active_returns)

# Benchmark total returns (for IC proxy)
bench_total_monthly = (bench_group_ret * bench_gw[bench_group_ret.columns]).sum(axis=1)

# Active weights (constant because portfolio is static)
active_weights = port_gw - bench_gw

ic_result = calculate_ic_proxy(active_weights, bench_group_ret, bench_total_monthly)
port_total_monthly = (port_group_ret * port_gw[port_group_ret.columns]).sum(axis=1)

# ─── Page header  ─────────────────────────────────────────────────────────────

col_banner, col_method = st.columns([2, 1])
with col_banner:
    st.subheader(f"Brinson Attribution — {model_label}")
    st.markdown(
        f"**Benchmark**: {bench_label} — *simplified, not an institutional index*\n\n"
        f"**Method**: {method_label}\n\n"
        f"**Periods**: {len(port_group_ret)} monthly | "
        f"{port_group_ret.index[0].strftime('%b %Y')} – {port_group_ret.index[-1].strftime('%b %Y')}"
    )

with col_method:
    if validation["pass"]:
        st.success(
            f"Reconciliation: PASS\n\nMax residual: {validation['max_residual']:.2e}",
            icon="✅",
        )
    else:
        st.error(
            f"Reconciliation: FAIL\n\nMax residual: {validation['max_residual']:.2e}",
            icon="⚠️",
        )

with st.expander("Methodology — what do these three effects mean?", expanded=False):
    st.markdown(f"""
**Attribution method**: {method_label}

For each group *g* and period *t*:

| Effect | Formula (Brinson-Fachler) | Interpretation |
|--------|--------------------------|----------------|
| **Allocation** | (w_p,g − w_b,g) × (r_b,g − r_b) | Did overweighting/underweighting groups relative to benchmark add value? |
| **Selection** | w_b,g × (r_p,g − r_b,g) | Did asset selection *within* each group beat the equivalent benchmark group? |
| **Interaction** | (w_p,g − w_b,g) × (r_p,g − r_b,g) | Combined effect of active weight AND active within-group return |

**Reconciliation identity** (holds exactly for both methods):

> Σ_g [Allocation + Selection + Interaction] = Portfolio Return − Benchmark Return

**BHB vs BF**: Only the allocation formula differs.
- BHB: `(w_p,g − w_b,g) × r_b,g` — allocation is positive if you overweight a group with positive absolute return
- BF:  `(w_p,g − w_b,g) × (r_b,g − r_b)` — allocation is positive only if you overweight a group that *beats the benchmark total*

BF is the standard for relative performance attribution. BHB can produce positive allocation
for groups that underperformed the benchmark total, which is counterintuitive.

**Cumulation**: arithmetic sum across periods (standard for reporting). For horizons
beyond 2–3 years, geometric linking (Carino/Menchero) is more precise but not implemented here.
    """)

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_period, tab_summary, tab_chart, tab_recon, tab_ic = st.tabs([
    "Period Attribution",
    "Summary by Group",
    "Cumulative Chart",
    "Reconciliation",
    "Allocation Effectiveness",
])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — Period-level attribution table
# ───────────────────────────────────────────────────────────────────────────────

with tab_period:
    st.subheader("Per-Period Attribution Table")
    st.caption(
        "Each row is one (month, group) observation. "
        "Totals per period sum to active return for that period."
    )

    # Period selector
    all_periods = sorted(attr_df["Period"].unique())
    period_options = ["All periods"] + [p.strftime("%b %Y") for p in all_periods]
    selected_period_str = st.selectbox("Filter by period", period_options, index=0)

    if selected_period_str == "All periods":
        display_attr = attr_df.copy()
    else:
        mask = attr_df["Period"].dt.strftime("%b %Y") == selected_period_str
        display_attr = attr_df[mask].copy()

    display_attr["Period"] = display_attr["Period"].dt.strftime("%b %Y")

    # Format as percentages for display
    pct_cols = [
        "Port Weight", "Bench Weight", "Active Weight",
        "Port Return", "Bench Return", "Bench Total",
        "Alloc Effect", "Select Effect", "Inter Effect", "Total Effect",
    ]
    fmt_df = display_attr.copy()
    for c in pct_cols:
        if c in fmt_df.columns:
            fmt_df[c] = fmt_df[c].map(lambda v: f"{v*100:+.3f}%")

    st.dataframe(fmt_df, use_container_width=True, hide_index=True)

    # Period total row
    if selected_period_str != "All periods":
        period_total = attr_df[attr_df["Period"].dt.strftime("%b %Y") == selected_period_str][
            ["Alloc Effect", "Select Effect", "Inter Effect", "Total Effect"]
        ].sum()
        act_ret = active_returns[
            active_returns.index.strftime("%b %Y") == selected_period_str
        ].iloc[0] if len(active_returns[active_returns.index.strftime("%b %Y") == selected_period_str]) else float("nan")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Alloc (period)", fmt_pct(period_total["Alloc Effect"]))
        c2.metric("Select (period)", fmt_pct(period_total["Select Effect"]))
        c3.metric("Inter (period)", fmt_pct(period_total["Inter Effect"]))
        c4.metric("Total (period)", fmt_pct(period_total["Total Effect"]))
        c5.metric("Active Return", fmt_pct(act_ret))


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Summary by group
# ───────────────────────────────────────────────────────────────────────────────

with tab_summary:
    st.subheader("Cumulative Attribution by Asset Class")
    st.caption(
        "Arithmetic sum of each effect across all periods.  "
        "Represents total contribution of each group over the full history."
    )

    group_summary = attr_df.groupby("Group").agg(
        Port_Weight=("Port Weight", "first"),
        Bench_Weight=("Bench Weight", "first"),
        Active_Weight=("Active Weight", "first"),
        Cum_Alloc=("Alloc Effect", "sum"),
        Cum_Select=("Select Effect", "sum"),
        Cum_Inter=("Inter Effect", "sum"),
        Cum_Total=("Total Effect", "sum"),
    ).reset_index()

    group_summary = group_summary.sort_values("Cum_Total", ascending=False)

    # Display version
    fmt_sum = group_summary.copy()
    for c in ["Port_Weight", "Bench_Weight", "Active_Weight",
              "Cum_Alloc", "Cum_Select", "Cum_Inter", "Cum_Total"]:
        fmt_sum[c] = fmt_sum[c].map(lambda v: f"{v*100:+.3f}%" if c.startswith("Cum") or c.endswith("Weight")
                                    else f"{v*100:.2f}%")
    fmt_sum.columns = [
        "Group", "Port Wt", "Bench Wt", "Active Wt",
        "Cum Alloc", "Cum Select", "Cum Interaction", "Cum Total",
    ]
    st.dataframe(fmt_sum, use_container_width=True, hide_index=True)

    # Stacked bar chart by group
    fig_bar = go.Figure()
    for col, label, color in [
        ("Cum_Alloc",  "Allocation",   C_ALLOC),
        ("Cum_Select", "Selection",    C_SELECT),
        ("Cum_Inter",  "Interaction",  C_INTER),
    ]:
        fig_bar.add_trace(go.Bar(
            name=label,
            x=group_summary["Group"],
            y=group_summary[col] * 100,
            marker_color=color,
            hovertemplate=f"{label}: %{{y:+.3f}}%<extra>%{{x}}</extra>",
        ))

    fig_bar.add_hline(y=0, line_color="grey", line_dash="dot")
    fig_bar.update_layout(
        **LAYOUT,
        title="Cumulative Attribution Effects by Asset Class",
        barmode="relative",
        yaxis_title="Cumulative Effect (%)",
        yaxis_ticksuffix="%",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Summary totals
    st.markdown("### Overall Totals")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumulative Allocation", fmt_pct(float(attr_df["Alloc Effect"].sum())))
    c2.metric("Cumulative Selection",  fmt_pct(float(attr_df["Select Effect"].sum())))
    c3.metric("Cumulative Interaction", fmt_pct(float(attr_df["Inter Effect"].sum())))
    c4.metric("Cumulative Active Return", fmt_pct(float(active_returns.sum())))


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — Cumulative attribution chart
# ───────────────────────────────────────────────────────────────────────────────

with tab_chart:
    st.subheader("Cumulative Attribution Effects Over Time")
    st.caption(
        "Arithmetic cumulation of monthly effects. "
        "Compare 'Cumulative Total Effect' with 'Cumulative Active Return' — "
        "they should track identically (reconciliation check)."
    )

    fig_cum = go.Figure()

    fig_cum.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df["Cum Alloc"] * 100,
        name="Allocation", line=dict(color=C_ALLOC, width=2),
        hovertemplate="Cum Alloc: %{y:+.3f}%<extra></extra>",
    ))
    fig_cum.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df["Cum Select"] * 100,
        name="Selection", line=dict(color=C_SELECT, width=2),
        hovertemplate="Cum Select: %{y:+.3f}%<extra></extra>",
    ))
    fig_cum.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df["Cum Inter"] * 100,
        name="Interaction", line=dict(color=C_INTER, width=2),
        hovertemplate="Cum Inter: %{y:+.3f}%<extra></extra>",
    ))
    fig_cum.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df["Cum Total"] * 100,
        name="Total Attribution", line=dict(color=C_ACTIVE, width=2.5, dash="dash"),
        hovertemplate="Cum Total: %{y:+.3f}%<extra></extra>",
    ))
    # Actual monthly portfolio and benchmark for reference
    fig_cum.add_trace(go.Scatter(
        x=active_returns.index,
        y=active_returns.cumsum() * 100,
        name="Active Return (cumsum)", line=dict(color="grey", width=1.5, dash="dot"),
        hovertemplate="Cum Active: %{y:+.3f}%<extra></extra>",
    ))

    fig_cum.add_hline(y=0, line_color="lightgrey", line_dash="dot")
    fig_cum.update_layout(
        **LAYOUT,
        title="Cumulative Attribution Effects vs Cumulative Active Return",
        yaxis_title="Cumulative Effect (%)",
        yaxis_ticksuffix="%",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Monthly period bar (period-by-period effects)
    st.subheader("Monthly Attribution Effects")
    period_totals = attr_df.groupby("Period")[
        ["Alloc Effect", "Select Effect", "Inter Effect"]
    ].sum()

    fig_monthly = go.Figure()
    for col, label, color in [
        ("Alloc Effect",  "Allocation",  C_ALLOC),
        ("Select Effect", "Selection",   C_SELECT),
        ("Inter Effect",  "Interaction", C_INTER),
    ]:
        fig_monthly.add_trace(go.Bar(
            x=period_totals.index,
            y=period_totals[col] * 100,
            name=label, marker_color=color,
            hovertemplate=f"{label}: %{{y:+.3f}}%<extra>%{{x|%b %Y}}</extra>",
        ))
    fig_monthly.add_hline(y=0, line_color="grey", line_dash="dot")
    fig_monthly.update_layout(
        **LAYOUT,
        title="Monthly Attribution Effects (stacked)",
        barmode="relative",
        yaxis_title="Effect (%)",
        yaxis_ticksuffix="%",
        xaxis_title="Period",
    )
    st.plotly_chart(fig_monthly, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — Active return reconciliation
# ───────────────────────────────────────────────────────────────────────────────

with tab_recon:
    st.subheader("Active Return Reconciliation")

    # Key principle explanation
    st.info(
        "**Reconciliation check**: for each period, "
        "Allocation + Selection + Interaction must equal Portfolio Return − Benchmark Return.\n\n"
        "This holds analytically when portfolio and benchmark weight vectors each sum to 1. "
        "Any residual indicates floating-point drift or a weight normalisation issue.",
        icon="📐",
    )

    col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
    col_r1.metric("Portfolio Return (attribution)", fmt_pct(float(port_total_monthly.sum())))
    col_r2.metric("Benchmark Return (attribution)", fmt_pct(float(bench_total_monthly.sum())))
    col_r3.metric("Active Return", fmt_pct(float(active_returns.sum())))
    col_r4.metric("Total Attribution", fmt_pct(float(attr_df["Total Effect"].sum())))
    residual = float(attr_df["Total Effect"].sum()) - float(active_returns.sum())
    col_r5.metric("Residual", f"{residual*100:+.2e}%")

    # Reconciliation status
    if validation["pass"]:
        st.success(
            f"**Reconciliation PASSED** across all {validation['n_periods']} periods. "
            f"Maximum per-period residual: {validation['max_residual']:.2e}",
            icon="✅",
        )
    else:
        st.error(
            f"**Reconciliation WARNING** — {len(validation['periods_failing'])} period(s) "
            f"have residual > 1e-8. Max residual: {validation['max_residual']:.2e}",
            icon="⚠️",
        )

    # Per-period reconciliation chart
    period_totals_series = attr_df.groupby("Period")["Total Effect"].sum()
    residuals = validation["residuals"]

    fig_recon = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
    )

    fig_recon.add_trace(go.Scatter(
        x=port_total_monthly.index, y=port_total_monthly * 100,
        name="Portfolio (attribution)", line=dict(color=C_ALLOC, width=2),
    ), row=1, col=1)
    fig_recon.add_trace(go.Scatter(
        x=bench_total_monthly.index, y=bench_total_monthly * 100,
        name="Benchmark (attribution)", line=dict(color=C_BENCH, width=2, dash="dash"),
    ), row=1, col=1)
    fig_recon.add_trace(go.Scatter(
        x=active_returns.index, y=active_returns * 100,
        name="Active Return", line=dict(color=C_ACTIVE, width=1.5, dash="dot"),
    ), row=1, col=1)

    fig_recon.add_trace(go.Bar(
        x=residuals.index, y=residuals * 1e8,
        name="Residual (×1e-8)",
        marker_color=["#e84855" if v < 0 else "#1a3a5c" for v in residuals.values],
        hovertemplate="Residual: %{y:.2e} × 1e-8<extra>%{x|%b %Y}</extra>",
    ), row=2, col=1)

    fig_recon.update_layout(
        **LAYOUT,
        title="Monthly Portfolio vs Benchmark Return (attribution-consistent)",
        height=500,
    )
    fig_recon.update_yaxes(title_text="Monthly Return (%)", ticksuffix="%", row=1, col=1)
    fig_recon.update_yaxes(title_text="Residual (×1e-8)", row=2, col=1)
    st.plotly_chart(fig_recon, use_container_width=True)

    st.caption(
        "**Note**: 'Portfolio (attribution)' and 'Benchmark (attribution)' are computed "
        "as Σ_g(w_g × r_g,t) using static weights and compounded monthly asset returns. "
        "This differs slightly from the actual portfolio return (compounded daily) in "
        "session state, because daily rebalancing approximation differs from "
        "beginning-of-period static weight assumption used in Brinson attribution."
    )

    # Weight table
    with st.expander("Benchmark weight comparison", expanded=False):
        weight_rows = []
        for a in available_assets:
            g = classification.get(a, "Other")
            weight_rows.append({
                "Asset":           a,
                "Group":           g,
                "Portfolio Weight": f"{port_asset_w.get(a, 0)*100:.2f}%",
                "Benchmark Weight": f"{bench_asset_w.get(a, 0)*100:.2f}%",
                "Active Weight":    f"{(port_asset_w.get(a, 0) - bench_asset_w.get(a, 0))*100:+.2f}%",
            })
        st.dataframe(pd.DataFrame(weight_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Benchmark: {bench_label}. "
            "This is a simplified benchmark — not an institutional market index."
        )


# ───────────────────────────────────────────────────────────────────────────────
# TAB 5 — Allocation effectiveness / IC proxy
# ───────────────────────────────────────────────────────────────────────────────

with tab_ic:
    st.subheader("Allocation Effectiveness Proxy")

    st.warning(
        "**This is NOT the true Information Coefficient.**\n\n"
        "True IC requires analyst forecast scores for each asset. "
        "No forecast scores exist in this system.\n\n"
        "This proxy measures: *on average, do the groups we overweight (vs benchmark) "
        "subsequently outperform the benchmark total?* "
        "This is a reasonable proxy for group-level allocation skill, "
        "but it is NOT interchangeable with IC from a forecasting model. "
        "Do not present this as IC in an interview without this caveat.",
        icon="⚠️",
    )

    if ic_result["ic_proxy"] is not None:
        ic_val  = ic_result["ic_proxy"]
        t_val   = ic_result.get("t_stat")
        n_per   = ic_result["n_periods"]

        col_ic1, col_ic2, col_ic3 = st.columns(3)
        col_ic1.metric(
            "Allocation Effectiveness Proxy",
            f"{ic_val:.3f}",
            help="Mean cross-sectional Pearson correlation between active weights and "
                 "subsequent benchmark-relative group returns.",
        )
        if t_val is not None:
            col_ic2.metric(
                "t-statistic (H₀: IC=0)",
                f"{t_val:.2f}",
                help="Under H₀: no allocation effectiveness. |t| > 2 suggests significance at ~5%.",
            )
        col_ic3.metric("Periods used", n_per)

        if ic_val > 0.1:
            st.success(
                f"IC proxy = {ic_val:.3f}: overweighted groups tended to outperform the benchmark. "
                "Suggests the group allocation decisions had some systematic alignment with outcomes.",
                icon="🟢",
            )
        elif ic_val < -0.1:
            st.error(
                f"IC proxy = {ic_val:.3f}: overweighted groups tended to underperform. "
                "Group allocation decisions were on average negatively aligned with returns.",
                icon="🔴",
            )
        else:
            st.info(
                f"IC proxy = {ic_val:.3f}: no strong systematic relationship between "
                "active weights and subsequent group performance.",
                icon="⚪",
            )

        # Chart: per-period correlation
        groups_for_ic = [g for g in active_weights.index if g in bench_group_ret.columns]
        aw_vec = active_weights[groups_for_ic]

        period_ic = []
        for period in bench_group_ret.index:
            if period not in bench_total_monthly.index:
                continue
            rb_g   = bench_group_ret.loc[period, groups_for_ic]
            rb_t   = float(bench_total_monthly[period])
            active_ret = rb_g - rb_t
            if active_ret.std() < 1e-12 or aw_vec.std() < 1e-12:
                continue
            corr = float(np.corrcoef(aw_vec.values, active_ret.values)[0, 1])
            period_ic.append({"Period": period, "IC Proxy": corr})

        if period_ic:
            ic_ts = pd.DataFrame(period_ic).set_index("Period")
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Bar(
                x=ic_ts.index, y=ic_ts["IC Proxy"],
                marker_color=["#57cc99" if v >= 0 else "#e84855" for v in ic_ts["IC Proxy"]],
                hovertemplate="IC Proxy: %{y:.3f}<extra>%{x|%b %Y}</extra>",
                name="Monthly IC Proxy",
            ))
            fig_ic.add_hline(y=0, line_color="grey", line_dash="dot")
            if ic_val is not None:
                fig_ic.add_hline(
                    y=ic_val, line_color="#1a3a5c", line_dash="dash",
                    annotation_text=f"Mean = {ic_val:.3f}", annotation_position="top right",
                )
            fig_ic.update_layout(
                **LAYOUT,
                title="Allocation Effectiveness Proxy — Per Period",
                yaxis_title="Correlation (active wts vs group active returns)",
            )
            st.plotly_chart(fig_ic, use_container_width=True)

        st.caption(
            "**Formula**: for each period, Pearson correlation between "
            "the vector of group active weights and the vector of benchmark-relative "
            "group returns (r_b,g,t − r_b,t). Mean is reported as the IC proxy."
        )
    else:
        st.info(
            "IC proxy could not be computed. This can happen when: "
            "(1) active weights are all zero (portfolio = benchmark), "
            "or (2) all groups earn the same benchmark-relative return every period.",
            icon="ℹ️",
        )

    # Active weight display
    with st.expander("Group active weights (constant — static portfolio)", expanded=False):
        aw_df = pd.DataFrame({
            "Group":          list(active_weights.index),
            "Portfolio Wt":   [f"{v*100:.2f}%" for v in port_gw[active_weights.index]],
            "Benchmark Wt":   [f"{v*100:.2f}%" for v in bench_gw[active_weights.index]],
            "Active Wt":      [f"{v*100:+.2f}%" for v in active_weights],
        })
        st.dataframe(aw_df, use_container_width=True, hide_index=True)
        st.caption(
            "Since portfolio weights are static (no rebalancing), "
            "active weights are constant over the full period."
        )


# ─── Interpretation ───────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Institutional Commentary")

interpretation = generate_interpretation(
    attr_df, cum_df, validation,
    benchmark_label=bench_label,
    ic_result=ic_result,
)
st.markdown(interpretation)

st.caption(
    "**Limitations**: (1) Benchmark is simplified and not investable. "
    "(2) Portfolio weights are static — no rebalancing modelled within the attribution period. "
    "(3) Arithmetic cumulation is an approximation for long horizons. "
    "(4) Attribution is at group (asset class) level, not individual security level. "
    "(5) Allocation effectiveness proxy is not a true IC — no forecast scores available."
)
