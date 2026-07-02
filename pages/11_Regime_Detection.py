"""
Page 11 — Regime Detection

Fits a two-state Gaussian Hidden Markov Model (HMM) to returns to
systematically identify market regimes.

Key outputs:
  - Labelled regime timeline (Low Vol / Bull vs High Vol / Bear)
  - Conditional performance: Sharpe, drawdown, win rate per regime
  - Transition probability matrix (regime persistence & switching)
  - Return distributions per regime
  - Conditional beta & alpha vs benchmark per regime
  - Regime duration statistics

Why this matters:
  Momentum and trend strategies perform very differently across regimes.
  Manually flagging regimes is time-consuming and inconsistent.
  This gives a reproducible, data-driven regime signal.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.regime_detection import (
    REGIME_NAMES,
    REGIME_COLORS,
    REGIME_LINE_COLORS,
    MIN_REQUIRED_OBSERVATIONS,
    fit_hmm,
    label_regimes,
    regime_statistics,
    regime_beta_alpha,
    transition_matrix,
    regime_durations,
    regime_emission_params,
    contiguous_blocks,
    _HMMLEARN_AVAILABLE,
)
from analytics.returns import drawdown_series
from ui.components.metrics import fmt_pct, fmt_ratio

# ── constants ────────────────────────────────────────────────────────────────

PALETTE = ["#2e86ab", "#e84855", "#f4a261", "#9b5de5", "#57cc99"]
LAYOUT  = dict(
    font=dict(family="Inter, Arial, sans-serif", size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#e0e0e0", borderwidth=1),
)

st.title("Regime Detection")
st.markdown(
    "A two-state Hidden Markov Model fitted to return data to "
    "identify **Low Vol (Bull)** and **High Vol (Bear)** market regimes — "
    "and measure how your portfolio behaves in each."
)
st.markdown("---")

# ── dependency check ─────────────────────────────────────────────────────────

if not _HMMLEARN_AVAILABLE:
    st.error(
        "**hmmlearn is not installed.**\n\n"
        "Run `pip install hmmlearn` in your terminal, then restart Streamlit.",
        icon="🔧",
    )
    st.stop()

# ── guard: require data ──────────────────────────────────────────────────────

if "simple_returns" not in st.session_state:
    st.warning("Please load data first on the **Universe & Data** page.", icon="⚠️")
    st.stop()

returns        = st.session_state["simple_returns"]
bench_returns  = st.session_state.get("bench_returns")
rf_returns     = st.session_state.get("rf_returns")
bench_label    = st.session_state.get("benchmark_label", "Benchmark")
port_returns   = st.session_state.get("portfolio_returns")
port_model_lbl = st.session_state.get("current_model", "Portfolio")

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Regime Model")

    # Signal source
    signal_options = []
    if port_returns is not None:
        signal_options.append(f"Portfolio ({port_model_lbl})")
    if bench_returns is not None:
        signal_options.append(bench_label)
    # Always offer individual assets
    for ticker in returns.columns[:10]:
        signal_options.append(f"Asset: {ticker}")

    if not signal_options:
        st.warning("No return series available.")
        st.stop()

    signal_source = st.selectbox(
        "Fit HMM on",
        signal_options,
        help=(
            "The HMM learns regime structure from this series. "
            "Using the benchmark detects broad market regimes; "
            "using the portfolio detects your strategy's own regimes."
        ),
    )

    st.markdown(
        "*The model always uses **2 states**.  "
        "States are automatically labelled by fitted volatility — "
        "no manual tuning required.*"
    )
    st.markdown("---")

    fit_btn = st.button("Fit Regime Model", type="primary", use_container_width=True)

    if "_regime_cache" in st.session_state:
        cache = st.session_state["_regime_cache"]
        st.caption(
            f"Last fit: **{cache['signal_source']}**  "
            f"| {cache['n_obs']:,} obs  "
            f"| LL = {cache['log_likelihood']:.1f}"
        )

# ── resolve the signal series ─────────────────────────────────────────────────

def _get_signal_series(label: str) -> pd.Series | None:
    if label.startswith("Asset: "):
        ticker = label[len("Asset: "):]
        return returns[ticker].dropna() if ticker in returns.columns else None
    if label == bench_label and bench_returns is not None:
        return bench_returns.dropna()
    if port_returns is not None and label.startswith("Portfolio"):
        return port_returns.dropna()
    return None

# ── fit the HMM ───────────────────────────────────────────────────────────────

if fit_btn:
    signal = _get_signal_series(signal_source)
    if signal is None or len(signal) < MIN_REQUIRED_OBSERVATIONS:
        st.error(
            f"Not enough data to fit the HMM on **{signal_source}** "
            f"(need at least {MIN_REQUIRED_OBSERVATIONS} observations).",
            icon="⚠️",
        )
    else:
        with st.spinner("Fitting Hidden Markov Model…"):
            try:
                model, state_seq, ll = fit_hmm(signal, n_states=2, n_restarts=15)
                labels = label_regimes(model, state_seq, signal)

                st.session_state["_regime_cache"] = {
                    "signal_source":  signal_source,
                    "signal":         signal,
                    "model":          model,
                    "labels":         labels,
                    "log_likelihood": ll,
                    "n_obs":          len(signal),
                }
            except Exception as e:
                st.error(f"Regime model failed: {e}", icon="⚠️")
                st.stop()

# ── render results ────────────────────────────────────────────────────────────

if "_regime_cache" not in st.session_state:
    st.info("Select a signal source in the sidebar and click **Fit Regime Model** to begin.")
    st.stop()

cache       = st.session_state["_regime_cache"]
signal      = cache["signal"]
model       = cache["model"]
labels      = cache["labels"]
ll          = cache["log_likelihood"]
signal_src  = cache["signal_source"]

# ── validation banner ─────────────────────────────────────────────────────────

ep       = regime_emission_params(model)
vol_low  = ep.loc[REGIME_NAMES[0], "Ann. Volatility (implied)"]
vol_high = ep.loc[REGIME_NAMES[1], "Ann. Volatility (implied)"]
vol_ratio = vol_high / vol_low if vol_low > 1e-8 else 1.0

if vol_ratio < 1.3:
    st.warning(
        f"The two regimes have similar implied volatilities "
        f"({vol_low:.1%} vs {vol_high:.1%}, ratio = {vol_ratio:.2f}×). "
        "The HMM may not have found a meaningful separation — "
        "try a different signal or a longer data history.",
        icon="⚠️",
    )

# ── summary KPIs ──────────────────────────────────────────────────────────────

n_low  = int((labels == REGIME_NAMES[0]).sum())
n_high = int((labels == REGIME_NAMES[1]).sum())
tm     = transition_matrix(model)
p_stay_low  = float(tm.loc[REGIME_NAMES[0], REGIME_NAMES[0]])
p_stay_high = float(tm.loc[REGIME_NAMES[1], REGIME_NAMES[1]])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Low Vol (Bull) days",  f"{n_low:,}",  f"{n_low / len(signal):.0%} of sample")
k2.metric("High Vol (Bear) days", f"{n_high:,}", f"{n_high / len(signal):.0%} of sample")
k3.metric("Prob stay Low Vol",  f"{p_stay_low:.1%}",
          help="Probability of remaining in Low Vol regime tomorrow (persistence).")
k4.metric("Prob stay High Vol", f"{p_stay_high:.1%}",
          help="Probability of remaining in High Vol regime tomorrow (persistence).")

st.markdown("---")

# ── Section 1: Regime Timeline ────────────────────────────────────────────────

st.subheader("Regime Timeline")

blocks = contiguous_blocks(labels)
cum_ret = (1 + signal).cumprod()

fig_timeline = go.Figure()

# Regime shading
for regime, start, end in blocks:
    fig_timeline.add_vrect(
        x0=str(start), x1=str(end),
        fillcolor=REGIME_COLORS[regime],
        layer="below",
        line_width=0,
    )

# Cumulative return line
fig_timeline.add_trace(go.Scatter(
    x=cum_ret.index,
    y=cum_ret.values,
    name=signal_src,
    line=dict(color="#1a3a5c", width=1.8),
    hovertemplate="%{x|%Y-%m-%d}<br>Growth: %{y:.3f}<extra></extra>",
))

# Invisible traces for the legend
for regime in REGIME_NAMES:
    fig_timeline.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=12, color=REGIME_LINE_COLORS[regime], symbol="square"),
        name=regime,
        showlegend=True,
    ))

fig_timeline.update_layout(
    **LAYOUT,
    title=f"Cumulative Return — {signal_src} (shaded by regime)",
    yaxis_title="Growth of £1",
    xaxis_title="",
    height=380,
)
st.plotly_chart(fig_timeline, use_container_width=True)

# ── Section 2: Conditional Statistics ────────────────────────────────────────

st.subheader("Performance by Regime")

rf_for_stats = (rf_returns.reindex(signal.index).fillna(0)
                if rf_returns is not None else None)
stats = regime_statistics(signal, labels, rf_for_stats)

# Format for display
stats_disp = pd.DataFrame(index=stats.index)
stats_disp["Days"]          = stats["N Days"].map(lambda x: f"{x:,}")
stats_disp["% of Sample"]   = stats["Pct Sample"].map(lambda x: f"{x:.1%}")
stats_disp["Ann. Return"]   = stats["Ann. Return"].map(fmt_pct)
stats_disp["Ann. Vol"]      = stats["Ann. Volatility"].map(fmt_pct)
stats_disp["Sharpe"]        = stats["Sharpe Ratio"].map(fmt_ratio)
stats_disp["Sortino"]       = stats["Sortino Ratio"].map(fmt_ratio)
stats_disp["Max Drawdown"]  = stats["Max Drawdown"].map(fmt_pct)
stats_disp["Daily Win Rate"]= stats["Daily Win Rate"].map(lambda x: f"{x:.1%}")

st.dataframe(
    stats_disp,
    use_container_width=True,
    height=120,
)

# ── Section 3: Return Distributions ──────────────────────────────────────────

st.subheader("Return Distribution by Regime")

fig_dist = go.Figure()
for i, regime in enumerate(REGIME_NAMES):
    mask   = labels == regime
    ret_r  = signal[mask]
    fig_dist.add_trace(go.Histogram(
        x=ret_r.values * 100,
        name=regime,
        opacity=0.65,
        marker_color=REGIME_LINE_COLORS[regime],
        nbinsx=80,
        hovertemplate=f"{regime}<br>Return: %{{x:.2f}}%<br>Count: %{{y}}<extra></extra>",
    ))

fig_dist.update_layout(
    **LAYOUT,
    barmode="overlay",
    title="Daily Return Distributions by Regime",
    xaxis_title="Daily Return (%)",
    yaxis_title="Count",
    height=350,
)
st.plotly_chart(fig_dist, use_container_width=True)

# ── Section 4: Transition Matrix ─────────────────────────────────────────────

col_tm, col_dur = st.columns(2)

with col_tm:
    st.subheader("Transition Probabilities")
    st.caption("Probability of moving from one regime to another on the next day.")

    text_vals = [[f"{v:.1%}" for v in row] for row in tm.values]
    fig_tm = go.Figure(go.Heatmap(
        z=tm.values,
        x=list(tm.columns),
        y=list(tm.index),
        colorscale=[[0, "#f0f4fa"], [1, "#1a3a5c"]],
        zmin=0, zmax=1,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=16),
        hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{z:.1%}<extra></extra>",
        showscale=True,
        colorbar=dict(tickformat=".0%", len=0.7),
    ))
    fig_tm.update_layout(
        **LAYOUT,
        height=300,
        xaxis_title="To regime",
        yaxis_title="From regime",
    )
    fig_tm.update_layout(margin=dict(l=120, r=20, t=30, b=80))
    st.plotly_chart(fig_tm, use_container_width=True)

with col_dur:
    st.subheader("Regime Duration Statistics")
    st.caption("How long each regime typically persists once entered.")

    durs = regime_durations(labels)
    durs_disp = pd.DataFrame(index=durs.index)
    durs_disp["Episodes"]    = durs["N Episodes"].map(lambda x: f"{x:,}")
    durs_disp["Mean (days)"] = durs["Mean Days"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    durs_disp["Median"]      = durs["Median Days"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    durs_disp["Max (days)"]  = durs["Max Days"].map(lambda x: f"{x:,}")
    durs_disp["Min (days)"]  = durs["Min Days"].map(lambda x: f"{x:,}")
    st.dataframe(durs_disp, use_container_width=True, height=120)

    # Implied expected holding period = 1 / (1 - P_stay)
    st.markdown("")
    for regime in REGIME_NAMES:
        p_stay = float(tm.loc[regime, regime])
        if p_stay < 1.0:
            expected_days = 1.0 / (1.0 - p_stay)
            st.caption(
                f"**{regime}**: expected stay = **{expected_days:.0f} days** "
                f"({expected_days / 21:.1f} months)"
            )

# ── Section 5: Conditional Beta / Alpha ───────────────────────────────────────

st.markdown("---")
st.subheader("Conditional Beta & Alpha vs Benchmark")

if bench_returns is None:
    st.info("Benchmark returns not loaded — beta/alpha analysis unavailable.", icon="ℹ️")
else:
    ba = regime_beta_alpha(signal, bench_returns, labels)
    ba_disp = pd.DataFrame(index=ba.index)
    ba_disp["Alpha (Ann.)"]  = ba["Alpha (Ann.)"].map(fmt_pct)
    ba_disp["Beta"]          = ba["Beta"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    ba_disp["R²"]            = ba["R-squared"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    ba_disp["p (beta)"]      = ba["p-value (beta)"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—"
    )
    st.dataframe(ba_disp, use_container_width=True, height=120)
    st.caption(
        "Alpha and Beta are computed via OLS of signal returns on benchmark returns, "
        "using only dates within each regime. Alpha is annualised (daily × 252)."
    )

# ── Section 6: Model Validation ───────────────────────────────────────────────

with st.expander("Model Validation & Fitted Parameters", expanded=False):
    st.markdown(f"""
**HMM log-likelihood:** `{ll:.2f}` — higher is better; "

**Data used:** {signal_src}, {len(signal):,} observations
({signal.index[0].strftime('%Y-%m-%d')} → {signal.index[-1].strftime('%Y-%m-%d')})

**Volatility separation ratio:** {vol_ratio:.2f}×
*(target > 1.5× for meaningful regime separation)*
    """)

    st.markdown("**Fitted emission distributions (annualised)**")
    ep_disp = pd.DataFrame(index=ep.index)
    ep_disp["Daily Mean"]          = ep["Daily Mean Return"].map(lambda x: f"{x:.5f}")
    ep_disp["Daily Std"]           = ep["Daily Std Return"].map(lambda x: f"{x:.5f}")
    ep_disp["Ann. Mean Return"]    = ep["Ann. Mean Return"].map(fmt_pct)
    ep_disp["Ann. Vol (implied)"]  = ep["Ann. Volatility (implied)"].map(fmt_pct)
    st.dataframe(ep_disp, use_container_width=True, height=120)

    st.markdown("""
**How to interpret:**
- **Ann. Mean Return** — average return the model attributes to this regime
- **Ann. Vol (implied)** — the HMM's estimate of annualised volatility for this regime
  *(compare to the realised volatility in the performance table above)*
- If the two Ann. Vol numbers are very close, consider using a longer data history or a
  broader signal (e.g. a market index rather than a single-asset portfolio)

**Label-switching:** States are always labelled by fitted variance — Low Vol (Bull) always
has the lower variance state, regardless of which integer index the HMM assigned internally.
    """)

# ── Section 7: Drawdown by regime ─────────────────────────────────────────────

with st.expander("Drawdown Profile by Regime", expanded=False):
    fig_dd = go.Figure()

    for regime, start, end in blocks:
        fig_dd.add_vrect(
            x0=str(start), x1=str(end),
            fillcolor=REGIME_COLORS[regime],
            layer="below",
            line_width=0,
        )

    dd_series = drawdown_series(signal)
    fig_dd.add_trace(go.Scatter(
        x=dd_series.index,
        y=dd_series.values * 100,
        name="Drawdown",
        fill="tozeroy",
        line=dict(color="#1a3a5c", width=1.2),
        fillcolor="rgba(26, 58, 92, 0.20)",
        hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))

    for regime in REGIME_NAMES:
        fig_dd.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=REGIME_LINE_COLORS[regime], symbol="square"),
            name=regime,
        ))

    fig_dd.update_layout(
        **LAYOUT,
        title=f"Drawdown — {signal_src} (shaded by regime)",
        yaxis_title="Drawdown (%)",
        xaxis_title="",
        height=350,
    )
    st.plotly_chart(fig_dd, use_container_width=True)
