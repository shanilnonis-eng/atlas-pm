"""
PDF report export for Atlas PM.

Generates a professional Investment Committee report PDF using fpdf2.
The PDF contains:
  - Cover page with Atlas PM branding
  - Portfolio summary and allocation table
  - Performance statistics vs benchmark
  - Risk metrics (VaR, CVaR, drawdown)
  - Stress test results
  - Governance / disclaimer section

Design: clean, minimal, institutional — no garish colours.
Charts are not embedded (they require kaleido/browser rendering).
Tables and metrics are the focus.

Usage:
    from reporting.pdf_export import generate_pdf_report
    pdf_bytes = generate_pdf_report(stats, weights, stress_df, ...)
    # in Streamlit: st.download_button("Download PDF", pdf_bytes, "report.pdf")
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import io

import pandas as pd
import numpy as np

try:
    from fpdf import FPDF, XPos, YPos
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colour palette (RGB)
# ---------------------------------------------------------------------------
NAVY    = (26, 58, 92)
BLUE    = (46, 134, 171)
RED     = (232, 72, 85)
GREEN   = (46, 204, 113)
GREY    = (149, 165, 166)
LGREY   = (240, 244, 248)
WHITE   = (255, 255, 255)
BLACK   = (30, 30, 30)


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

def _safe(text: str) -> str:
    """Replace characters outside latin-1 with ASCII equivalents."""
    return (
        str(text)
        .replace("—", " - ")   # em dash
        .replace("–", " - ")   # en dash
        .replace("’", "'")     # right single quote
        .replace("‘", "'")     # left single quote
        .replace("“", '"')     # left double quote
        .replace("”", '"')     # right double quote
        .replace("•", "-")     # bullet
        .replace("…", "...")   # ellipsis
        .replace("²", "2")     # superscript 2
        .replace("³", "3")     # superscript 3
        .replace("®", "(R)")   # registered trademark
        .replace("©", "(c)")   # copyright
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


class AtlasPDF(FPDF):
    """Custom FPDF subclass with institutional header and footer."""

    def __init__(self, title: str, model: str, period: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_title  = title
        self.model_name    = _safe(model)
        self.report_period = _safe(period)
        self.set_margins(20, 15, 20)
        self.set_auto_page_break(True, margin=20)

    def normalize_text(self, text: str) -> str:
        """Auto-sanitise all strings to latin-1 before FPDF processes them."""
        return super().normalize_text(_safe(str(text)))

    def header(self):
        if self.page_no() == 1:
            return  # cover page has its own design
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*NAVY)
        self.cell(0, 6, f"Atlas PM | Investment Committee Report | {self.model_name}", align="L")
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GREY)
        self.cell(
            0, 5,
            "CONFIDENTIAL | NOT INVESTMENT ADVICE | Atlas PM - For educational and analytical purposes only",
            align="C",
        )

    # ------------------------------------------------------------------
    # Helper: section title
    # ------------------------------------------------------------------
    def section_title(self, text: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*NAVY)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.5)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)
        self.set_text_color(*BLACK)

    # ------------------------------------------------------------------
    # Helper: two-column metric
    # ------------------------------------------------------------------
    def metric_row(self, label: str, value: str, highlight: bool = False):
        self.set_font("Helvetica", "", 9)
        self.set_fill_color(*LGREY)
        self.set_text_color(*BLACK)
        fill = highlight
        self.cell(100, 6, f"  {label}", fill=fill, border=0)
        self.set_font("Helvetica", "B", 9)
        self.cell(70, 6, value, fill=fill, align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, border=0)

    # ------------------------------------------------------------------
    # Helper: table
    # ------------------------------------------------------------------
    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[float]):
        # header row
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        for h, w in zip(headers, col_widths):
            self.cell(w, 6, f"  {h}", fill=True, border=0)
        self.ln()

        # data rows (alternating shade)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*BLACK)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(*LGREY if fill else WHITE)
            for val, w in zip(row, col_widths):
                self.cell(w, 5.5, f"  {val}", fill=fill, border=0)
            self.ln()
        self.set_text_color(*BLACK)
        self.ln(2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf_report(
    model_name: str,
    period: str,
    stats: dict,
    weights: pd.Series,
    bench_stats: dict,
    stress_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    var_1d: float,
    cvar_1d: float,
    ann_vol: float,
    max_dd: float,
    confidence: float = 0.95,
    portfolio_value: float = 1_000_000,
    ai_commentary: str | None = None,
) -> bytes:
    """
    Generate a professional IC report PDF and return as bytes.

    Parameters
    ----------
    All parameters mirror the data already computed in the IC Report page.
    Returns bytes that can be passed directly to st.download_button().
    """
    if not _FPDF_AVAILABLE:
        raise ImportError(
            "fpdf2 is required for PDF export. Install with: pip install fpdf2"
        )

    report_date = datetime.today().strftime("%d %B %Y")

    # sanitise all caller-supplied strings up front
    model_name = _safe(model_name)
    period     = _safe(period)

    pdf = AtlasPDF(
        title="Investment Committee Report",
        model=model_name,
        period=period,
    )

    # ------------------------------------------------------------------
    # Cover page
    # ------------------------------------------------------------------
    pdf.add_page()

    # Navy background strip
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 80, "F")

    # Logo text
    pdf.set_xy(20, 20)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 12, "Atlas PM", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(200, 220, 240)
    pdf.cell(0, 8, "AI-Augmented Portfolio Management", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Report metadata
    pdf.set_xy(20, 95)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 10, "Investment Committee Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*BLACK)
    for line in [
        f"Report Date: {report_date}",
        f"Portfolio Model: {model_name}",
        f"Analysis Period: {period}",
        f"Portfolio Value: £{portfolio_value:,.0f} (notional)",
    ]:
        pdf.ln(2)
        pdf.cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Disclaimer box on cover
    pdf.set_xy(20, 200)
    pdf.set_fill_color(*LGREY)
    pdf.set_draw_color(*GREY)
    pdf.rect(20, 200, 170, 40, "F")
    pdf.set_xy(24, 204)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*RED)
    pdf.cell(0, 5, "IMPORTANT DISCLAIMER", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(24, 210)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(
        162, 4.5,
        "This report is produced by Atlas PM, a portfolio analytics and educational tool. "
        "It does not constitute investment advice, a recommendation to buy or sell any security, "
        "or a solicitation of any investment. All analysis is based on historical data. "
        "Past performance is not indicative of future results. AI-generated commentary "
        "requires independent human review before any reliance is placed upon it."
    )

    # ------------------------------------------------------------------
    # Page 2: Executive Summary
    # ------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("1. Executive Summary")

    def fmt_pct(v):
        return f"{v:.2%}" if isinstance(v, float) and not np.isnan(v) else "N/A"

    def fmt_ratio(v, d=2):
        if isinstance(v, float) and not np.isnan(v):
            return f"{v:.{d}f}"
        return "N/A"

    metrics = [
        ("Total Return",           fmt_pct(stats.get("Total Return", float("nan")))),
        ("Annualised Return (CAGR)", fmt_pct(stats.get("Ann. Return", float("nan")))),
        ("Annualised Volatility",   fmt_pct(ann_vol)),
        ("Sharpe Ratio",            fmt_ratio(stats.get("Sharpe Ratio", float("nan")))),
        ("Sortino Ratio",           fmt_ratio(stats.get("Sortino Ratio", float("nan")))),
        ("Calmar Ratio",            fmt_ratio(stats.get("Calmar Ratio", float("nan")))),
        ("Maximum Drawdown",        fmt_pct(max_dd)),
        (f"1-Day VaR ({int(confidence*100)}%)",  fmt_pct(var_1d)),
        (f"1-Day CVaR ({int(confidence*100)}%)", fmt_pct(cvar_1d)),
        ("Beta (vs Benchmark)",     fmt_ratio(stats.get("Beta", float("nan")), 3)),
        ("Jensen's Alpha (ann.)",   fmt_pct(stats.get("Alpha (ann.)", float("nan")))),
        ("Information Ratio",       fmt_ratio(stats.get("Information Ratio", float("nan")))),
    ]

    for i, (label, value) in enumerate(metrics):
        pdf.metric_row(label, value, highlight=(i % 2 == 0))

    # ------------------------------------------------------------------
    # Page 2 continued: Allocation
    # ------------------------------------------------------------------
    pdf.section_title("2. Portfolio Allocation")

    w_sorted = weights.sort_values(ascending=False)
    alloc_rows = [
        [_safe(str(asset)), f"{w:.2%}", f"GBP {w * portfolio_value:,.0f}"]
        for asset, w in w_sorted.items()
        if w > 0.001
    ]
    pdf.table(
        ["Asset Class", "Weight", f"Notional (GBP {portfolio_value/1e6:.1f}M)"],
        alloc_rows,
        [100, 35, 35],
    )

    # HHI
    hhi = float((weights ** 2).sum())
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5, f"Herfindahl-Hirschman Index (concentration): {hhi:.4f}  "
                   f"(0 = fully diversified, 1 = fully concentrated)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*BLACK)

    # ------------------------------------------------------------------
    # Page 3: Performance vs Benchmark
    # ------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("3. Performance vs Benchmark")

    compare_items = [
        ("Total Return",         "Total Return"),
        ("Ann. Return",          "Ann. Return"),
        ("Ann. Volatility",      "Ann. Volatility"),
        ("Sharpe Ratio",         "Sharpe Ratio"),
        ("Sortino Ratio",        "Sortino Ratio"),
        ("Calmar Ratio",         "Calmar Ratio"),
        ("Max Drawdown",         "Max Drawdown"),
        ("Skewness",             "Skewness"),
        ("Kurtosis (excess)",    "Kurtosis (excess)"),
        ("% Positive Days",      "% Positive Days"),
        ("Beta",                 "Beta"),
        ("Alpha (ann.)",         "Alpha (ann.)"),
        ("Information Ratio",    "Information Ratio"),
    ]

    pct_keys = {"Total Return","Ann. Return","Ann. Volatility","Max Drawdown",
                "Alpha (ann.)","% Positive Days"}

    perf_rows = []
    for label, key in compare_items:
        pv = stats.get(key, float("nan"))
        bv = bench_stats.get(key, float("nan"))
        fmt = fmt_pct if key in pct_keys else fmt_ratio
        perf_rows.append([label, fmt(pv), fmt(bv)])

    pdf.table(["Metric", model_name, "Benchmark"], perf_rows, [90, 50, 30])

    # ------------------------------------------------------------------
    # Page 3 continued: Risk
    # ------------------------------------------------------------------
    pdf.section_title("4. Risk Metrics")

    risk_metrics = [
        (f"1-Day Historical VaR ({int(confidence*100)}%)",  fmt_pct(var_1d),    f"£{var_1d * portfolio_value:,.0f}"),
        (f"1-Day CVaR / ES ({int(confidence*100)}%)",       fmt_pct(cvar_1d),   f"£{cvar_1d * portfolio_value:,.0f}"),
        (f"10-Day VaR ({int(confidence*100)}%)",
            fmt_pct(var_1d * np.sqrt(10)),
            f"£{var_1d * np.sqrt(10) * portfolio_value:,.0f}"),
        ("Annualised Volatility",                          fmt_pct(ann_vol),   ""),
        ("Maximum Drawdown",                               fmt_pct(max_dd),    ""),
    ]
    pdf.table(["Risk Metric", "Percentage", "£ Notional"], risk_metrics, [90, 40, 40])

    # Risk contribution table
    if not risk_df.empty:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 6, "Risk Contribution by Asset", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*BLACK)
        rc_rows = []
        for asset in risk_df.index:
            rc_rows.append([
                asset,
                f"{risk_df.loc[asset, 'Weight']:.2%}",
                f"{risk_df.loc[asset, '% Risk Contribution']:.2%}",
            ])
        pdf.table(["Asset", "Weight", "% Risk Contribution"], rc_rows, [100, 40, 30])

    # ------------------------------------------------------------------
    # Page 4: Stress Tests
    # ------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("5. Stress Test Analysis")

    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(
        0, 4.5,
        "The following scenarios apply historically-calibrated instantaneous shocks to "
        "the current portfolio. These are not forecasts — they are planning tools to "
        "understand portfolio sensitivity. Correlation dynamics during stress events are "
        "not modelled; results should be treated as indicative."
    )
    pdf.ln(3)

    stress_rows = []
    for scenario in stress_df.index:
        pnl = stress_df.loc[scenario, "Portfolio P&L"]
        pnl_gbp = pnl * portfolio_value
        flag = "v" if pnl < 0 else "^"   # ASCII arrows (no unicode)
        stress_rows.append([
            _safe(str(scenario))[:55],
            f"{flag} {pnl:.2%}",
            f"GBP {pnl_gbp:,.0f}",        # avoid £ symbol encoding issues
        ])

    pdf.table(["Scenario", "Portfolio P&L", "£ Impact"], stress_rows, [120, 30, 20])

    # ------------------------------------------------------------------
    # Page 5: AI Commentary (if provided)
    # ------------------------------------------------------------------
    if ai_commentary:
        pdf.add_page()
        pdf.section_title("6. Investment Commentary (AI-Generated)")

        pdf.set_fill_color(*LGREY)
        pdf.rect(20, pdf.get_y(), 170, 8, "F")
        pdf.set_xy(22, pdf.get_y() + 1)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*RED)
        pdf.cell(0, 6,
                 "⚠ AI-GENERATED CONTENT — Requires independent human review. Not investment advice.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)
        pdf.set_text_color(*BLACK)

        # Strip markdown and non-latin-1 chars for built-in Helvetica font
        clean_commentary = (
            ai_commentary
            .replace("**", "").replace("##", "").replace("#", "")
            .replace("—", " - ").replace("–", " - ")   # em/en dash
            .replace("‘", "'").replace("’", "'")        # smart quotes
            .replace("“", '"').replace("”", '"')
            .replace("•", "-").replace("…", "...")      # bullet, ellipsis
            .replace("²", "2").replace("³", "3")        # superscripts
        )
        # replace any remaining non-latin-1 chars with ?
        clean_commentary = clean_commentary.encode("latin-1", errors="replace").decode("latin-1")
        pdf.set_font("Helvetica", "", 8.5)
        pdf.multi_cell(0, 5, clean_commentary)

    # ------------------------------------------------------------------
    # Last page: Governance
    # ------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("7. Governance & Model Risk")

    gov_items = [
        ("Data Source",            "Yahoo Finance via yfinance (free tier). Independent verification not performed."),
        ("Survivorship Bias",      "ETF proxies reduce but do not eliminate survivorship bias."),
        ("Look-Ahead Bias",        "All analytics use only data available at the calculation date."),
        ("Transaction Costs",      "Not modelled in performance figures (see Turnover Analysis page)."),
        ("Return Estimation",      "Historical mean returns are poor forward-looking predictors."),
        ("Covariance Stability",   "Correlations are assumed stable. They break down in crises."),
        ("VaR Scaling",            "Multi-day VaR uses √T scaling. Assumes i.i.d. returns."),
        ("AI Commentary",          "Generated by Claude (LLM). May contain errors. Human review required."),
        ("Rebalancing",            "Assumes frictionless execution at closing prices."),
        ("Not Investment Advice",  "THIS TOOL IS FOR EDUCATIONAL AND ANALYTICAL PURPOSES ONLY."),
    ]

    for label, text in gov_items:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*NAVY)
        pdf.cell(55, 5.5, f"  {label}:", border=0)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*BLACK)
        pdf.multi_cell(115, 5.5, text)

    # Footer note
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(
        0, 4.5,
        f"Report generated by Atlas PM on {report_date}. "
        "Atlas PM is a portfolio analytics tool built in Python using Streamlit, yfinance, scipy, "
        "plotly, and Anthropic Claude API. This report does not constitute investment advice."
    )

    return bytes(pdf.output())


def is_available() -> bool:
    return _FPDF_AVAILABLE
