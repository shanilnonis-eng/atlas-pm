"""
Central configuration for Atlas PM.

All asset universe definitions, benchmark tickers, display labels, and
global constants live here. Change this file to expand the universe or
adjust default parameters — nothing else needs to change.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Asset Universe
# Each entry maps a human-readable label to a liquid ETF ticker on Yahoo
# Finance.  ETFs are used so we get clean daily pricing back to ~2005-2010
# without survivorship-bias issues from individual stock selection.
# ---------------------------------------------------------------------------
UNIVERSE: Dict[str, str] = {
    "US Equities (S&P 500)":       "SPY",
    "UK Equities (FTSE 100)":      "ISF.L",
    "European Equities (Euro Stoxx)": "IEUR",
    "Emerging Markets":            "EEM",
    "US Aggregate Bonds":          "AGG",
    "Global Bonds (Hedged)":       "BNDW",
    "Gold":                        "GLD",
    "Commodities (Broad)":         "PDBC",
    "REITs (Global)":              "REET",
    "Cash Proxy (T-Bills)":        "BIL",
}

# Short codes used in charts and tables (must match UNIVERSE keys exactly)
ASSET_SHORT_NAMES: Dict[str, str] = {
    "US Equities (S&P 500)":          "US EQ",
    "UK Equities (FTSE 100)":         "UK EQ",
    "European Equities (Euro Stoxx)": "EU EQ",
    "Emerging Markets":               "EM EQ",
    "US Aggregate Bonds":             "US BOND",
    "Global Bonds (Hedged)":          "GL BOND",
    "Gold":                           "GOLD",
    "Commodities (Broad)":            "COMMOD",
    "REITs (Global)":                 "REIT",
    "Cash Proxy (T-Bills)":           "CASH",
}

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
BENCHMARK_TICKER = "SPY"
BENCHMARK_LABEL  = "S&P 500 (SPY)"

RISK_FREE_TICKER = "BIL"          # 1-month T-Bill ETF as daily risk-free proxy
RISK_FREE_LABEL  = "T-Bill (BIL)"

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_START_DATE   = "2015-01-01"
DEFAULT_END_DATE     = None          # None → today
DEFAULT_REBAL_FREQ   = "M"          # pandas offset alias: M=monthly, Q=quarterly
TRADING_DAYS_PER_YEAR = 252

# Optimisation constraints
MIN_WEIGHT  = 0.00   # allow zero allocation
MAX_WEIGHT  = 0.40   # no single asset above 40 %
MIN_ASSETS  = 3      # portfolio must hold at least 3 assets

# Portfolio construction model names (canonical list)
MODEL_NAMES = ["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity"]

# Risk parameters
VAR_CONFIDENCE   = 0.95   # 95 % VaR / CVaR
ROLLING_WINDOW   = 63     # ~3-month rolling window (trading days)

# ---------------------------------------------------------------------------
# Stress scenarios
# Each scenario is a dict of {asset_short_name: shock_pct} where shock is
# applied as a one-day instantaneous return shock to the current portfolio.
# Shocks are approximate historical analogues — documented with sources.
# ---------------------------------------------------------------------------
STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "2008 Global Financial Crisis (Sep–Nov 2008)": {
        "US EQ":   -0.35,
        "UK EQ":   -0.30,
        "EU EQ":   -0.33,
        "EM EQ":   -0.45,
        "US BOND": +0.05,
        "GL BOND": +0.03,
        "GOLD":    -0.15,
        "COMMOD":  -0.40,
        "REIT":    -0.50,
        "CASH":     0.00,
    },
    "COVID-19 Crash (Feb–Mar 2020)": {
        "US EQ":   -0.34,
        "UK EQ":   -0.32,
        "EU EQ":   -0.35,
        "EM EQ":   -0.28,
        "US BOND": +0.04,
        "GL BOND": +0.02,
        "GOLD":    -0.03,
        "COMMOD":  -0.35,
        "REIT":    -0.40,
        "CASH":     0.00,
    },
    "2022 Rate Shock (Equities & Bonds Sell Off)": {
        "US EQ":   -0.19,
        "UK EQ":   -0.05,
        "EU EQ":   -0.15,
        "EM EQ":   -0.22,
        "US BOND": -0.13,
        "GL BOND": -0.16,
        "GOLD":    -0.02,
        "COMMOD":  +0.15,
        "REIT":    -0.28,
        "CASH":    +0.02,
    },
    "Equity Bull / Bond Bear (+20 % Equities, -10 % Bonds)": {
        "US EQ":   +0.20,
        "UK EQ":   +0.18,
        "EU EQ":   +0.18,
        "EM EQ":   +0.22,
        "US BOND": -0.10,
        "GL BOND": -0.12,
        "GOLD":    +0.05,
        "COMMOD":  +0.08,
        "REIT":    +0.15,
        "CASH":     0.00,
    },
    "Stagflation (High Inflation, Low Growth)": {
        "US EQ":   -0.15,
        "UK EQ":   -0.10,
        "EU EQ":   -0.18,
        "EM EQ":   -0.12,
        "US BOND": -0.10,
        "GL BOND": -0.08,
        "GOLD":    +0.20,
        "COMMOD":  +0.25,
        "REIT":    -0.05,
        "CASH":    +0.03,
    },
}

# ---------------------------------------------------------------------------
# AI Commentary
# ---------------------------------------------------------------------------
AI_MODEL = "claude-opus-4-8"        # best reasoning for investment commentary
MAX_COMMENTARY_TOKENS = 1500

# ---------------------------------------------------------------------------
# UI / Branding
# ---------------------------------------------------------------------------
APP_TITLE       = "Atlas PM"
APP_SUBTITLE    = "AI-Augmented Portfolio Management"
APP_ICON        = "🌐"
ACCENT_COLOR    = "#1a3a5c"         # dark navy — institutional look
POSITIVE_COLOR  = "#2ecc71"
NEGATIVE_COLOR  = "#e74c3c"
NEUTRAL_COLOR   = "#95a5a6"
