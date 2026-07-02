# Atlas PM — AI-Augmented Portfolio Management

A professional portfolio management and investment analysis platform demonstrating institutional-grade investment workflows, quantitative risk management, and AI-powered investment commentary.

Built as a portfolio project targeting roles in London asset management.

---

## What This Project Demonstrates

| Skill Area | Implementation |
|-----------|---------------|
| **Portfolio Construction** | Equal Weight, Minimum Variance, Maximum Sharpe (tangency), Risk Parity (ERC) via scipy convex optimisation |
| **Return Analytics** | CAGR, Sharpe, Sortino, Calmar, Information Ratio, Alpha, Beta, rolling metrics, monthly heatmap |
| **Risk Management** | Historical & Parametric VaR/CVaR, Kupiec back-test, stress scenarios, component risk contribution, Ledoit-Wolf shrinkage |
| **AI Integration** | Claude (LLM) generates IC-style commentary grounded in actual portfolio data, with explicit hallucination controls |
| **Data Engineering** | yfinance ETF proxy universe, caching, alignment, missing data handling |
| **Visualisation** | Plotly interactive charts: efficient frontier, correlation heatmap, monthly returns heatmap, drawdown analysis |
| **Financial Governance** | Documented assumptions, limitations, bias disclosures, model risk section |

---

## Project Structure

```
atlas-pm/
├── app.py                       # Streamlit entry point
├── requirements.txt
├── config/
│   └── settings.py              # Universe, parameters, constants
├── data/
│   └── loader.py                # yfinance data loading & caching
├── analytics/
│   ├── returns.py               # Return metrics (Sharpe, Sortino, drawdown, etc.)
│   └── risk.py                  # VaR, CVaR, stress tests, risk contribution
├── construction/
│   └── optimiser.py             # EW, MinVar, MaxSharpe, Risk Parity
├── ai/
│   └── commentary.py            # Claude API investment commentary
└── ui/
    ├── components/
    │   ├── charts.py            # Plotly chart components
    │   └── metrics.py           # Metric formatting & display
    └── pages/
        ├── 1_Universe_and_Data.py
        ├── 2_Portfolio_Construction.py
        ├── 3_Performance_Analytics.py
        ├── 4_Risk_Management.py
        ├── 5_AI_Commentary.py
        └── 6_IC_Report.py
```

---

## Installation

```bash
git clone https://github.com/yourusername/atlas-pm
cd atlas-pm
pip install -r requirements.txt

# Optional: AI commentary (requires Anthropic API key)
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

streamlit run app.py
```

---

## Asset Universe

10 liquid ETF proxies covering major asset classes:

| Asset Class | ETF | Rationale |
|------------|-----|-----------|
| US Equities (S&P 500) | SPY | Deepest, most liquid large-cap US equity exposure |
| UK Equities (FTSE 100) | ISF.L | London-listed; GBP-denominated |
| European Equities | IEUR | Eurozone equity exposure |
| Emerging Markets | EEM | EM risk/return premium |
| US Aggregate Bonds | AGG | Investment-grade fixed income anchor |
| Global Bonds (Hedged) | BNDW | Diversified global bond exposure |
| Gold | GLD | Inflation hedge / safe haven |
| Commodities | PDBC | Broad commodity exposure |
| REITs | REET | Real estate / inflation-sensitive |
| Cash Proxy (T-Bills) | BIL | Near-zero duration; risk-free rate proxy |

ETFs are used rather than individual stocks to: (1) represent true asset-class exposure, (2) reduce single-stock idiosyncratic risk, (3) provide clean dividend-adjusted pricing, and (4) mirror how multi-asset institutional portfolios are actually constructed.

---

## Portfolio Construction Models

### Equal Weight
Allocates 1/N to each selected asset. No estimation required. Serves as a hard-to-beat baseline.

### Minimum Variance
Minimises portfolio variance: `min w'Σw` subject to `Σw=1` and weight bounds.
Uses Ledoit-Wolf shrinkage on the covariance matrix to reduce estimation error.

### Maximum Sharpe (Tangency Portfolio)
Maximises `(μ - rf) / σ`. Solved via multi-start SLSQP to avoid local minima.
Uses historical mean returns as expected return inputs (documented limitation).

### Risk Parity (Equal Risk Contribution)
Each asset contributes equally to total portfolio variance.
Objective: `w_i × (Σw)_i = σ²_p / N` for all i.
Does not require return estimates — purely a risk-based model.

---

## Risk Methodology

- **Historical VaR**: empirical quantile of the return distribution — no distributional assumption
- **CVaR / Expected Shortfall**: average loss beyond VaR — a coherent risk measure
- **Parametric VaR**: Gaussian assumption; shown as comparison to illustrate fat-tail underestimation
- **Kupiec back-test**: tests whether the actual breach rate matches the theoretical rate
- **Component risk contribution**: decomposes portfolio volatility to asset level
- **Stress scenarios**: 5 historically-calibrated macro shock scenarios

---

## AI Commentary Controls

The Claude AI integration follows strict controls to prevent misleading outputs:

1. The model is given only actual computed portfolio statistics — no internet access
2. The system prompt explicitly prohibits inventing numbers not in the context
3. All output is labelled as AI-generated and subject to human review
4. Future return predictions are explicitly forbidden
5. All commentary ends with a "not investment advice" disclaimer

---

## Limitations & Governance

This tool is built for educational and research purposes. Key limitations:

- Historical data is not predictive of future returns
- Transaction costs are not modelled
- Rebalancing assumes frictionless execution at closing prices
- Covariance estimates are noisy and time-varying
- Maximum Sharpe is highly sensitive to return estimation error
- AI commentary may contain errors and requires expert review
- **This is not investment advice**

---

## Tech Stack

- **Python 3.12**
- **Streamlit** — multi-page web app framework
- **yfinance** — free historical market data
- **pandas / numpy** — data manipulation
- **scipy** — convex optimisation (SLSQP solver)
- **plotly** — interactive charts
- **Anthropic Claude API** — AI commentary (claude-opus-4-8)
- **python-dotenv** — environment variable management

---

*Built with Python · Streamlit · Anthropic Claude · Not Investment Advice*
