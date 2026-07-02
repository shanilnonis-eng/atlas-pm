import streamlit as st

st.title("🌐 Atlas PM")
st.markdown("### AI-Augmented Portfolio Management")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
**Atlas PM** is a professional portfolio management and investment analysis platform
built to demonstrate institutional-grade investment workflows.

#### What you can do:

| Step | Page | Description |
|------|------|-------------|
| 1 | **Universe & Data** | Select your investment universe and date range |
| 2 | **Portfolio Construction** | Build optimal portfolios using multiple models |
| 3 | **Performance Analytics** | Analyse returns, drawdowns, and risk-adjusted performance |
| 4 | **Risk Management** | VaR, CVaR, stress testing, and risk contribution |
| 5 | **AI Commentary** | Investment committee-style narrative powered by Claude |
| 6 | **IC Report** | Export a full monthly investment committee report |
| 7 | **Black-Litterman** | Express manager views and blend with market equilibrium |
| 8 | **Factor Attribution** | Fama-French factor decomposition and transaction costs |

#### How to start:
1. Click **Universe & Data** in the sidebar
2. Select assets and date range, then click **Load Data**
3. Go to **Portfolio Construction** to choose your model and optimise
4. Explore analytics and risk metrics, then generate AI commentary

---
**Disclaimer:** Atlas PM is a portfolio analytics and educational tool. It does not constitute
investment advice. All analysis is based on historical data. Past performance is not
indicative of future results.
    """)

with col2:
    st.markdown("""
**Features**
- Multi-asset ETF universe (10 asset classes)
- 4 portfolio construction models
- Black-Litterman with manager views
- 15+ quantitative risk metrics
- Historical & parametric VaR/CVaR
- 5 pre-built stress test scenarios
- Rolling analytics & drawdown analysis
- Correlation matrix & risk attribution
- Fama-French factor attribution
- Transaction cost modelling
- AI-powered IC commentary (Claude)
- PDF report export

**Tech Stack**
- Python 3.12+
- Streamlit
- yfinance (market data)
- pandas / numpy / scipy
- Plotly (charts)
- Anthropic Claude API (AI)
    """)

    st.info("Use the **sidebar** on the left to navigate.", icon="👈")
