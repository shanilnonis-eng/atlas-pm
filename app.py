"""
Atlas PM — AI-Augmented Portfolio Management
Main entry point using st.navigation() (Streamlit 1.36+).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Atlas PM",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem;
    }
    h2 { color: #1a3a5c; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.3rem; }
    h3 { color: #1a3a5c; }
    [data-testid="stSidebar"] { background-color: #f0f4f8; }
</style>
""", unsafe_allow_html=True)

# Define all pages
pages = {
    "Atlas PM": [
        st.Page("pages/0_Home.py",                   title="Home",                   icon="🏠"),
    ],
    "Analysis": [
        st.Page("pages/1_Universe_and_Data.py",      title="Universe & Data",        icon="📊"),
        st.Page("pages/2_Portfolio_Construction.py", title="Portfolio Construction", icon="🏗️"),
        st.Page("pages/3_Performance_Analytics.py",  title="Performance Analytics",  icon="📈"),
        st.Page("pages/4_Risk_Management.py",        title="Risk Management",        icon="⚠️"),
    ],
    "Advanced": [
        st.Page("pages/7_Black_Litterman.py",        title="Black-Litterman",        icon="🎯"),
        st.Page("pages/8_Factor_Attribution.py",     title="Factor Attribution",     icon="🔬"),
        st.Page("pages/9_Walkforward_Backtest.py",   title="Walk-Forward Backtest",  icon="🔁"),
        st.Page("pages/10_Brinson_Attribution.py",   title="Brinson Attribution",    icon="🏛️"),
        st.Page("pages/11_Regime_Detection.py",      title="Regime Detection",       icon="🔀"),
    ],
    "AI & Reports": [
        st.Page("pages/5_AI_Commentary.py",          title="AI Commentary",          icon="🤖"),
        st.Page("pages/6_IC_Report.py",              title="IC Report",              icon="📄"),
    ],
}

pg = st.navigation(pages)
pg.run()
