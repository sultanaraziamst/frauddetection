import streamlit as st
import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard import init_app, render_sidebar, load_metrics, render_overview, render_analysis
from dashboard.styles import get_css

METRICS_FILE = 'models/metrics.json'

def main():
    # 1. Initialize Page Config
    init_app()
    
    # 2. Sidebar Navigation & Theme Selection
    page, theme = render_sidebar()
    
    # Apply Theme CSS
    st.markdown(get_css(theme), unsafe_allow_html=True)
    
    # 3. Load Data
    metrics = load_metrics(METRICS_FILE)
    if not metrics:
        # Try looking up one level if running from src
        metrics = load_metrics(os.path.join('..', METRICS_FILE))
        
    available_models = list(metrics.keys()) if metrics else []
    
    # 4. Render Pages
    if page == "Overview":
        render_overview(metrics, theme)
    elif page == "Model Analysis":
        render_analysis(metrics, available_models, theme)

if __name__ == "__main__":
    main()

