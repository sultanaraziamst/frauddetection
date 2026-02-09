import datetime
import streamlit as st
import os
import json
import numpy as np


def load_metrics(metrics_file):
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def render_sidebar():
    st.sidebar.markdown("### 🛡️ Fraud Guard")
    
    # Theme Selection
    theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
    
    st.sidebar.markdown("---")
    
    # Simple Radio without label
    page = st.sidebar.radio("", ["Overview", "Model Analysis"], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"System Version: v2.2\nMode: {theme}")
    
    return page, theme

def init_app():
    st.set_page_config(
        page_title="Fraud Detection Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def generate_post_prediction_trend(pred_probs, days=7):
    """
    pred_probs: array-like fraud probabilities from model
    returns dataframe for next 7 days fraud trend
    """
    base_rate = np.mean(pred_probs)

    future_dates = [
        (datetime.now() + datetime.timedelta(days=i)).strftime('%a')
        for i in range(1, days + 1)
    ]

    # Small realistic fluctuation
    trend = [
        max(0.05, min(0.5, base_rate + np.random.normal(0, 0.015)))
        for _ in range(days)
    ]

    normal = [100 - (t * 100) for t in trend]

    return pd.DataFrame({
        "date": future_dates,
        "rate": np.round(np.array(trend) * 100, 2),
        "normal": np.round(normal, 2)
    })
