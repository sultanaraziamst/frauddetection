import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def metric_card(label, value, delta=None, color=None):
    delta_html = ""
    if delta is not None:
        delta_class = "delta-pos" if '+' in str(delta) or float(str(delta).replace('%','')) > 0 else "delta-neg"
        delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>'
    
    style_color = f'style="color: {color}"' if color else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" {style_color}>{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def get_plot_config(theme):
    font_color = "white" if theme == "Dark" else "#1f2937"
    grid_color = "rgba(255,255,255,0.1)" if theme == "Dark" else "rgba(0,0,0,0.1)"
    return font_color, grid_color

def plot_fraud_trend(data, theme="Dark"):
    font_color, grid_color = get_plot_config(theme)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['date'], y=data['rate'],
        mode='lines+markers',
        name='Fraud Rate',
        line=dict(color='#f59e0b', width=3),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=data['date'], y=data['normal'],
        mode='lines',
        name='Normal',
        line=dict(color='#3b82f6', width=2)
    ))
    
    fig.update_layout(
        title="Fraud Rate Trend (Last 7 Days)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color)
    )
    return fig

def plot_feature_importance(imp_dict, theme="Dark"):
    font_color, grid_color = get_plot_config(theme)
    
    df_imp = pd.DataFrame(list(imp_dict.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values('Importance', ascending=True)
    
    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                 color='Importance', color_continuous_scale='Viridis')
    
    fig.update_layout(
        title="Top Feature Importance",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=font_color),
        xaxis=dict(showgrid=True, gridcolor=grid_color),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def plot_confusion_matrix(cm, theme="Dark"):
    font_color, _ = get_plot_config(theme)
    
    z = cm
    x = ['Safe', 'Fraud']
    y = ['Safe', 'Fraud']
    
    fig = px.imshow(z, x=x, y=y, color_continuous_scale='Blues', text_auto=True)
    fig.update_layout(
        title="Confusion Matrix",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=font_color)
    )
    return fig

def plot_roc_curve(fpr, tpr, auc_score, theme="Dark"):
    font_color, grid_color = get_plot_config(theme)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC = {auc_score:.4f}', mode='lines', line=dict(color='#4f46e5', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', line=dict(dash='dash', color='gray')))
    
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=font_color),
        xaxis=dict(showgrid=True, gridcolor=grid_color),
        yaxis=dict(showgrid=True, gridcolor=grid_color)
    )
    return fig

def plot_pr_curve(recall, precision, theme="Dark"):
    font_color, grid_color = get_plot_config(theme)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, name='Precision-Recall', mode='lines', line=dict(color='#10b981', width=3)))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=font_color),
        xaxis=dict(showgrid=True, gridcolor=grid_color),
        yaxis=dict(showgrid=True, gridcolor=grid_color)
    )
    return fig
