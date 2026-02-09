import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import random
from datetime import datetime, timedelta
from .components import metric_card, plot_fraud_trend, plot_feature_importance, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from utils import RobustPreprocessor

MODEL_DIR = 'models'

def render_overview(metrics, theme="Dark"):
    st.title("Credit Card Fraud Detection Analysis")
    
    # Top KPI Cards
    # Use real metrics from best model if available, else static
    best_acc = "N/A"
    best_f1 = "N/A"
    if metrics:
        best_model = max(metrics, key=lambda x: metrics[x]['accuracy'])
        best_acc = f"{metrics[best_model]['accuracy']*100:.2f}%"
        best_f1 = f"{metrics[best_model]['f1_score']*100:.2f}%"

    val_color = "#ffffff" if theme == "Dark" else "#1f2937"
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Transactions", "284,807", "+12%", color=val_color)
    with c2: metric_card("Fraudulent Transactions", "492", "+2", color="#ef4444")
    with c3: metric_card("Fraud Rate", "0.17%", "-0.01%", color="#f59e0b")
    with c4: metric_card("Best Accuracy", best_acc, "+0.02%", color="#10b981")
    
    # Row 2: Live Analysis & Alerts
    col_left, col_mid, col_right = st.columns([2, 1, 1.5])
    
    with col_left:
        st.subheader("Live Transaction Analysis")
        # Mock Trend Data - consistent with "Total Transactions" concept
        dates = [(datetime.now() - timedelta(days=i)).strftime('%a') for i in range(7)][::-1]
        trend_data = pd.DataFrame({
            'date': dates,
            'rate': [0.15, 0.18, 0.16, 0.21, 0.17, 0.14, 0.17], 
            'normal': [99.85, 99.82, 99.84, 99.79, 99.83, 99.86, 99.83] 
        })
        st.plotly_chart(plot_fraud_trend(trend_data, theme=theme), use_container_width=True)
        
    with col_mid:
        st.subheader("Alerts Overview")
        st.markdown('<div class="alert-card-high"><b>21 High Risk Alerts</b><br>Immediate Action Required</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-card-med"><b>14 Medium Risk Alerts</b><br>Under Review</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-card-low"><b>6 Low Risk Alerts</b><br>Resolved</div>', unsafe_allow_html=True)

    with col_right:
        st.subheader("Recent Fraud Cases")
        fraud_cases = pd.DataFrame({
            "ID": ["3421", "2185", "8765", "9912", "1102"],
            "Amount": ["$1,200", "$1,060", "$1,500", "$320", "$980"],
            "Status": ["Blocked", "Review", "Blocked", "Verified", "Review"]
        })
        # Style the table to look like the image (dark headers etc)
        st.dataframe(fraud_cases, hide_index=True, use_container_width=True)

    # Row 3: Model Performance
    st.subheader("System Performance & Trends")
    r3_1, r3_2, r3_3 = st.columns([1.5, 1, 1.5])
    
    with r3_1:
         st.markdown("##### Fraud Rate Trend")
         st.info("Trend chart displayed above.")

    with r3_2:
        st.markdown("##### Best Model Performance")
        if metrics:
            best_model = max(metrics, key=lambda x: metrics[x]['accuracy'])
            m = metrics[best_model]
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
                <h4 style="margin:0; text-align:center; color: #4f46e5;">{best_model}</h4>
                <div style="display:flex; justify-content:space-between; margin-top:10px;">
                    <div style="text-align:center">
                        <div style="font-size: 12px; color: #9ca3af;">Accuracy</div>
                        <div style="font-size: 18px; font-weight:bold;">{m['accuracy']:.1%}</div>
                    </div>
                    <div style="text-align:center">
                        <div style="font-size: 12px; color: #9ca3af;">Precision</div>
                        <div style="font-size: 18px; font-weight:bold;">{m['precision']:.1%}</div>
                    </div>
                    <div style="text-align:center">
                        <div style="font-size: 12px; color: #9ca3af;">Recall</div>
                        <div style="font-size: 18px; font-weight:bold;">{m['recall']:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with r3_3:
        st.markdown("##### Model Comparison")
        if metrics:
             data = [{"Model": n, "Acc": m['accuracy']} for n, m in metrics.items()]
             st.dataframe(pd.DataFrame(data).set_index('Model').style.format("{:.1%}"), use_container_width=True)

def render_analysis(metrics, available_models, theme="Dark"):
    st.title("Detailed Model Analysis")
    
    # Default to Best Model
    default_idx = 0
    if metrics:
        best_model_name = max(metrics, key=lambda x: metrics[x]['accuracy'])
        if best_model_name in available_models:
             default_idx = available_models.index(best_model_name)
    
    c_sel, c_up = st.columns([1, 2])
    
    with c_sel:
        st.subheader("Model Selection")
        selected_model = st.selectbox("Choose Model", available_models, index=default_idx)
        
    with c_up:
        st.subheader("Test & Validate")
        uploaded_file = st.file_uploader("Upload Transaction CSV/Parquet (Max 200MB)", type=['csv', 'parquet', 'xlsx'])
        if st.button("Load Sample Test Data"):
             st.session_state['use_sample'] = True
    
    # Consolidated Analysis View (Grid Layout)
    if selected_model and metrics:
        m_data = metrics[selected_model]
        
        st.markdown("---")
        st.subheader(f"📊 {selected_model} Comprehensive Analysis")
        
        # Grid: 2x2
        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)
        
        with g1:
            st.markdown("**Feature Importance**")
            if 'feature_importance' in m_data and m_data['feature_importance']:
                st.plotly_chart(plot_feature_importance(m_data['feature_importance'], theme=theme), use_container_width=True)
            else:
                st.info("N/A")
                
        with g2:
            st.markdown("**Confusion Matrix**")
            if 'confusion_matrix' in m_data:
                st.plotly_chart(plot_confusion_matrix(m_data['confusion_matrix'], theme=theme), use_container_width=True)
        
        with g3:
            st.markdown("**ROC Curve**")
            if 'curves' in m_data and 'roc' in m_data['curves']:
                roc = m_data['curves']['roc']
                st.plotly_chart(plot_roc_curve(roc['fpr'], roc['tpr'], roc['auc'], theme=theme), use_container_width=True)
            else:
                st.info("Waiting for training...")

        with g4:
            st.markdown("**Precision-Recall Curve**")
            if 'curves' in m_data and 'pr' in m_data['curves']:
                pr = m_data['curves']['pr']
                st.plotly_chart(plot_pr_curve(pr['recall'], pr['precision'], theme=theme), use_container_width=True)
            else:
                st.info("Waiting for training...")

    # Prediction Logic
    if (uploaded_file or st.session_state.get('use_sample')) and selected_model:
        st.markdown("### ⚡ Prediction Results")
        try:
            clean_name = selected_model.replace(" ", "_").lower()
            model_path = os.path.join(MODEL_DIR, f"{clean_name}.pkl")
            if not os.path.exists(model_path): model_path = f"models/{clean_name}.pkl"
             
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                 
                df = None
                if st.session_state.get('use_sample') and not uploaded_file:
                    from utils import load_data
                    df = load_data(limit=1000) 
                    st.info(f"Loaded {len(df)} sample rows.")
                elif uploaded_file:
                    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'): df = pd.read_parquet(uploaded_file)
                    else: df = pd.read_excel(uploaded_file)
                 
                if df is not None:
                    # 1. SHOW INPUT DATA (Explicitly requested "original dataset")
                    st.markdown("### 📄 Test Data Source")
                    st.markdown("Run analysis on the following transactions:")
                    st.dataframe(df.head(5), use_container_width=True)

                    # Preprocess
                    prep_path = os.path.join(MODEL_DIR, 'preprocessor.pkl')
                    if not os.path.exists(prep_path): prep_path = 'models/preprocessor.pkl'
                     
                    if os.path.exists(prep_path):
                        # Load RobustPreprocessor
                        from utils import RobustPreprocessor
                        preprocessor = RobustPreprocessor.load(prep_path)
                        
                        X_processed = preprocessor.transform(df)
                        
                        # Predict
                        preds = model.predict(X_processed)
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_processed)[:, 1]
                        else:
                            probs = preds
                        
                        # Prepare Results DataFrame
                        res_df = df.copy()
                        res_df['Fraud_Probability'] = probs
                        res_df['Prediction'] = preds
                        
                        # Ensure TransactionID
                        if 'TransactionID' not in res_df.columns and 'TransactionID' in res_df.index.names:
                            res_df.reset_index(inplace=True)
                        elif 'TransactionID' not in res_df.columns:
                            # If completely missing (e.g. uploaded file without it), try to use index as ID
                            res_df['TransactionID'] = res_df.index
                        
                        # Display Columns
                        display_cols = ['TransactionID', 'Fraud_Probability', 'Prediction']
                        # Filter to available
                        display_cols = [c for c in display_cols if c in res_df.columns]
                        
                        # Format
                        res_df['Fraud_Probability'] = res_df['Fraud_Probability'].map('{:.4f}'.format)

                        st.markdown("### ⚡ Prediction Results")
                        st.write(f"Analyzed {len(df)} Transactions")

                        # Highlight frauds
                        frauds = res_df[res_df['Prediction'] == 1]
                        
                        if not frauds.empty:
                            st.error(f"🚨 DETECTED {len(frauds)} FRAUDULENT TRANSACTIONS")
                            st.dataframe(frauds[display_cols], use_container_width=True)
                        else:
                            st.success("✅ No fraud detected.")
                            st.dataframe(res_df[display_cols].head(50), use_container_width=True)
                            
                    else:
                        st.error("Preprocessor not found. Please wait for training to complete.")
                else:
                    st.error("Failed to load dataframe.")
            else:
                st.error(f"Model file {model_path} not found.")
                 
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

