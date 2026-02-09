import os
import time
import json
import joblib
import sys
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from utils import load_data, RobustPreprocessor

# --- Configuration ---
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
METRICS_FILE = os.path.join(MODEL_DIR, 'metrics.json')

def train_all_models():
    # 1. Load Data
    # Using a larger limit for better training
    df = load_data(limit=50000)
    
    y = df['isFraud']

    X = df.drop(columns=['isFraud', 'TransactionID'])
    # TransactionDT is needed for RobustPreprocessor, so we keep it in X for now.
    # The preprocessor or model should handle dropping it if strictly necessary, 
    # but feature_engineering.py uses it for time diffs.
    
    # 2. Preprocess
    print("Initial X Shape:", X.shape)
    obj_cols = X.select_dtypes(include=['object']).columns.tolist()
    print("Initial Object Cols:", obj_cols)
    
    preprocessor = RobustPreprocessor()
    preprocessor.fit(X)
    print("Detected Cat Cols:", preprocessor.cat_cols)
    
    X_processed = preprocessor.transform(X)
    print("Processed X Shape:", X_processed.shape)
    remaining_obj = X_processed.select_dtypes(include=['object']).columns.tolist()
    print("Remaining Object Cols:", remaining_obj)
    
    # Validation: Store final columns for inference
    preprocessor.final_columns = X_processed.columns.tolist()
    
    # Save Preprocessor with the columns
    preprocessor.save(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Define Models
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            eval_metric='auc', random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, solver='liblinear'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    
    results = {}
    
    print("Starting Multi-Model Training...")
    
    for name, clf in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Curves
        curves = {}
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
            
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            # Downsample for JSON size
            indices = np.linspace(0, len(fpr)-1, 100, dtype=int)
            curves['roc'] = {
                'fpr': fpr[indices].tolist(),
                'tpr': tpr[indices].tolist(),
                'auc': roc_auc
            }
            
            # PR
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            # Downsample
            indices_pr = np.linspace(0, len(precision)-1, 100, dtype=int)
            curves['pr'] = {
                'precision': precision[indices_pr].tolist(),
                'recall': recall[indices_pr].tolist()
            }
        
        # Feature Importance (if available)
        feat_imp = {}
        if hasattr(clf, 'feature_importances_'):
            imp = clf.feature_importances_
            features = X_train.columns
            # Top 10
            indices = imp.argsort()[-10:][::-1]
            feat_imp = {features[i]: float(imp[i]) for i in indices}
        elif hasattr(clf, 'coef_'):
            # For Linear models
            imp = clf.coef_[0]
            features = X_train.columns
            indices = abs(imp).argsort()[-10:][::-1]
            feat_imp = {features[i]: float(imp[i]) for i in indices}
            
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "feature_importance": feat_imp,
            "curves": curves,
            "training_time": time.time() - start_time
        }
        
        print(f"  Accuracy: {acc:.4f}")
        
        # Save Model
        clean_name = name.replace(" ", "_").lower()
        joblib.dump(clf, os.path.join(MODEL_DIR, f'{clean_name}.pkl'))
        
    # Save Metrics
    with open(METRICS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Training Complete. Metrics saved to {METRICS_FILE}")

if __name__ == '__main__':
    train_all_models()
