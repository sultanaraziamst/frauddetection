import os
import json
import joblib
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

# --- Configuration ---
CONFIG_FILE = 'config/config.json'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Config & Connect to DB ---
def get_db_engine():
    if not os.path.exists(CONFIG_FILE):
         # Try going up one level
         config_path = os.path.join('..', CONFIG_FILE)
    else:
        config_path = CONFIG_FILE

    with open(config_path, 'r') as f:
        config = json.load(f)['db']
    
    conn_str = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    return create_engine(conn_str)

# --- Fetch Data ---
def fetch_data(engine, limit=None):
    print("Fetching training data from database...")
    query_identity = "SELECT * FROM train_identity"
    query_transaction = "SELECT * FROM train_transaction"
    
    if limit:
        query_identity += f" LIMIT {limit}"
        query_transaction += f" LIMIT {limit}"
        
    df_identity = pd.read_sql(query_identity, engine)
    df_transaction = pd.read_sql(query_transaction, engine)
    
    return df_transaction, df_identity

# --- Feature Engineering ---
def preprocess_data(df_trans, df_id):
    print("Preprocessing data...")
    # Merge on TransactionID
    df = df_trans.merge(df_id, on='TransactionID', how='left')
    
    # Target variable
    y = df['isFraud']
    X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])

    # Handle Categorical Columns
    # Identify objects and encode them
    # For a real system, we'd need to save these encoders to apply to new data.
    # For simplicity here, we'll use LabelEncoding and handle unseen labels in the app by default/try-except
    
    cat_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        # Convert to string to handle mixed types/NaNs
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Fill NaNs with -999 (XGBoost handles missing values, but simple filling specific to unknown)
    X = X.fillna(-999)
    
    return X, y, encoders

# --- Train Model ---
def train_and_save_model():
    engine = get_db_engine()
    
    # Use a limit for faster iteration during dev, remove limit for full train
    # Warning: The full dataset is large, might run into memory info if not careful.
    # We will use a subset for demonstration/speed.
    df_trans, df_id = fetch_data(engine, limit=50000) 
    
    X, y, encoders = preprocess_data(df_trans, df_id)
    
    print(f"Data shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training XGBoost model...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model ID: XGBoost")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save artifacts
    model_path = os.path.join(MODEL_DIR, 'xgb_fraud_model.pkl')
    encoders_path = os.path.join(MODEL_DIR, 'encoders.pkl')
    
    joblib.dump(clf, model_path)
    joblib.dump(encoders, encoders_path)
    
    print(f"Model saved to {model_path}")
    print(f"Encoders saved to {encoders_path}")

if __name__ == '__main__':
    train_and_save_model()
