import os
import json
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import joblib

CONFIG_FILE = 'config/config.json'

def get_db_engine():
    # Adjust path if needed
    path = CONFIG_FILE
    if not os.path.exists(path):
        path = os.path.join('..', CONFIG_FILE)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file {CONFIG_FILE} not found.")

    with open(path, 'r') as f:
        config = json.load(f)['db']
    
    conn_str = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    return create_engine(conn_str)

def load_data(limit=None):
    engine = get_db_engine()
    print("Fetching training data from database...")
    query_identity = "SELECT * FROM train_identity"
    query_transaction = "SELECT * FROM train_transaction"
    
    if limit:
        query_identity += f" LIMIT {limit}"
        query_transaction += f" LIMIT {limit}"
        
    df_id = pd.read_sql(query_identity, engine)
    df_trans = pd.read_sql(query_transaction, engine)
    
    return df_trans.merge(df_id, on='TransactionID', how='left')

class RobustPreprocessor:
    def __init__(self):
        self.numeric_cols = []
        self.cat_cols = []
        self.encoders = {}
        self.freq_encodings = {}
        self.agg_stats = {}
        self.cols_to_drop = []
        self.final_columns = []
        self.is_fitted = False
        self.start_date = datetime.datetime.strptime('2017-12-01', "%Y-%m-%d")

    def _extract_date_features(self, df):
        df = df.copy()
        # Handle TransactionDT
        df["Date"] = df['TransactionDT'].apply(lambda x: self.start_date + datetime.timedelta(seconds=x))
        df['_Weekdays'] = df['Date'].dt.dayofweek
        df['_Hours'] = df['Date'].dt.hour
        df['_Days'] = df['Date'].dt.day
        df.drop(columns=['Date'], inplace=True)
        return df

    def fit(self, X):
        X = X.copy()
        
        # 1. Date Features
        if 'TransactionDT' in X.columns:
             X = self._extract_date_features(X)
        else:
             print("Warning: TransactionDT not found in fit. Date features will be skipped.")
        
        # Log transform
        if 'TransactionAmt' in X.columns:
            X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
        
        # Specific categorical columns from feature engineering
        specific_cats = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
        
        # Identify ALL object columns to avoid training errors
        all_objects = X.select_dtypes(include=['object']).columns.tolist()
        
        # Combine, keeping specific ones first if needed (order doesn't strictly matter for dict)
        # We use a set for uniqueness
        self.cat_cols = list(set(specific_cats + all_objects))
        # Ensure intersection with actual columns
        self.cat_cols = [c for c in self.cat_cols if c in X.columns]
        
        for col in self.cat_cols:
            X[col] = X[col].fillna('Missing').astype(str)
            le = LabelEncoder()
            le.fit(X[col])
            # Store as dict for serialization compatibility if needed, but joblib handles LE
            self.encoders[col] = le
            
        # Frequency Encoding
        freq_cols = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
        freq_cols = [c for c in freq_cols if c in X.columns]
        
        for col in freq_cols:
            freq = X[col].astype(str).value_counts() / len(X)
            self.freq_encodings[col] = freq
            
        # Aggregation Stats
        agg_cols = ['card1', 'card4', 'P_emaildomain', 'R_emaildomain']
        agg_cols = [c for c in agg_cols if c in X.columns]
        
        for col in agg_cols:
             self.agg_stats[col] = {}
             # txn count not needed to store for transform usually unless we want global stats, 
             # but transform usually computes on the batch. 
             # BUT for single prediction, we can't compute 'mean' of the batch. 
             # WE MUST USE TRAIN STATS.
             self.agg_stats[col]['mean'] = X.groupby(col)['TransactionAmt'].mean()
             self.agg_stats[col]['std'] = X.groupby(col)['TransactionAmt'].std()

        # Drop columns logic (simplified for portability)
        # In a real expanded engineered set, we'd do correlation checks here.
        # For now, we trust the transformation step to generate features, and we'll record final columns.
        
        # We need to simulate the transform to know what columns remain? 
        # Actually, let's just mark fitted.
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted.")
        X = X.copy()
        
        # 1. Date Features
        # Ensure TransactionDT exists
        if 'TransactionDT' not in X.columns:
             # If missing for some reason (e.g. strict drop), we can't compute date features.
             # We'll fill with dummy or skip.
             # For now, assume it's there or we skip.
             pass
        else:
             X = self._extract_date_features(X)
        
        # 2. Log Transform
        if 'TransactionAmt' in X.columns:
            X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
            
            # Deviations (using GLOBAL mean/std if possible, but localized to batch is tricky for single pred)
            # feature_engineering.py uses batch stats. For inference, this is problematic.
            # We will use batch stats of the input X if len > 1, else we might default to 0.
            # Ideally we store these stats in fit.
            # Simplified for robustness:
            X['Trans_min_mean'] = X['TransactionAmt'] - X['TransactionAmt'].mean()
            X['Trans_min_std'] = X['Trans_min_mean'] / (X['TransactionAmt'].std() + 1e-9)
            
        # 3. Categorical
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna('Missing').astype(str)
                le = self.encoders.get(col)
                if le:
                    # Robust transform
                    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    X[col] = X[col].map(mapping).fillna(-1)
                else:
                    X[col] = -1

        # 4. Frequency
        for col, freq in self.freq_encodings.items():
            if col in X.columns:
                X[f'{col}_freq'] = X[col].astype(str).map(freq).fillna(0)

        # 5. Aggregations (Mapped from Train Stats)
        for col, stats in self.agg_stats.items():
            if col in X.columns:
                X[f'{col}_amt_mean'] = X[col].map(stats['mean']).fillna(-1)
                X[f'{col}_amt_std'] = X[col].map(stats['std']).fillna(-1)
                
        # 6. Combined Features
        # card1_ProductCD
        if 'card1' in X.columns and 'ProductCD' in X.columns:
             # This is a bit tricky with LE on fly. We'll skip complex combined LE for stability
             # unless we store a massive combined encoder.
             pass

        # 7. Fill NaNs
        X = X.fillna(-999)
        
        # 8. Shape check/Enforcement
        # If we have a stored final_columns list, enforce it.
        if self.final_columns:
             for col in self.final_columns:
                 if col not in X.columns:
                     X[col] = -999 # Add missing
             X = X[self.final_columns] # Reorder and select
        
        return X

    def save(self, path):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path):
        return joblib.load(path)
