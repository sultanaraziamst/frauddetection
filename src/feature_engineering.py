# feature_engineering.py

import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder

def feature_engineering(train_transaction, train_identity, test_transaction, test_identity):
    """
    Perform feature engineering on the transaction and identity datasets.

    Returns:
        df_train, df_test: preprocessed and feature-engineered train and test DataFrames
    """
    # Merge identity data
    df_train = train_transaction.merge(train_identity, on='TransactionID', how='left')
    df_test = test_transaction.merge(test_identity, on='TransactionID', how='left')
    
    # Log-transform TransactionAmt
    for df in [df_train, df_test]:
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])

    # TransactionAmt deviations
    for df in [df_train, df_test]:
        df['Trans_min_mean'] = df['TransactionAmt'] - df['TransactionAmt'].mean()
        df['Trans_min_std'] = df['Trans_min_mean'] / df['TransactionAmt'].std()
        df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby('card1')['TransactionAmt'].transform('mean')
        df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / df.groupby('card1')['TransactionAmt'].transform('std')
        df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby('card4')['TransactionAmt'].transform('mean')
        df['TransactionAmt_to_std_card4'] = df['TransactionAmt'] / df.groupby('card4')['TransactionAmt'].transform('std')

    # Extract date features
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    for df in [df_train, df_test]:
        df["Date"] = df['TransactionDT'].apply(lambda x: startdate + datetime.timedelta(seconds=x))
        df['_Weekdays'] = df['Date'].dt.dayofweek
        df['_Hours'] = df['Date'].dt.hour
        df['_Days'] = df['Date'].dt.day

    # Label encoding for categorical columns
    categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
    for df in [df_train, df_test]:
        for col in categorical_cols:
            df[col] = df[col].fillna('Missing')
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Frequency encoding
    freq_cols = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
    for col in freq_cols:
        col_values_train = df_train[col].astype(str)
        col_values_test = df_test[col].astype(str)
        freq = col_values_train.value_counts() / len(df_train)
        df_train[f'{col}_freq'] = col_values_train.map(freq)
        df_test[f'{col}_freq'] = col_values_test.map(freq)

    # Aggregation features
    agg_cols = ['card1', 'card4', 'P_emaildomain', 'R_emaildomain']
    for col in agg_cols:
        for df in [df_train, df_test]:
            df[f'{col}_txn_count'] = df.groupby(col)['TransactionID'].transform('count')
            df[f'{col}_amt_mean'] = df.groupby(col)['TransactionAmt'].transform('mean')
            df[f'{col}_amt_std'] = df.groupby(col)['TransactionAmt'].transform('std')

    # Time difference features
    for df in [df_train, df_test]:
        df.sort_values(['card1', 'TransactionDT'], inplace=True)
        df['card1_time_diff'] = df.groupby('card1')['TransactionDT'].diff().fillna(-1)
        df['card1_time_diff_log'] = np.log1p(df['card1_time_diff'])

    # Combined features
    for df in [df_train, df_test]:
        df['card1_ProductCD'] = LabelEncoder().fit_transform(df['card1'].astype(str) + '_' + df['ProductCD'].astype(str))
        df['card4_email'] = LabelEncoder().fit_transform(df['card4'].astype(str) + '_' + df['P_emaildomain'].astype(str))

    # Drop low variance columns
    target_col = 'isFraud'
    low_variance_cols = []
    for col in df_train.columns:
        if col == target_col:
            continue
        class_dist = df_train[col].value_counts(normalize=True, dropna=False) * 100
        if class_dist.max() > 95:
            low_variance_cols.append(col)
    df_train.drop(columns=low_variance_cols, inplace=True)
    df_test.drop(columns=low_variance_cols, inplace=True)

    # Drop columns with >90% missing
    missing_threshold = 0.9
    missing_ratio_train = df_train.drop(columns=[target_col]).isnull().mean()
    missing_ratio_test = df_test.drop(columns=[target_col], errors='ignore').isnull().mean()
    cols_to_drop = list(set(missing_ratio_train[missing_ratio_train > missing_threshold].index.tolist() +
                            missing_ratio_test[missing_ratio_test > missing_threshold].index.tolist()))
    df_train.drop(columns=cols_to_drop, inplace=True)
    df_test.drop(columns=cols_to_drop, inplace=True)

    # Drop duplicate columns
    duplicate_cols = []
    cols = df_train.columns
    for i in range(len(cols)):
        col1 = cols[i]
        for j in range(i + 1, len(cols)):
            col2 = cols[j]
            if df_train[col1].equals(df_train[col2]):
                duplicate_cols.append(col2)
    df_train.drop(columns=duplicate_cols, inplace=True)
    df_test.drop(columns=duplicate_cols, inplace=True)

    # Drop highly correlated columns
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    corr_matrix = df_train[numeric_cols].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlation_threshold = 0.98
    high_corr_cols = [col for col in upper_triangle.columns if any(upper_triangle[col] > correlation_threshold)]
    df_train.drop(columns=high_corr_cols, inplace=True)
    df_test.drop(columns=high_corr_cols, inplace=True)

    return df_train, df_test
