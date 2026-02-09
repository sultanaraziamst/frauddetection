# Fraud Detection System

## Overview
This project constructs a fraud detection pipeline including data storage in MySQL, machine learning model training (XGBoost, Random Forest, etc.), and an interactive Streamlit dashboard for analysis.

## Structure
- `src/`: Source code
  - `train_model.py`: **Feature Engineering** and Model Training logic.
  - `utils.py`: **Feature Engineering** helper classes (`FraudPreprocessor`).
  - `app.py`: Streamlit Dashboard application.
  - `csv_to_sql.py`: Data ingestion script.
- `sql/`: SQL scripts (schema and DDL).
- `data/`: Dataset CSVs.
- `models/`: Saved models and metrics.
- `config/`: Configuration files.

## Feature Engineering
Feature engineering logic is located in:
1.  **`src/utils.py`**: The `FraudPreprocessor` class handles categorical encoding (Label Encoding), missing value imputation (-999), and unifying schema between train/test.
2.  **`src/train_model.py`**: Contains the `preprocess_data` function that merges Identity and Transaction tables and applies the preprocessor.

## Pipeline Execution

We have provided a unified pipeline script to handle feature engineering, training, and dashboard launch.

### Run Full Pipeline
```bash
python src/run_pipeline.py
```
This command will:
1.  **Extract & Transform**: Load data and apply `FraudPreprocessor` (Feature Engineering).
2.  **Train**: Train XGBoost, LightGBM, Random Forest, etc.
3.  **Visualize**: Launch the Streamlit Dashboard.

### Individual Steps

**1. Data Ingestion (Optional)**
```bash
python src/csv_to_sql.py
```

**2. Training**
```bash
python src/train_multi.py
```

**3. Dashboard**
```bash
streamlit run src/app.py
```

