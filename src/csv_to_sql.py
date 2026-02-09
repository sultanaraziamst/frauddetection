import json
import os
import pandas as pd
from sqlalchemy import create_engine, text
import pymysql

# --- Load DB Config ---
def load_config(config_file='config/config.json'):
    # Adjust path if running from src or root
    if not os.path.exists(config_file):
        # Try going up one level if file not found (e.g., if running from src)
        config_file = os.path.join('..', config_file)
    
    if not os.path.exists(config_file):
         raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return config['db']

# --- Create Database if Not Exists ---
def create_database(config):
    """
    Connects to MySQL server (without DB) and creates the database if it doesn't exist.
    """
    conn_str = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}"
    engine = create_engine(conn_str)
    
    db_name = config['database']
    try:
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            print(f"Database '{db_name}' checked/created.")
    except Exception as e:
        print(f"Error creating database: {e}")
        raise

# --- Get Engine for Database ---
def get_db_engine(config):
    conn_str = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    engine = create_engine(conn_str, echo=False)
    return engine

# --- Load CSV into SQL ---
def load_all_data(data_dir='data'):
    # Adjust path if running from src
    if not os.path.exists(data_dir):
        data_dir = os.path.join('..', data_dir)

    config = load_config()
    create_database(config)
    engine = get_db_engine(config)

    files_to_load = [
        'train_identity.csv',
        'train_transaction.csv',
        'test_identity.csv',
        'test_transaction.csv'
    ]

    for file_name in files_to_load:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        table_name = os.path.splitext(file_name)[0]
        print(f"Loading {file_name} into table '{table_name}'...")
        
        try:
            # Load in chunks to avoid memory issues with large files
            chunksize = 10000 
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
                if_exists = 'replace' if i == 0 else 'append'
                chunk.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
                if i % 10 == 0:
                    print(f"  Processed {i * chunksize} rows...")
            
            print(f"Successfully loaded {file_name}!")
            
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

if __name__ == '__main__':
    print("Starting data loading process...")
    load_all_data()
    print("Data loading complete.")
