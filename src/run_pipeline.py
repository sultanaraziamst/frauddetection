import os
import subprocess
import time
import sys

def run_step(description, command, cwd=None):
    print(f"\n[PIPELINE] Starting: {description}...")
    try:
        if cwd:
            subprocess.run(command, cwd=cwd, check=True, shell=True)
        else:
            subprocess.run(command, check=True, shell=True)
        print(f"[PIPELINE] ✅ Completed: {description}")
    except subprocess.CalledProcessError as e:
        print(f"[PIPELINE] ❌ Failed: {description}")
        print(f"Error: {e}")
        sys.exit(1)

def main():
    print("="*60)
    print("🚀 FRAUD DETECTION SYSTEM: FULL PIPELINE")
    print("="*60)
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # 1. Database & Ingestion Setup
    # Note: Assuming database is running. 
    # If users want to reload data, they can uncomment the line below.
    # run_step("Data Ingestion (CSV to SQL)", f"python {os.path.join(base_dir, 'csv_to_sql.py')}")

    # 2. Feature Engineering & Model Training
    # This script (train_multi.py) handles:
    # - Loading data from SQL
    # - Feature Engineering (via utils.FraudPreprocessor)
    # - Training Models (XGBoost, LightGBM, etc.)
    # - Saving Metrics
    run_step("Feature Engineering & Model Training", f"python {os.path.join(base_dir, 'train_multi.py')}")
    
    # 3. Dashboard Launch
    print("\n[PIPELINE] 📊 Launching Dashboard...")
    dashboard_path = os.path.join(base_dir, 'app.py')
    
    # Using sys.executable to ensure we use the same python env
    subprocess.run(f"streamlit run {dashboard_path}", shell=True)

if __name__ == "__main__":
    main()
