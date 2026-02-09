import pandas as pd
import os

def generate_create_table(csv_path, table_name, max_rows=1000):
    """
    Generates a MySQL CREATE TABLE statement from a CSV file.
    Infers types from pandas dtypes.
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return ""
        
    print(f"Reading {csv_path}...")
    # Read a sample to infer types
    df = pd.read_csv(csv_path, nrows=max_rows)
    
    ddl = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    
    col_defs = []
    for col, dtype in df.dtypes.items():
        sql_type = "TEXT"
        
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "DOUBLE"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = "DATETIME"
        
        # specific overrides/optimizations could go here
        if col == "TransactionID":
            sql_type = "INT PRIMARY KEY"
            
        col_defs.append(f"    `{col}` {sql_type}")
    
    ddl += ",\n".join(col_defs)
    ddl += "\n);\n"
    return ddl

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    sql_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sql')
    os.makedirs(sql_dir, exist_ok=True)
    
    output_file = os.path.join(sql_dir, 'schema.sql')
    
    files = {
        'train_identity.csv': 'train_identity',
        'train_transaction.csv': 'train_transaction',
        'test_identity.csv': 'test_identity',
        'test_transaction.csv': 'test_transaction'
    }
    
    with open(output_file, 'w') as f:
        for filename, table_name in files.items():
            path = os.path.join(data_dir, filename)
            ddl = generate_create_table(path, table_name)
            if ddl:
                f.write(ddl)
                f.write("\n\n")
                
    print(f"Schema saved to {output_file}")

if __name__ == "__main__":
    main()
