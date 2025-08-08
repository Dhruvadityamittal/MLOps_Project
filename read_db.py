import sqlite3
import pandas as pd

db_path = "mlflow.db"
conn = sqlite3.connect(db_path)

# Read experiments table
df = pd.read_sql_query("SELECT * FROM experiments;", conn)
print(df)

# Read runs table
runs_df = pd.read_sql_query("SELECT * FROM runs;", conn)
print(runs_df)

conn.close()