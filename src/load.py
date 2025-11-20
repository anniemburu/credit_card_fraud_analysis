import duckdb
from pathlib import Path

def load():
    data_dir = Path("data")
    raw_data_dir = data_dir / "raw/creditcard_fraud_raw.parquet"

    DB = data_dir/ "rs_warehouse.duckdb"

    conn = duckdb.connect(str(DB))
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS road_safety_train_raw AS SELECT * FROM read_parquet('{raw_data_dir}')"
    )
    print(f"Loaded raw data into DuckDB: {DB}")
