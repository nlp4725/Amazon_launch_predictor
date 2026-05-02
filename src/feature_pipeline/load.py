"""
Load raw product data from SQLite and save a local parquet checkpoint.

- Reads the products table from product_launch.db
- Saves a parquet copy to data/raw/ for faster re-runs
- Returns the DataFrame for in-memory pipeline chaining
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/raw/product_launch.db")
RAW_DIR = Path("data/raw")


def load_data(
    raw_path: Path | str = DB_PATH,
    output_dir: Path | str = RAW_DIR,
) -> pd.DataFrame:
    with sqlite3.connect(raw_path) as conn:
        df = pd.read_sql("SELECT * FROM products", conn)

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "product_launch.parquet", index=False)

    print(f"Loaded {df.shape[0]} rows from products table.")
    return df


if __name__ == "__main__":
    load_data()
    