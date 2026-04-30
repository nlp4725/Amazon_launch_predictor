'''
Load 
'''

import pandas as pd
from pathlib import Path
import sqlite3

DATA_DIR=Path('data/raw') # output dirctory

def load_data(
    raw_path: str ='data/raw/product_launch.db',
    output_dir: Path | str =DATA_DIR
):  
    # open connection, query and then close
    with sqlite3.connect(raw_path) as conn:
        df=pd.read_sql("SELECT * FROM products",conn)


    outdir = Path(output_dir) # convert output path to a Path object
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "product_launch.parquet", index=False)
    print(f"Loaded {df.shape[0]} rows")
    return df

if __name__=="__main__":
    load_data()
    