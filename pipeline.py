"""
Full ML pipeline: load → preprocess → feature engineering → train → evaluate.

Run this to retrain the model from scratch:
    python pipeline.py
"""

from pathlib import Path

import pandas as pd

from src.feature_pipeline.load import load_data
from src.feature_pipeline.preprocessing import run_preprocess
from src.feature_pipeline.feature_engineering import run_feature_engineering
from src.training_pipeline.train import train
from src.training_pipeline.evaluate import evaluate

RAW_PARQUET = Path("data/raw/product_launch.parquet")
PREPROCESSED_PARQUET = Path("data/processed/preprocessed.parquet")


def run_pipeline():
    print("=== Step 1: Load ===")
    if RAW_PARQUET.exists():
        print("Parquet found — skipping SQLite load")
        df = pd.read_parquet(RAW_PARQUET)
    else:
        df = load_data()

    print("\n=== Step 2: Preprocess ===")
    if PREPROCESSED_PARQUET.exists():
        print("Preprocessed parquet found — skipping preprocessing")
        df = pd.read_parquet(PREPROCESSED_PARQUET)
    else:
        df = run_preprocess(df)

    print("\n=== Step 3: Feature Engineering ===")
    X_train, y_train, X_test, y_test = run_feature_engineering(df)

    print("\n=== Step 4: Train ===")
    model = train(X_train, y_train)

    print("\n=== Step 5: Evaluate ===")
    metrics = evaluate(X_test, y_test, model=model)

    print("\n=== Pipeline complete ===")
    return metrics


if __name__ == "__main__":
    run_pipeline()
