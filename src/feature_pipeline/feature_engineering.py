"""
Feature engineering: label, filter, split, encode, and join embeddings.

- Joins title embeddings first to preserve index alignment with the full dataset
- Filters to ASINs with at least 180 days of review data and creates binary label
- Splits into train/test by date (train < 2025-06-01, test >= 2025-06-01)
- Fits ColumnTransformer on train only and applies to test (prevents leakage)
- Saves features_train.parquet, features_test.parquet, y_train.parquet, y_test.parquet, and preprocessor.pkl
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
EMB_PATH = Path("data/raw/title_embedding_all.npy")


# ---------- main functions ----------

def add_embedding(df: pd.DataFrame, emb_path: Path | str = EMB_PATH) -> pd.DataFrame:
    """
    Load title embeddings from .npy and join as 384 emb_ columns.
    Must run before any row filtering to keep index aligned with the embedding array.

    In: full preprocessed df, path to title_embedding_all.npy
    Out: df with emb_0 ... emb_383 columns joined
    """
    embeddings = np.load(emb_path)
    em_cols = [f'emb_{i}' for i in range(384)]
    df_embedding = pd.DataFrame(embeddings, index=df.index, columns=em_cols)
    return df.join(df_embedding)


def filter_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to ASINs with >= 180 days of review data and create binary success label.

    In: df with most_recent_review_time and most_recent_review columns
    Out: filtered df with label column (1 = success: >10 reviews, 0 = failure)
    """
    df = df[df['most_recent_review_time'] >= 180].copy()
    df['label'] = np.where(df['most_recent_review'] > 10, 1, 0)
    return df


def drop_and_extract_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select model features, drop rows missing seller, and extract year/month from date.

    In: labeled df
    Out: df with columns [month, year, date, cat, price, seller, emb_*, label]
    """
    em_cols = [c for c in df.columns if c.startswith('emb_')]
    df = df[['month', 'cat', 'price', 'seller', 'label'] + em_cols].copy()
    df = df.dropna(subset=['seller'])
    df['date'] = df['month']
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df


def split(df: pd.DataFrame):
    """
    Time-based train/test split. Train < 2025-06-01, test >= 2025-06-01.

    In: df with date and label columns
    Out: X_train, y_train, X_test, y_test
    """
    train = df[df['date'] < '2025-06-01']
    test = df[df['date'] >= '2025-06-01']

    features = [c for c in df.columns if c != 'label']
    X_train = train[features]
    y_train = train['label']
    X_test = test[features]
    y_test = test['label']

    return X_train, y_train, X_test, y_test


def fit_and_transform(X_train: pd.DataFrame, X_test: pd.DataFrame, models_dir: Path | str = MODELS_DIR):
    """
    Fit ColumnTransformer on train only, apply to test, save fitted preprocessor.
    fit_transform on train, transform only on test — prevents leakage.

    In: X_train, X_test DataFrames, models_dir for saving preprocessor.pkl
    Out: X_train_transformed (np.array), X_test_transformed (np.array)
    """
    em_cols = [c for c in X_train.columns if c.startswith('emb_')]

    cat_transformer = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), ['price', 'month', 'year'] + em_cols),
        ('cat', cat_transformer, ['cat', 'seller']),
    ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    outdir = Path(models_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    dump(preprocessor, outdir / "preprocessor.pkl")
    print(f"Preprocessor saved to {outdir / 'preprocessor.pkl'}")

    return X_train_transformed, X_test_transformed


def run_feature_engineering(df: pd.DataFrame, emb_path: Path | str = EMB_PATH, output_dir: Path | str = PROCESSED_DIR, models_dir: Path | str = MODELS_DIR) -> tuple:
    """
    Run full feature engineering pipeline and save outputs to disk.

    In: clean df from run_preprocess()
    Out: X_train, y_train, X_test, y_test as numpy arrays
    """
    df = add_embedding(df,emb_path)
    df = filter_and_label(df)
    df = drop_and_extract_date(df)
    X_train, y_train, X_test, y_test = split(df)
    X_train_transformed, X_test_transformed = fit_and_transform(X_train, X_test,models_dir)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train_transformed).to_parquet(outdir / "features_train.parquet", index=False)
    pd.DataFrame(X_test_transformed).to_parquet(outdir / "features_test.parquet", index=False)
    y_train.to_frame().to_parquet(outdir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(outdir / "y_test.parquet", index=False)

    print(f"Train: {X_train_transformed.shape}, Test: {X_test_transformed.shape}")
    return X_train_transformed, y_train, X_test_transformed, y_test


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/preprocessed.parquet")
    run_feature_engineering(df)
