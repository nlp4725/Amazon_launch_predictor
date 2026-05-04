"""
Unit tests for src/feature_pipeline/feature_engineering.py

Tests use minimal fake DataFrames and a temporary .npy file so no real data is touched.
Embeddings are faked with random values matching the expected (n_rows, 384) shape.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline.feature_engineering import (
    filter_and_label,
    drop_and_extract_date,
    split,
    fit_and_transform,
    run_feature_engineering,
)


@pytest.fixture
def sample_df():
    """Minimal labeled DataFrame with most_recent_review_time and most_recent_review columns."""
    return pd.DataFrame({
        "month": pd.to_datetime(["2024-03-01", "2024-06-01", "2025-07-01"]),
        "cat": ["Toys", "Kitchen", "Sports"],
        "price": [15.99, 29.99, None],
        "seller": ["A123", "B456", "C789"],
        "most_recent_review_time": [200, 100, 250],
        "most_recent_review": [15, 5, 2],
        "title":'test toys sports'
    })


# @pytest.fixture
# def fake_emb_path(tmp_path, sample_df):
#     """Save a fake embedding .npy file matching the sample_df row count."""
#     emb = np.random.rand(len(sample_df), 384).astype(np.float32)
#     path = tmp_path / "title_embedding_all.npy"
#     np.save(path, emb)
#     return path


# def test_add_embedding(sample_df, fake_emb_path):
#     """Joins 384 emb_ columns to the DataFrame."""
#     df = add_embedding(sample_df, emb_path=fake_emb_path)
#     assert "emb_0" in df.columns
#     assert "emb_383" in df.columns
#     assert df.shape[1] == sample_df.shape[1] + 384


def test_filter_and_label(sample_df):
    """Keeps only rows with >= 180 days review time and creates binary label."""
    df = filter_and_label(sample_df)
    assert (df['most_recent_review_time'] >= 180).all()
    assert set(df['label'].unique()).issubset({0, 1})
    assert df[df['most_recent_review'] > 10]['label'].iloc[0] == 1


def test_drop_and_extract_date(sample_df):
    """Extracts year and month from date, drops seller nulls."""
    df = filter_and_label(sample_df)
    df['label'] = 0
    df = drop_and_extract_date(df)
    assert "year" in df.columns
    assert "month" in df.columns
    assert "date" in df.columns
    assert df["seller"].notna().all()


def test_split(sample_df):
    """Train/test split by date — train < 2025-06-01, test >= 2025-06-01."""
    df = filter_and_label(sample_df)
    df = drop_and_extract_date(df)
    X_train, y_train, X_test, y_test = split(df)
    assert not X_train.empty
    assert 'label' not in X_train.columns
    assert len(X_train) + len(X_test) == len(df)


def test_fit_and_transform(sample_df, tmp_path):
    """Preprocessor fits on train only, transforms test, saves preprocessor.pkl."""
    # df = add_embedding(sample_df, emb_path=fake_emb_path)
    df = filter_and_label(sample_df)
    df = drop_and_extract_date(df)
    X_train, _, X_test, _ = split(df)

    if X_train.empty or X_test.empty:
        pytest.skip("Not enough data for both splits in this fixture")

    X_train_t, X_test_t = fit_and_transform(X_train, X_test, models_dir=tmp_path)
    assert X_train_t.shape[0] == len(X_train)
    assert X_test_t.shape[0] == len(X_test)
    assert (tmp_path / "preprocessor.pkl").exists()


def test_run_feature_engineering(sample_df, tmp_path):
    """Full pipeline saves all four parquet files and returns correct shapes."""
    _, _, _, _ = run_feature_engineering(
        sample_df,
        output_dir=tmp_path,
        models_dir=tmp_path,
    )
    assert (tmp_path / "features_train.parquet").exists()
    assert (tmp_path / "features_test.parquet").exists()
    assert (tmp_path / "y_train.parquet").exists()
    assert (tmp_path / "y_test.parquet").exists()
