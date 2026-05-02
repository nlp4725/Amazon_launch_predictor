"""
Unit tests for src/feature_pipeline/preprocessing.py

Tests use a minimal fake DataFrame with realistic JSON strings in raw_data and buybox
so the real database is never touched.
"""

import json
import pandas as pd
import pytest

from src.feature_pipeline.preprocessing import (
    add_price_seller_title,
    add_review_n_monthly_sold,
    run_preprocess,
)


@pytest.fixture
def sample_df():
    """Minimal fake DataFrame with realistic raw_data and buybox JSON."""
    raw = json.dumps({
        "listedSince": 7000000,
        "csv": [None, None, None, None, [None, 1500]],
        "title": "Test Product",
        "monthlySoldHistory": [7000000, 100, 7001440, 120],
        "reviews": {"reviewCount": [7000000, 5, 7100000, 15]},
        "rootCategory": 123,
        "salesRanks": {"123": [7000000, 5000, 7100000, 4000]},
    })
    buybox = json.dumps({
        "buyBoxSellerIdHistory": ["seller1", "A123XYZ"]
    })
    return pd.DataFrame({
        "asin": ["A001"],
        "month": ["2024-03-01"],
        "cat": ["Toys"],
        "raw_data": [raw],
        "buybox": [buybox],
        "rating": [None],
    })


def test_add_price_seller_title(sample_df):
    """Extracts price, seller, and title from JSON fields."""
    df = add_price_seller_title(sample_df)
    assert "price" in df.columns
    assert "seller" in df.columns
    assert "title" in df.columns
    assert df["title"].iloc[0] == "Test Product"
    assert df["seller"].iloc[0] == "A123XYZ"


def test_add_review_n_monthly_sold(sample_df):
    """Extracts review and monthly sold signals from raw_data."""
    df = add_price_seller_title(sample_df)
    df = add_review_n_monthly_sold(df)
    assert "most_recent_review" in df.columns
    assert "most_recent_review_time" in df.columns
    assert df["most_recent_review"].iloc[0] == 15


def test_run_preprocess(sample_df, tmp_path):
    """Full preprocessing pipeline returns DataFrame with all expected columns and saves parquet."""
    df = run_preprocess(sample_df, output_dir=tmp_path)
    assert "price" in df.columns
    assert "most_recent_review_time" in df.columns
    assert not df.empty
    assert (tmp_path / "preprocessed.parquet").exists()
