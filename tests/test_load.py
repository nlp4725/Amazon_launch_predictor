"""
Unit tests for src/feature_pipeline/load.py

Tests use a temporary SQLite database so the real product_launch.db is never touched.
tmp_path is a pytest built-in fixture that provides a fresh temporary directory per test.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.feature_pipeline.load import load_data


def test_load_data_returns_dataframe(tmp_path):
    """load_data() reads from SQLite, saves parquet, and returns a non-empty DataFrame."""
    db_path = tmp_path / "test.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE products(asin TEXT, month TEXT, title TEXT, cat TEXT, raw_data TEXT)"
        )
        conn.execute(
            "INSERT INTO products VALUES('A001', '2024-03-01', 'Test Product', 'Toys', '{}')"
        )

    df = load_data(raw_path=str(db_path), output_dir=tmp_path)

    assert not df.empty
    assert "asin" in df.columns
    assert (tmp_path / "product_launch.parquet").exists()





