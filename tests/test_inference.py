"""
Unit tests for src/inference_pipeline/inference.py

Builds a minimal fitted preprocessor and model in tmp_path so no real .pkl files are needed.
fetch_product_from_keepa and predict_from_asin are tested with mocked HTTP responses
so no real Keepa API calls are made.
"""

import numpy as np
import pandas as pd
import pytest
from joblib import dump
from lightgbm import LGBMClassifier
from unittest.mock import MagicMock, patch

from src.feature_pipeline.feature_engineering import fit_and_transform
from src.inference_pipeline.inference import predict, fetch_product_from_keepa, predict_from_asin

FAKE_KEEPA_RESPONSE = {
    "products": [{
        "asin": "B08TEST123",
        "title": "yoga mat non slip",
        "categoryTree": [{"name": "Sports"}],
        "csv": [None, None, None, None, [12345, 1999]],
        "buyBoxSellerIdHistory": ["A123", "B456"],
    }]
}


@pytest.fixture
def fitted_models(tmp_path):
    """Fit a minimal preprocessor and model, save both to tmp_path."""
    X_train = pd.DataFrame({
        "price": [10.0, 20.0, 15.0, 25.0],
        "month": [1, 2, 3, 4],
        "year": [2024, 2024, 2024, 2024],
        "cat": ["Toys", "Kitchen", "Sports", "Toys"],
        "seller": ["A", "B", "C", "A"],
        "title": ["blue mat", "steel pan", "bands set", "foam roller"],
    })
    X_test = pd.DataFrame({
        "price": [18.0],
        "month": [5],
        "year": [2024],
        "cat": ["Kitchen"],
        "seller": ["B"],
        "title": ["silicone spatula"],
    })
    y_train = np.array([1, 0, 1, 0])

    X_transformed, _ = fit_and_transform(X_train, X_test, models_dir=tmp_path)

    model = LGBMClassifier(class_weight="balanced")
    model.fit(X_transformed, y_train)
    dump(model, tmp_path / "lgbm_tfidf_model.pkl")

    return tmp_path


# ---------- predict() ----------

def test_predict_returns_expected_keys(fitted_models):
    """predict() returns a dict with predicted_probability and predicted_label."""
    result = predict("yoga mat", 15.99, "Sports", models_dir=fitted_models)
    assert "predicted_probability" in result
    assert "predicted_label" in result


def test_predict_probability_in_range(fitted_models):
    """predicted_probability is between 0 and 1."""
    result = predict("yoga mat", 15.99, "Sports", models_dir=fitted_models)
    assert 0.0 <= result["predicted_probability"] <= 1.0


def test_predict_label_is_binary(fitted_models):
    """predicted_label is 0 or 1."""
    result = predict("yoga mat", 15.99, "Sports", models_dir=fitted_models)
    assert result["predicted_label"] in {0, 1}


def test_predict_with_seller(fitted_models):
    """predict() accepts an optional seller argument."""
    result = predict("yoga mat", 15.99, "Sports", seller="A123", models_dir=fitted_models)
    assert "predicted_probability" in result


# ---------- fetch_product_from_keepa() ----------

def test_fetch_product_returns_expected_keys():
    """fetch_product_from_keepa() returns title, price, cat, seller from Keepa response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_KEEPA_RESPONSE

    with patch("src.inference_pipeline.inference.requests.get", return_value=mock_resp):
        product = fetch_product_from_keepa("B08TEST123")

    assert product["title"] == "yoga mat non slip"
    assert product["cat"] == "Sports"
    assert product["price"] == 19.99
    assert product["seller"] == "B456"


def test_fetch_product_raises_on_missing_asin():
    """fetch_product_from_keepa() raises ValueError when ASIN not found."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"products": []}

    with patch("src.inference_pipeline.inference.requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="not found"):
            fetch_product_from_keepa("BADASIN")


# ---------- predict_from_asin() ----------

def test_predict_from_asin_returns_all_keys(fitted_models):
    """predict_from_asin() returns product fields and prediction keys together."""
    with patch("src.inference_pipeline.inference.fetch_product_from_keepa") as mock_fetch:
        mock_fetch.return_value = {
            "title": "yoga mat non slip",
            "price": 19.99,
            "cat": "Sports",
            "seller": "B456",
        }
        result = predict_from_asin("B08TEST123", models_dir=fitted_models)

    assert "title" in result
    assert "predicted_probability" in result
    assert "predicted_label" in result





