"""
Unit tests for src/training_pipeline/evaluate.py

Trains a dummy LGBMClassifier in-memory and passes it directly to evaluate()
so no .pkl file is needed.
"""

import numpy as np
import pytest
from lightgbm import LGBMClassifier

from src.training_pipeline.evaluate import evaluate

EXPECTED_KEYS = {"roc_auc", "pr_auc", "precision", "recall", "f1"}


@pytest.fixture
def trained_model():
    """Fit a small LGBMClassifier on random data — just needs to be a valid model object."""
    X = np.random.rand(200, 10)
    y = np.random.randint(0, 2, 200)
    model = LGBMClassifier(class_weight="balanced")
    model.fit(X, y)
    return model


@pytest.fixture
def test_data():
    """Random X_test and y_test matching the shape trained_model expects."""
    X_test = np.random.rand(50, 10)
    y_test = np.random.randint(0, 2, 50)
    return X_test, y_test


def test_evaluate_returns_correct_keys(trained_model, test_data, tmp_path):
    """evaluate() returns a dict with all expected metric keys."""
    X_test, y_test = test_data
    metrics = evaluate(X_test, y_test, model=trained_model, output_dir=tmp_path)
    assert set(metrics.keys()) == EXPECTED_KEYS


def test_evaluate_metrics_in_range(trained_model, test_data, tmp_path):
    """All returned metric values are between 0 and 1."""
    X_test, y_test = test_data
    metrics = evaluate(X_test, y_test, model=trained_model, output_dir=tmp_path)
    for name, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{name} out of range: {val}"


def test_evaluate_saves_parquet(trained_model, test_data, tmp_path):
    """evaluate() saves evaluation_results.parquet to output_dir."""
    X_test, y_test = test_data
    evaluate(X_test, y_test, model=trained_model, output_dir=tmp_path)
    assert (tmp_path / "evaluation_results.parquet").exists()
