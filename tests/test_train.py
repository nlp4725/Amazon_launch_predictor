"""
Unit tests for src/training_pipeline/train.py

X_train is a random numpy array — by the time data reaches train.py it has already
been encoded by ColumnTransformer, so no column names or categories are needed.
"""

import numpy as np
import pytest

from src.training_pipeline.train import train


@pytest.fixture
def training_data():
    """Fake encoded features and binary labels matching expected input shape."""
    X_train = np.random.rand(100, 500)
    y_train = np.random.randint(0, 2, 100)
    return X_train, y_train


def test_train_saves_model(tmp_path, training_data):
    """train() fits model and saves lgbm_tfidf_model.pkl to models_dir."""
    X_train, y_train = training_data
    model = train(X_train, y_train, models_dir=tmp_path)
    assert (tmp_path / "lgbm_tfidf_model.pkl").exists()
    assert model is not None
