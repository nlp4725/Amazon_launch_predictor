"""
Train LGBMClassifier on feature-engineered data and save the model to disk.

- Reads features_train.parquet and y_train.parquet from data/processed/
- Fits LGBMClassifier with class_weight='balanced' to handle class imbalance (~12% success rate)
- Saves trained model to models/lgbm_embedding_model.pkl
"""

from pathlib import Path

import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_dir: Path | str = MODELS_DIR,
):
    """
    Fit LGBMClassifier and save to disk.

    In: X_train (features), y_train (binary labels), models_dir for saving model
    Out: fitted model saved to models/lgbm_embedding_model.pkl
    """
    model = LGBMClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    outdir = Path(models_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    dump(model, outdir / "lgbm_tfidf_model.pkl")
    print(f"Model saved to {outdir / 'lgbm_tfidf_model.pkl'}")

    return model


if __name__ == "__main__":
    X_train = pd.read_parquet(PROCESSED_DIR / "features_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").squeeze()
    train(X_train, y_train)

    
