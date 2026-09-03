"""
Inference: predict launch success for a new product from user inputs or ASIN.

- predict(): takes title, price, cat directly from the user
- known_categories() / title_vocab_hits(): what the fitted preprocessor actually recognises
- predict_from_asin(): queries Keepa for the ASIN, extracts fields, calls predict()
- Defaults seller to "unknown" and month/year to today (new launch)
- Loads fitted preprocessor.pkl and lgbm_tfidf_model.pkl from models/
- Returns predicted_probability and predicted_label
"""

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from joblib import load

load_dotenv()
KEEPA_API_KEY = os.getenv("KEEPA_API_KEY")

MODELS_DIR = Path("src/serving/model")
THRESHOLD = 0.6


@lru_cache(maxsize=None)
def _fitted_preprocessor(models_dir: str):
    """Cached load of preprocessor.pkl — the introspection helpers below hit it per request."""
    return load(Path(models_dir) / "preprocessor.pkl")


def _step(models_dir: str | Path, name: str):
    transformers = {n: t for n, t, _ in _fitted_preprocessor(str(models_dir)).transformers_}
    return transformers[name]


def known_categories(models_dir: Path | str = MODELS_DIR) -> list[str]:
    """
    The category values the fitted encoder recognises.

    Out: sorted list of category names, excluding the "unknown" imputation bucket.
    Anything not in this list one-hots to all zeros (handle_unknown="ignore"), so the
    model silently ignores it rather than raising.
    """
    encoder = _step(models_dir, "cat").named_steps["encoder"]
    return sorted(c for c in encoder.categories_[0] if c != "unknown")


def title_vocab_hits(title: str, models_dir: Path | str = MODELS_DIR) -> list[str]:
    """
    The TF-IDF vocabulary terms a title matches.

    Out: sorted list of matched terms. An empty list means the title contributes
    nothing to the prediction — the vocabulary is capped at 200 terms, so most
    words fall outside it and the model is left reading price and category alone.
    """
    tfidf = _step(models_dir, "tfidf")
    inverse = {index: term for term, index in tfidf.vocabulary_.items()}
    return sorted(inverse[i] for i in tfidf.transform([title]).nonzero()[1])


def predict(
    title: str,
    price: float,
    cat: str,
    seller: str = "unknown",
    models_dir: Path | str = MODELS_DIR,
) -> dict:
    """
    Predict launch success for a new product.

    In: title (str), price (float), cat (str), seller (str, defaults to "unknown")
    Out: dict with predicted_probability (float) and predicted_label (0 or 1)
    """
    models_dir = Path(models_dir)
    preprocessor = load(models_dir / "preprocessor.pkl")
    model = load(models_dir / "lgbm_tfidf_model.pkl")

    now = datetime.today()
    row = pd.DataFrame([{
        "title": title,
        "price": price,
        "cat": cat,
        "seller": seller or "unknown",
        "month": now.month,
        "year": now.year,
    }])

    X = preprocessor.transform(row)
    prob = float(model.predict_proba(X)[0, 1])

    return {
        "predicted_probability": round(prob, 4),
        "predicted_label": int(prob >= THRESHOLD),
    }


def fetch_product_from_keepa(asin: str) -> dict:
    """
    Fetch product data for a single ASIN from Keepa.

    In: asin (str)
    Out: dict with title, price, cat — or raises ValueError if ASIN not found
    """
    url = f"https://api.keepa.com/product?key={KEEPA_API_KEY}&domain=1&asin={asin}&history=1&buybox=1"
    resp = requests.get(url)
    resp.raise_for_status()
    products = resp.json().get("products")

    if not products:
        raise ValueError(f"ASIN {asin} not found in Keepa")

    product = products[0]

    title = product.get("title")
    cat_tree = product.get("categoryTree")
    cat = cat_tree[0]["name"] if cat_tree else None

    # csv[4] is the buy box price list — alternating [keepa_time, price_cents, ...]
    # take the first non-null price value
    price_list = (product.get("csv") or [None] * 5)[4]
    if price_list and len(price_list) > 1 and price_list[1] is not None:
        price = float(price_list[1]) / 100
    else:
        price = None

    # buyBoxSellerIdHistory[-1] is the most recent buy box seller
    seller = (product.get("buyBoxSellerIdHistory") or [None])[-1]

    return {"asin": asin, "title": title, "price": price, "cat": cat, "seller": seller}


def predict_from_asin(asin: str, models_dir: Path | str = MODELS_DIR) -> dict:
    """
    Predict launch success for a product looked up by ASIN.

    In: asin (str)
    Out: dict with title, price, cat, predicted_probability, predicted_label
    """
    product = fetch_product_from_keepa(asin)
    result = predict(
        title=product["title"],
        price=product["price"],
        cat=product["cat"],
        seller=product["seller"],
        models_dir=models_dir,
    )
    return {**product, **result}


if __name__ == "__main__":
    # manual input example
    result = predict(
        title="premium silicone cooking spatula set",
        price=19.99,
        cat="Kitchen",
    )
    print(result)
