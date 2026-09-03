"""
Inference: predict launch success for a new product. Two models, different inputs.

TITLE MODEL (xgb_title_model.joblib)
    Input: title only. TF-IDF, 500 unigram+bigram features, then XGBClassifier.
    Scope: Home & Kitchen. Success: review velocity >= 0.056 (~5 reviews in 90 days).
    Entry points: predict_from_title(), predict_from_asin()

DETAILED MODEL (lgbm_tfidf_model.pkl + preprocessor.pkl)
    Input: title, price, cat, seller, and the current month/year.
    Scope: 20 categories. Success: more than 10 reviews at 180 days.
    Entry points: predict_detailed(), known_categories()

The two are NOT comparable — different targets, different scopes, different
thresholds — so a caller must say which one it wants and report which one ran.
Price is only ever read by the detailed model.
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

# ---------- title model ----------
TITLE_MODEL_FILE = "xgb_title_model.joblib"
TITLE_THRESHOLD = 0.4
TITLE_CATEGORY = "Home & Kitchen"
TITLE_SUCCESS = "at least 5 reviews within 90 days (review velocity >= 0.056)"

# ---------- detailed model ----------
DETAILED_MODEL_FILE = "lgbm_tfidf_model.pkl"
DETAILED_PREPROCESSOR_FILE = "preprocessor.pkl"
DETAILED_THRESHOLD = 0.6
DETAILED_SUCCESS = "more than 10 reviews at 180 days"


@lru_cache(maxsize=None)
def _load(models_dir: str, filename: str):
    """Cached load — without this every request would unpickle the artifact again."""
    return load(Path(models_dir) / filename)


# ==================== title model ====================

def predict_from_title(title: str, models_dir: Path | str = MODELS_DIR) -> dict:
    """
    Predict launch success from a title alone (Home & Kitchen).

    In: title (str) — the only feature this model takes
    Out: dict with predicted_probability, predicted_label, model, success_definition
    """
    model = _load(str(models_dir), TITLE_MODEL_FILE)
    prob = float(model.predict_proba(pd.DataFrame({"title": [title]}))[0, 1])

    return {
        "predicted_probability": round(prob, 4),
        "predicted_label": int(prob >= TITLE_THRESHOLD),
        "model": "xgboost-title",
        "success_definition": TITLE_SUCCESS,
    }


def title_vocab_hits(title: str, models_dir: Path | str = MODELS_DIR) -> list[str]:
    """
    The title model's vocabulary terms a title matches.

    Out: sorted list of matched terms. An empty list means the title contributes
    nothing — the 500 terms are learned from Home & Kitchen titles, so a product
    outside that category often matches none of them.
    """
    pipeline = _load(str(models_dir), TITLE_MODEL_FILE)
    tfidf = pipeline.named_steps["preprocessing"].named_transformers_["tfidf"]
    inverse = {index: term for term, index in tfidf.vocabulary_.items()}
    return sorted(inverse[i] for i in tfidf.transform([title]).nonzero()[1])


# ==================== detailed model ====================

def known_categories(models_dir: Path | str = MODELS_DIR) -> list[str]:
    """
    The category values the detailed model's encoder recognises.

    Out: sorted list, excluding the "unknown" imputation bucket. Anything not in
    this list one-hots to all zeros (handle_unknown="ignore"), so the model
    silently ignores it rather than raising.
    """
    preprocessor = _load(str(models_dir), DETAILED_PREPROCESSOR_FILE)
    encoder = {n: t for n, t, _ in preprocessor.transformers_}["cat"].named_steps["encoder"]
    return sorted(c for c in encoder.categories_[0] if c != "unknown")


def predict_detailed(
    title: str,
    price: float | None,
    cat: str,
    seller: str = "unknown",
    models_dir: Path | str = MODELS_DIR,
) -> dict:
    """
    Predict launch success from title, price and category across 20 categories.

    In: title (str), price (float or None — imputed to the training median),
        cat (str, must be one of known_categories()), seller (str)
    Out: dict with predicted_probability, predicted_label, model, success_definition
    """
    preprocessor = _load(str(models_dir), DETAILED_PREPROCESSOR_FILE)
    model = _load(str(models_dir), DETAILED_MODEL_FILE)

    now = datetime.today()
    row = pd.DataFrame([{
        "title": title,
        "price": price,
        "cat": cat,
        "seller": seller or "unknown",
        "month": now.month,
        "year": now.year,
    }])

    prob = float(model.predict_proba(preprocessor.transform(row))[0, 1])

    return {
        "predicted_probability": round(prob, 4),
        "predicted_label": int(prob >= DETAILED_THRESHOLD),
        "model": "lightgbm-detailed",
        "success_definition": DETAILED_SUCCESS,
    }


# ==================== Keepa lookup ====================

def fetch_product_from_keepa(asin: str) -> dict:
    """
    Fetch product data for a single ASIN from Keepa.

    In: asin (str)
    Out: dict with asin, title, price, cat, seller — or raises ValueError if not found

    Passing the key via params keeps it out of the exception message that
    raise_for_status() builds, which otherwise lands in the logs verbatim.
    """
    params = {"key": KEEPA_API_KEY, "domain": 1, "asin": asin, "history": 1, "buybox": 1}
    resp = requests.get("https://api.keepa.com/product", params=params)
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

    Keepa supplies price and category, so this routes to the detailed model when
    the category is one the model knows, and falls back to the title model when
    it is not.

    In: asin (str)
    Out: dict with the product fields plus the prediction
    """
    product = fetch_product_from_keepa(asin)
    if not product["title"]:
        raise ValueError(f"ASIN {asin} has no title in Keepa")

    if product["cat"] in known_categories(models_dir):
        result = predict_detailed(
            title=product["title"],
            price=product["price"],
            cat=product["cat"],
            seller=product["seller"],
            models_dir=models_dir,
        )
    else:
        result = predict_from_title(product["title"], models_dir=models_dir)

    return {**product, **result}
