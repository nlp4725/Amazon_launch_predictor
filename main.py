"""
FastAPI app — exposes inference as HTTP endpoints.

Run with:
    uvicorn main:app --reload

Endpoints:
    GET  /                 — health check, lists both models
    GET  /categories       — categories the detailed model recognises
    POST /predict/title    — title only (Home & Kitchen, XGBoost)
    POST /predict/detailed — title + price + category (20 categories, LightGBM)
    POST /predict/asin     — ASIN, fetched from Keepa, routed by category
    POST /predict/url      — Amazon URL, parsed to an ASIN, then as above

Two models with different targets, so every response says which one ran and
what "success" meant. Price is read only by the detailed model.
"""

import os
import re

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference_pipeline.inference import (
    DETAILED_SUCCESS,
    TITLE_CATEGORY,
    TITLE_SUCCESS,
    known_categories,
    predict_detailed,
    predict_from_asin,
    predict_from_title,
    title_vocab_hits,
)

load_dotenv()
KEEPA_API_KEY = os.getenv("KEEPA_API_KEY")

app = FastAPI()


def check_tokens() -> int:
    response = requests.get(f"https://api.keepa.com/token?key={KEEPA_API_KEY}")
    tokens=response.json()["tokensLeft"]
    if tokens < 10:
        raise HTTPException(
            status_code=503,
            detail=f"Not enough Keepa tokens ({tokens} left — try again later)"
        )
    return tokens


def with_title_terms(result: dict, title: str) -> dict:
    """Attach the title model's vocabulary hits, and warn when there are none."""
    matched = title_vocab_hits(title)
    result["title_terms_matched"] = matched
    if not matched:
        result["warning"] = (
            "No word in this title is in the model's 500-term vocabulary, so the "
            f"model read nothing from it. The vocabulary is learned from {TITLE_CATEGORY} "
            "titles — a product outside that category will often match none of it."
        )
    return result


@app.get("/")
def root():
    return {
        "message": "Amazon Launch Predictor API is running",
        "models": {
            "title": {
                "inputs": ["title"],
                "scope": TITLE_CATEGORY,
                "success_definition": TITLE_SUCCESS,
            },
            "detailed": {
                "inputs": ["title", "price", "cat"],
                "scope": f"{len(known_categories())} categories",
                "success_definition": DETAILED_SUCCESS,
            },
        },
    }


@app.get("/categories")
def categories():
    """Categories the detailed model recognises. Anything else is ignored by the model."""
    return {"categories": known_categories()}


class TitleOnly(BaseModel):
    """A product described by its title alone."""
    title: str


class DetailedProduct(BaseModel):
    """A product described by title, category and (optionally) price."""
    title: str
    cat: str
    price: float | None = None


@app.post("/predict/title")
def predict_title(product: TitleOnly):
    """Predict from a title alone, using the Home & Kitchen XGBoost model."""
    if not product.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    return with_title_terms(predict_from_title(product.title), product.title)


@app.post("/predict/detailed")
def predict_detailed_endpoint(product: DetailedProduct):
    """
    Predict from title, category and optional price, using the LightGBM model.

    Category is required and validated: an unrecognised one one-hots to zeros,
    which the model would silently ignore rather than reject.
    """
    if not product.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    valid = known_categories()
    if product.cat not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category {product.cat!r}. Must be one of: {', '.join(valid)}",
        )

    return predict_detailed(title=product.title, price=product.price, cat=product.cat)


@app.post("/predict/manual")
def predict_manual(product: TitleOnly):
    """Deprecated alias for /predict/title, kept so existing callers keep working."""
    return predict_title(product)


@app.post("/predict/asin")
def predict_asin(asin: str):
    """Predict for a product looked up by ASIN from Keepa."""
    check_tokens()
    result = predict_from_asin(asin)
    return with_title_terms(result, result["title"])


@app.post("/predict/url")
def predict_url_str(url_str:str):
    match = re.search(r'/dp/([A-Z0-9]{10})', url_str)
    if not match:
        raise HTTPException(status_code=400, detail="Could not parse ASIN from URL")
    asin = match.group(1)
    check_tokens()
    result = predict_from_asin(asin)
    return with_title_terms(result, result["title"])
