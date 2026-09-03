"""
FastAPI app — exposes inference as HTTP endpoints.

Run with:
    uvicorn main:app --reload

Endpoints:
    GET  /                — health check
    GET  /categories      — category values the model recognises
    POST /predict/manual  — takes title, price, cat from user (no Keepa lookup)
    POST /predict/asin    — takes ASIN, fetches from Keepa, returns prediction
"""

import os
import re

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference_pipeline.inference import (
    known_categories,
    predict,
    predict_from_asin,
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


@app.get("/")
def root():
    return {"message": "Amazon Launch Predictor API is running"}


@app.get("/categories")
def categories():
    """The category values the fitted encoder recognises. Anything else is ignored by the model."""
    return {"categories": known_categories()}


class ManualProduct(BaseModel):
    """A product described by the user rather than looked up by ASIN."""
    title: str
    cat: str
    price: float | None = None


@app.post("/predict/manual")
def predict_manual(product: ManualProduct):
    """
    Predict launch success from a title and category. Price is optional.

    Category is required: an unrecognised one one-hots to zeros, and a title whose
    words all fall outside the 200-term vocabulary contributes nothing either, which
    together drive the model into a near-constant fallback prediction.
    """
    if not product.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    valid = known_categories()
    if product.cat not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category {product.cat!r}. Must be one of: {', '.join(valid)}",
        )

    result = predict(title=product.title, price=product.price, cat=product.cat)
    matched = title_vocab_hits(product.title)
    result["title_terms_matched"] = matched
    if not matched:
        result["warning"] = (
            "No word in this title is in the model's 200-term vocabulary, so the title "
            "was ignored — this prediction reflects only the category and price."
        )
    return result


@app.post("/predict/asin")
def predict_asin(asin: str):
    """Predict launch success for a product looked up by ASIN from Keepa."""
    check_tokens()
    return predict_from_asin(asin)

@app.post("/predict/url")
def predict_url_str(url_str:str):
    match = re.search(r'/dp/([A-Z0-9]{10})', url_str)
    if not match:
        raise HTTPException(status_code=400, detail="Could not parse ASIN from URL")
    asin = match.group(1)
    return predict_from_asin(asin)
