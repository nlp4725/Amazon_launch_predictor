"""
FastAPI app — exposes inference as HTTP endpoints.

Run with:
    uvicorn main:app --reload

Endpoints:
    GET  /                — health check
    POST /predict/manual  — takes title, price, cat from user (no Keepa lookup)
    POST /predict/asin    — takes ASIN, fetches from Keepa, returns prediction
"""

import os
import re

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference_pipeline.inference import predict, predict_from_asin

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


class ManualProduct(BaseModel):
    """A product described by the user rather than looked up by ASIN."""
    title: str
    price: float | None = None
    cat: str | None = None


@app.post("/predict/manual")
def predict_manual(product: ManualProduct):
    """Predict launch success from a title alone. Price and category are optional."""
    if not product.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    return predict(title=product.title, price=product.price, cat=product.cat)


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
