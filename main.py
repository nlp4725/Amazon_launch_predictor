"""
FastAPI app — exposes inference as HTTP endpoints.

Run with:
    uvicorn main:app --reload

Endpoints:
    GET  /                — health check
    POST /predict/manual  — takes title, price, cat from user
    POST /predict/asin    — takes ASIN, fetches from Keepa, returns prediction
"""

import os

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


@app.post("/predict/asin")
def predict_asin(asin: str):
    """Predict launch success for a product looked up by ASIN from Keepa."""
    check_tokens()
    return predict_from_asin(asin)

@app.post("/predict/url")
def predict_url_str(url_str:str):
    url_str = url_str.removeprefix("https://").removeprefix("http://")
    char_list=url_str.split('/')
    asin=char_list[3]
    check_tokens()
    return predict_from_asin(asin)
