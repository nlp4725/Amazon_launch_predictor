"""
Streamlit UI — Amazon Launch Predictor.

Run with:
    streamlit run app.py

Requires FastAPI server running at localhost:8000:
    uvicorn main:app --reload
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

TITLE_MODE = "Title only — Home & Kitchen"
DETAILED_MODE = "Title + price + category — 20 categories"


@st.cache_data(ttl=3600)
def fetch_categories():
    """Category list comes from the fitted model, so the dropdown can't drift out of sync."""
    resp = requests.get(f"{API_URL}/categories", timeout=30)
    resp.raise_for_status()
    return resp.json()["categories"]


st.title("Amazon Launch Predictor")

mode = st.radio(
    "Model",
    [TITLE_MODE, DETAILED_MODE],
    captions=[
        "Predicts ~5 reviews within 90 days. Reads the title and nothing else.",
        "Predicts more than 10 reviews at 180 days. Price is only used here.",
    ],
)
st.caption(
    "The two models are trained on different targets — their probabilities are "
    "not comparable to each other."
)

if "clear_count" not in st.session_state:
    st.session_state.clear_count = 0

title_input = st.text_input("Product Title", key=f"title_{st.session_state.clear_count}")

cat_input, price_input = None, None
if mode == DETAILED_MODE:
    try:
        categories = fetch_categories()
    except Exception:
        categories = []
    cat_input = st.selectbox(
        "Category",
        categories,
        index=None,
        placeholder="Select a category…",
        key=f"cat_{st.session_state.clear_count}",
    )
    price_input = st.number_input(
        "Price ($) — optional", min_value=0.0, value=None, step=1.0,
        key=f"price_{st.session_state.clear_count}",
    )

st.markdown("**or look the product up on Amazon** (requires an active Keepa plan)")
asin_input = st.text_input("ASIN", key=f"asin_{st.session_state.clear_count}")
url_input = st.text_input("Amazon Product URL", key=f"url_{st.session_state.clear_count}")

col_predict, col_clear = st.columns([1, 1])
predict_clicked = col_predict.button("Predict")
clear_clicked = col_clear.button("Clear")

if clear_clicked:
    st.session_state.clear_count += 1
    st.rerun()

if predict_clicked:
    if not title_input and not asin_input and not url_input:
        st.error("Please enter a product title, an ASIN, or a URL.")
    elif title_input and mode == DETAILED_MODE and not cat_input:
        st.error("Please pick a category — this model needs it to score a title.")
    else:
        with st.spinner("Predicting..."):
            try:
                if title_input and mode == DETAILED_MODE:
                    resp = requests.post(f"{API_URL}/predict/detailed", json={
                        "title": title_input,
                        "cat": cat_input,
                        "price": price_input,
                    })
                elif title_input:
                    resp = requests.post(f"{API_URL}/predict/title", json={"title": title_input})
                elif asin_input:
                    resp = requests.post(f"{API_URL}/predict/asin", params={"asin": asin_input})
                else:
                    resp = requests.post(f"{API_URL}/predict/url", params={"url_str": url_input})

                if resp.status_code == 200:
                    data = resp.json()

                    # probability and label — prominent center display
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Launch Probability", f"{data['predicted_probability'] * 100:.1f}%")
                    with col2:
                        label = "Success" if data["predicted_label"] == 1 else "Failure"
                        st.metric("Prediction", label)
                    st.caption(
                        f"Model: `{data.get('model', 'n/a')}` — success means "
                        f"{data.get('success_definition', 'n/a')}."
                    )

                    # product info below
                    st.markdown("---")
                    st.subheader("Product Info")
                    st.write(f"**Title:** {data.get('title', title_input or 'N/A')}")
                    st.write(f"**ASIN:** {data.get('asin', asin_input or 'N/A')}")

                    if data.get("warning"):
                        st.warning(data["warning"])
                    elif data.get("title_terms_matched"):
                        terms = ", ".join(data["title_terms_matched"])
                        st.caption(f"Title terms the model recognised: {terms}")

                else:
                    try:
                        detail = resp.json().get('detail', 'Unknown error')
                    except Exception:
                        detail = resp.text or 'Unknown error'
                    st.error(f"Error {resp.status_code}: {detail}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API — make sure `uvicorn main:app --reload` is running.")
