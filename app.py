"""
Streamlit UI — Amazon Launch Predictor.

Run with:
    streamlit run app.py

Requires FastAPI server running at localhost:8000:
    uvicorn main:app --reload
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.title("Amazon Launch Predictor")
st.write("Enter an ASIN or Amazon product URL to predict launch success.")

if "clear_count" not in st.session_state:
    st.session_state.clear_count = 0

asin_input = st.text_input("ASIN", key=f"asin_{st.session_state.clear_count}")
url_input = st.text_input("Amazon Product URL", key=f"url_{st.session_state.clear_count}")

col_predict, col_clear = st.columns([1, 1])
predict_clicked = col_predict.button("Predict")
clear_clicked = col_clear.button("Clear")

if clear_clicked:
    st.session_state.clear_count += 1
    st.rerun()

if predict_clicked:
    if not asin_input and not url_input:
        st.error("Please enter an ASIN or a URL.")
    else:
        with st.spinner("Fetching product and predicting..."):
            try:
                if asin_input:
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

                    # product info below
                    st.markdown("---")
                    st.subheader("Product Info")
                    st.write(f"**Title:** {data.get('title', 'N/A')}")
                    st.write(f"**ASIN:** {data.get('asin', asin_input or 'N/A')}")

                else:
                    try:
                        detail = resp.json().get('detail', 'Unknown error')
                    except Exception:
                        detail = resp.text or 'Unknown error'
                    st.error(f"Error {resp.status_code}: {detail}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API — make sure `uvicorn main:app --reload` is running.")
