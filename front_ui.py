import streamlit as st
import requests

st.title(" Card Classification ")

st.write("Upload image to send to FastAPI model.")

uploaded_files = st.file_uploader(
    "Choose images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

API_URL = "http://127.0.0.1:8000/predict"

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption="Uploaded Image", width=200)

        files = {"main_photo": uploaded_file.getvalue()}

        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['predicted_card']}")
            st.info(f"Confidence: {result['confidence']:.4f}")
        else:
            st.error("API error ")