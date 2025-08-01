import streamlit as st
import requests
from PIL import Image
import pandas as pd
import os
import pathlib 


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

PREDICT_API_URL = "http://127.0.0.1:8000/predict/"
RETRAIN_API_URL = "http://127.0.0.1:8000/retrain/"

st.title("🌿 Plant Disease Classifier")
st.write("An end-to-end MLOps project to classify plant diseases from leaf images.")

# Create two columns
col1, col2 = st.columns(2)

# --- Column 1: Prediction ---
with col1:
    st.header("🔍 Predict Disease")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        
        if st.button('Predict'):
            with st.spinner('Analyzing the image...'):
                image_bytes = uploaded_file.getvalue()
                files = {'file': (uploaded_file.name, image_bytes, uploaded_file.type)}
                
                try:
                    response = requests.post(PREDICT_API_URL, files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"**Predicted Class:** {result['predicted_class']}")
                        st.info(f"**Confidence:** {result['confidence_score']:.2%}")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the API: {e}")

with col2:
    st.header("🔄 Model Retraining")
    st.write("Upload new images to enable model retraining.")

    # The filename must start with the class name (e.g., "Tomato___healthy_new.jpg")
    retrain_files = st.file_uploader(
        "Upload images with class names in the filename...", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if retrain_files:
        st.write(f"{len(retrain_files)} images selected for retraining.")
        if st.button("Start Retraining"):
            with st.spinner("Retraining process initiated..."):
                # Prepare files for the request
                files_to_upload = [("files", (file.name, file.getvalue(), file.type)) for file in retrain_files]
                
                try:
                    # Send the files to the /retrain endpoint
                    response = requests.post(RETRAIN_API_URL, files=files_to_upload)
                    if response.status_code == 200:
                        st.success("Model retraining process completed successfully!")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the API: {e}")