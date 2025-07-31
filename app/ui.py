# app.py

import streamlit as st
import requests
from PIL import Image
import io


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict/"


st.title("🌿 Plant Disease Classifier")
st.write("Upload an image of a plant leaf, and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    # Prediction button
    if st.button('Predict Disease'):
        with st.spinner('Analyzing the image...'):
            # Convert the uploaded file to bytes
            image_bytes = uploaded_file.getvalue()
            
            # Prepare the file for the POST request
            files = {'file': (uploaded_file.name, image_bytes, uploaded_file.type)}
            
            try:
                # Make the request to the FastAPI backend
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Predicted Class:** {result['predicted_class']}")
                    st.info(f"**Confidence:** {result['confidence_score']:.2%}")
                else:
                    st.error(f"Error: Received status code {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the API: {e}")