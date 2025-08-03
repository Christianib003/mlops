import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import requests
from PIL import Image
import pandas as pd
import os
import pathlib
from datetime import timedelta
from streamlit_autorefresh import st_autorefresh


from src.utils import get_dataset_insights

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

PREDICT_API_URL = "http://127.0.0.1:8000/predict/"
RETRAIN_API_URL = "http://127.0.0.1:8000/retrain/"
STATUS_API_URL = "http://127.0.0.1:8000/status/"

def format_uptime(seconds):
    """Formats uptime in seconds to a human-readable string."""
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"

@st.cache_data
def load_insights():
    """Loads and caches the dataset insights."""
    train_dir = pathlib.Path(__file__).parent.parent / "data" / "train"
    if train_dir.exists():
        return get_dataset_insights(str(train_dir))
    return None

with st.sidebar:
    st_autorefresh(interval=1000, key="status_refresher")

    st.header("API Status")
    try:
        response = requests.get(STATUS_API_URL)
        if response.status_code == 200:
            status_data = response.json()
            st.success("● Online")
            uptime_str = format_uptime(status_data['uptime_seconds'])
            st.metric(label="Model Uptime", value=uptime_str)
        else:
            st.error("● Offline")
            st.write(f"Status Code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("● Offline")
        st.write("Connection to API failed.")

st.title("🌿 Plant Disease Classifier")
st.write("An end-to-end MLOps project to classify plant diseases from leaf images.")

tab1, tab2, tab3 = st.tabs(["🔍 Predictor", "📊 Data Insights", "🔄 Retraining"])

with tab1:
    st.header("Predict Plant Disease")
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

with tab2:
    st.header("Exploratory Data Analysis")
    insights_data = load_insights()

    if insights_data:
        st.subheader("1. Class Distribution")
        df_dist = pd.DataFrame(list(insights_data["class_distribution"].items()), columns=['Class', 'Image Count']).set_index('Class')
        st.bar_chart(df_dist)
        st.markdown("""
        **Interpretation:** This chart shows the number of images available for each class in the training set. 
        - **Story:** The dataset is **imbalanced**. The `Tomato___Bacterial_spot` class is dominant, while the `Potato___healthy` class is severely underrepresented. This imbalance is a critical finding, as the model may become biased towards the majority classes. Our evaluation must therefore focus on per-class metrics (like the confusion matrix) rather than just overall accuracy.
        """)
        st.divider()

        st.subheader("2. Image Dimensions (Width vs. Height)")
        df_dims = pd.DataFrame(insights_data["image_dimensions"])
        st.scatter_chart(df_dims, x='width', y='height', color='class')
        st.markdown("""
        **Interpretation:** This scatter plot visualizes the dimensions of a sample of images from the dataset.
        - **Story:** The images come in a variety of sizes, with a heavy concentration around 256x256 pixels. This tells us that a **resizing step is essential** in our preprocessing pipeline to ensure all images have a uniform size before being fed into the model.
        """)
        st.divider()

        st.subheader("3. Average Color (RGB)")
        df_colors = pd.DataFrame(insights_data["average_colors"]).T.rename(columns={0: 'R', 1: 'G', 2: 'B'})
        st.bar_chart(df_colors)
        st.markdown("""
        **Interpretation:** This chart displays the average Red, Green, and Blue color values for each class.
        - **Story:** We can observe that healthy classes, like `Corn___healthy`, tend to have a higher average green value. In contrast, diseased classes like `Corn___Common_rust_` and `Potato___Early_blight` show a relative increase in red/brown tones and a decrease in green. This suggests that **color is a strong predictive feature** that the model can learn to distinguish between healthy and diseased leaves.
        """)
    else:
        st.warning("Could not load data for insights. Please ensure the `data/train` directory is present.")

with tab3:
    st.header("Model Retraining")
    st.write("Upload new images to enable model retraining.")
    
    retrain_files = st.file_uploader("Upload images with class names in the filename...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if retrain_files:
        st.write(f"{len(retrain_files)} images selected for retraining.")
        if st.button("Start Retraining"):
            with st.spinner("Retraining process initiated..."):
                files_to_upload = [("files", (file.name, file.getvalue(), file.type)) for file in retrain_files]
                try:
                    response = requests.post(RETRAIN_API_URL, files=files_to_upload)
                    if response.status_code == 200:
                        st.success("Model retraining process completed successfully!")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the API: {e}")