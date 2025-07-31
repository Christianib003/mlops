import tensorflow as tf
import numpy as np
from src.preprocessing import preprocess_image

CLASS_NAMES = [
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 
    'Potato___Early_blight', 'Potato___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___healthy'
]

model = tf.keras.models.load_model('models/plant_classifier_v1.keras')

def predict_single(image_bytes):
    """Makes a prediction on a single image."""
    processed_image = preprocess_image(image_bytes)
    prediction = model.predict(processed_image)
    
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence_score = float(np.max(prediction))
    
    return predicted_class_name, confidence_score