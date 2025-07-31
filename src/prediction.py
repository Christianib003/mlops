import tensorflow as tf
import numpy as np

# A list of output class names
CLASS_NAMES = [
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___healthy'
]

model = tf.keras.models.load_model('../models/plant_classifier_v1.keras')

def preprocess_image(image_bytes):
    """Preprocesses the input image bytes for prediction."""
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.expand_dims(img, axis=0) 
    return img

def predict_single(image_bytes):
    """Makes a prediction on a single image and returns the class name and confidence."""
    processed_image = preprocess_image(image_bytes)
    prediction = model.predict(processed_image)
    
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence_score = float(np.max(prediction))
    
    return predicted_class_name, confidence_score