# src/preprocessing.py

import tensorflow as tf

def preprocess_image(image_bytes):
    """Preprocesses the input image bytes for prediction."""
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.expand_dims(img, axis=0) 
    return img