import numpy as np
import tensorflow as tf
import os
import pathlib
from datetime import datetime
from src.utils import load_and_split_data
from src.prediction import CLASS_NAMES 


def build_model(num_classes):
    """
    Builds and compiles a new transfer learning model using MobileNetV2.
    """
    input_shape = (224, 224, 3)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model():
    """
    Builds and trains a new model on the full original dataset.
    This is the main 'Train' functionality.
    """
    print("--- Starting Full Model Training ---")
    
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    MODEL_PATH = 'models/plant_classifier_v1.keras'
    DATA_DIR = 'data'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')

    print("Loading original dataset...")
    training_set, validation_set, _, class_names = load_and_split_data(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    print("Building a new model...")
    model = build_model(num_classes=len(class_names))
    
    print("Training the model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        training_set,
        validation_data=validation_set,
        epochs=50, 
        callbacks=[early_stopping]
    )

    print(f"Saving trained model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("--- Full model training complete. ---")
    return model

def retrain_on_new_images(new_data_dir):
    """
    Loads the existing model and fine-tunes it using a custom data pipeline
    to handle subsets of new data.
    """
    print(f"--- Starting Retraining on New Images from {new_data_dir} ---")
    
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 8
    MODEL_PATH = 'models/plant_classifier_v1.keras'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RETRAINED_MODEL_PATH = f'models/plant_classifier_retrained_{timestamp}.keras'

    print("Building custom data pipeline for new images...")
    
    data_dir = pathlib.Path(new_data_dir)
    
    image_paths = list(data_dir.glob('*/*.*'))
    image_paths = [str(path) for path in image_paths]
    
    if not image_paths:
        raise ValueError("No images found in the new data directory.")

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    
    ALL_CLASS_NAMES = np.array(CLASS_NAMES)

    def process_path(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label_str = parts[-2]
        
        label = tf.argmax(label_str == ALL_CLASS_NAMES)
        label = tf.one_hot(label, len(ALL_CLASS_NAMES))
        
        img = tf.io.read_file(file_path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMAGE_SIZE)
        
        return img, label

    new_data_set = path_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    new_data_set = new_data_set.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"Loading existing model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Fine-tuning the model on new data...")
    model.fit(new_data_set, epochs=10)

    print(f"Saving retrained model to {RETRAINED_MODEL_PATH}...")
    model.save(RETRAINED_MODEL_PATH)
    print("--- Retraining on new images complete. ---")