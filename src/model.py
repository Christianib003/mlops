import tensorflow as tf
import os
from src.utils import load_and_split_data

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

def retrain_model():
    """
    Loads the existing model and continues training it on the dataset.
    This simulates a retraining pipeline.
    """
    print("--- Starting Model Retraining ---")
    
    # 1. Define Constants
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    MODEL_PATH = 'models/plant_classifier_v1.keras'
    DATA_DIR = 'data'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')

    # 2. Load the Data
    print("Loading and preparing data...")
    training_set, validation_set, _, _ = load_and_split_data(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # 3. Load the Existing Model
    print(f"Loading existing model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 4. Continue Training the Model
    print("Continuing training...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    
    history = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=10, 
        callbacks=[early_stopping]
    )

    # 5. Save the Newly Retrained Model
    print(f"Saving updated model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("--- Model retraining complete. ---")
    
    return history

if __name__ == '__main__':
    # This allows you to test the retraining process directly
    # from the terminal: `python src/model.py`
    retrain_model()