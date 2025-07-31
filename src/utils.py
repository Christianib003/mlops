import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_and_split_data(train_dir, val_dir, image_size, batch_size):
    """
    Loads and splits data into training, validation, and test sets.

    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        image_size (tuple): The target size for images (height, width).
        batch_size (int): The number of images per batch.

    Returns:
        tuple: A tuple containing training_set, 
        validation_set, test_set, and class_names.
    """
    # Load the training data
    training_set = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    # Load the validation data
    val_temp_set = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = training_set.class_names

    # Split the loaded validation data into final validation and test sets
    val_size = tf.data.experimental.cardinality(val_temp_set).numpy()
    split_point = val_size // 2
    
    validation_set = val_temp_set.take(split_point)
    test_set = val_temp_set.skip(split_point)
    
    return training_set, validation_set, test_set, class_names


def plot_batch(dataset, class_names):
    """Visualizes a batch of images from a dataset."""
    plt.figure(figsize=(12, 12))
    for images, labels in dataset.take(1):
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
    plt.show()


def count_samples(dataset, class_names):
    """Counts the number of samples for each class in a dataset."""
    class_counts = {name: 0 for name in class_names}
    # unbatch the dataset to count individual images
    for _, labels in dataset.unbatch(): 
        class_index = np.argmax(labels.numpy())
        class_name = class_names[class_index]
        class_counts[class_name] += 1
    return class_counts


def build_model(num_classes):
    """
    Builds and compiles a transfer learning model using MobileNetV2.

    Args:
        num_classes (int): The number of output classes for the final layer.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    input_shape = (224, 224, 3)
    
    # Load the MobileNetV2 base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base model
    base_model.trainable = False

    # Create the input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Apply the MobileNetV2 specific preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Pass the preprocessed input to the base model
    x = base_model(x, training=False)
    
    # Add our custom layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = tf.keras.Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def plot_history(history):
    """
    Plots the training and validation accuracy and loss.

    Args:
        history: A Keras History object.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix using seaborn.

    Args:
        y_true (array): Array of true labels.
        y_pred (array): Array of predicted labels.
        class_names (list): List of class names.
    """
    cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()