from PIL import Image
import pandas as pd
import os
import random
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
        shuffle=True,
        seed=123
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


def get_dataset_insights(data_dir, sample_size=100):
    """
    Analyzes the dataset to generate insights for visualization.

    Args:
        data_dir (str): Path to the training data directory.
        sample_size (int): Number of images to sample per class for analysis.

    Returns:
        dict: A dictionary containing data for class distribution, dimensions, and colors.
    """
    insights = {
        "class_distribution": {},
        "image_dimensions": [],
        "average_colors": {}
    }
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        image_files = os.listdir(class_path)
        
        insights["class_distribution"][class_name] = len(image_files)
        
        if len(image_files) > sample_size:
            image_files_sample = random.sample(image_files, sample_size)
        else:
            image_files_sample = image_files
        
        class_colors = []
        for image_file in image_files_sample:
            try:
                with Image.open(os.path.join(class_path, image_file)) as img:
                    insights["image_dimensions"].append({
                        "width": img.width,
                        "height": img.height,
                        "class": class_name
                    })
                    img_thumb = img.resize((50, 50))
                    class_colors.append(np.mean(np.array(img_thumb), axis=(0, 1)))
            except Exception:
                continue
        
        if class_colors:
            insights["average_colors"][class_name] = np.mean(class_colors, axis=0)
            
    return insights