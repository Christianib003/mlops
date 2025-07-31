import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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