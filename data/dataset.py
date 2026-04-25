"""Data pipeline for loading and preprocessing Medical MNIST."""

import tensorflow as tf
from configs import config

def load_dataset(data_dir: str) -> tf.data.Dataset:
    """Loads images from directory and prepares a tf.data pipeline."""
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        color_mode="grayscale",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        shuffle=True
    )

    def preprocess(image):
        """Normalizes the image and maps it to an (input, target) tuple."""
        image = tf.cast(image, tf.float32) / 255.0
        return image, image

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset