"""Visualization utilities for AE and VAE models."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

def plot_reconstructions(model, dataset, n=10, is_vae=False, noise_factor=0.0):
    """Plots original, noisy (optional), and reconstructed images."""
    for batch in dataset.take(1):
        original_images = batch[0]
        break

    input_images = original_images
    if noise_factor > 0.0:
        noise = tf.random.normal(shape=tf.shape(original_images), mean=0.0, stddev=noise_factor)
        input_images = tf.clip_by_value(original_images + noise, 0.0, 1.0)

    if is_vae:
        _, _, z = model.encoder(input_images)
        reconstructed_images = model.decoder(z)
    else:
        reconstructed_images = model(input_images)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(3 if noise_factor > 0 else 2, n, i + 1)
        plt.imshow(tf.squeeze(original_images[i]), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Noisy Input (if applicable)
        if noise_factor > 0.0:
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(tf.squeeze(input_images[i]), cmap="gray")
            plt.title("Noisy")
            plt.axis("off")

        # Reconstructed
        offset = 2 * n if noise_factor > 0 else n
        ax = plt.subplot(3 if noise_factor > 0 else 2, n, i + 1 + offset)
        plt.imshow(tf.squeeze(reconstructed_images[i]), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("reconstructions.png")
    plt.close()

def plot_generated_images(decoder, latent_dim, n=10):
    """Generates and plots new images from random latent vectors."""
    random_latent_vectors = tf.random.normal(shape=(n, latent_dim))
    generated_images = decoder(random_latent_vectors)

    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(tf.squeeze(generated_images[i]), cmap="gray")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("generated_samples.png")
    plt.close()

def plot_latent_space_2d(encoder, dataset):
    """Plots a 2D projection of the latent space using PCA."""
    z_means = []
    for batch in dataset.take(50):
        images = batch[0]
        z_mean, _, _ = encoder(images)
        z_means.append(z_mean)
    
    z_means = np.concatenate(z_means, axis=0)
    
    pca = PCA(n_components=2)
    z_means_2d = pca.fit_transform(z_means)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(z_means_2d[:, 0], z_means_2d[:, 1], alpha=0.5, s=2)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Latent Space 2D Visualization (PCA)")
    plt.savefig("latent_space.png")
    plt.close()

def plot_loss(history, title, filename):
    """Plots training loss."""
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename)
    plt.close()