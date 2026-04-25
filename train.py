"""Main script for training AE and VAE models and generating visualizations."""

import tensorflow as tf
from configs import config
from data.dataset import load_dataset
from models.ae import build_autoencoder
from models.vae import build_vae_components, VAE
from utils import visualization

def main():
    print("Loading dataset...")
    dataset = load_dataset(config.DATA_DIR)

    # 1. Train Standard Autoencoder
    print("\n--- Training Standard Autoencoder ---")
    ae, ae_encoder, ae_decoder = build_autoencoder(
        input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 1),
        latent_dim=config.LATENT_DIM
    )
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), loss="mse")
    
    ae_history = ae.fit(dataset, epochs=config.EPOCHS)
    visualization.plot_loss(ae_history, "AE Training Loss", "ae_loss.png")
    
    print("Evaluating AE Reconstructions...")
    visualization.plot_reconstructions(ae, dataset, is_vae=False)
    visualization.plot_reconstructions(ae, dataset, is_vae=False, noise_factor=0.2)

    # 2. Train Variational Autoencoder
    print("\n--- Training Variational Autoencoder ---")
    vae_encoder, vae_decoder = build_vae_components(
        input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 1),
        latent_dim=config.LATENT_DIM
    )
    vae = VAE(vae_encoder, vae_decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE))
    
    vae_history = vae.fit(dataset, epochs=config.EPOCHS)
    visualization.plot_loss(vae_history, "VAE Total Loss", "vae_loss.png")
    
    print("Evaluating VAE Reconstructions & Generating Samples...")
    visualization.plot_reconstructions(vae, dataset, is_vae=True)
    visualization.plot_reconstructions(vae, dataset, is_vae=True, noise_factor=0.2)
    
    visualization.plot_generated_images(vae_decoder, config.LATENT_DIM)
    visualization.plot_latent_space_2d(vae_encoder, dataset)

    print("\nTraining and evaluation complete. Check the generated PNG files.")

if __name__ == "__main__":
    main()