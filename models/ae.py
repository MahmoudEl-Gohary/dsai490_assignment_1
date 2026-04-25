"""Standard Autoencoder architecture."""

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_autoencoder(input_shape=(64, 64, 1), latent_dim=64):
    """Builds the encoder and decoder models."""
    
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    latent_outputs = layers.Dense(latent_dim, activation="relu")(x)
    
    encoder = Model(encoder_inputs, latent_outputs, name="encoder")
    
    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 64, activation="relu")(decoder_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
    
    decoder = Model(decoder_inputs, decoder_outputs, name="decoder")
    
    # Autoencoder
    ae_inputs = layers.Input(shape=input_shape)
    encoded = encoder(ae_inputs)
    decoded = decoder(encoded)
    
    autoencoder = Model(ae_inputs, decoded, name="autoencoder")
    
    return autoencoder, encoder, decoder