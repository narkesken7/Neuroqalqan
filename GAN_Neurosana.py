import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy Adam for M1/M2 compatibility
import matplotlib.pyplot as plt

# Remove mixed precision policy if running on CPU
if tf.config.list_physical_devices('GPU'):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

# Option to choose between grayscale or RGB images
USE_GRAYSCALE = True  # Set to False if you want to use RGB images


# GAN model definition
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_dim=100))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))

    # Adjust output layer based on image type (grayscale or RGB)
    if USE_GRAYSCALE:
        model.add(layers.Dense(28 * 28 * 1, activation='tanh'))
        model.add(layers.Reshape((28, 28, 1)))  # Grayscale
    else:
        model.add(layers.Dense(28 * 28 * 3, activation='tanh'))
        model.add(layers.Reshape((28, 28, 3)))  # RGB

    return model


def build_discriminator():
    model = tf.keras.Sequential()

    if USE_GRAYSCALE:
        model.add(layers.Flatten(input_shape=(28, 28, 1)))  # Grayscale input
    else:
        model.add(layers.Flatten(input_shape=(28, 28, 3)))  # RGB input

    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    discriminator.trainable = False
    z = layers.Input(shape=(100,))
    img = generator(z)
    valid = discriminator(img)
    gan = tf.keras.Model(z, valid)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan


# Hyperparameters
epochs = 200
batch_size = 64
latent_dim = 100
image_size = (28, 28)  # Fixed image size for generator and discriminator


# Load and preprocess the images from the directory
def load_local_images(image_dir, image_size, batch_size):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        label_mode=None,  # No labels, since this is unsupervised
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    # Optionally convert to grayscale
    if USE_GRAYSCALE:
        dataset = dataset.map(lambda x: tf.image.rgb_to_grayscale(x))

    # Normalize the images to [-1, 1]
    normalization_layer = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)
    dataset = dataset.map(lambda x: normalization_layer(x))

    return dataset


# Directory where your local images are stored
image_dir = '/Users/keshubai/Desktop/CS/Python/Neuroqalqan/Brain_Data_1/Training/meningioma'

# Load the dataset
dataset = load_local_images(image_dir, image_size, batch_size)

# Labels for the discriminator
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

# Build the GAN models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Training loop
for epoch in range(epochs):
    print(f"Starting epoch {epoch}")
    for real_imgs in dataset:
        real_imgs = real_imgs[:batch_size]
        current_batch_size = real_imgs.shape[0]

        real = np.ones((current_batch_size, 1))
        fake = np.zeros((current_batch_size, 1))

        # Generate fake images
        noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Print losses for each batch
        print(f"Discriminator loss real: {d_loss_real}, fake: {d_loss_fake}")

        # Train Generator
        noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)

    # Print epoch summary
    print(f"Epoch {epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

    # Save generated image samples every 10 epochs
    if epoch % 10 == 0:
        noise = np.random.normal(0, 1, (1, latent_dim))
        gen_img = generator.predict(noise)
        gen_img = 0.5 * gen_img + 0.5

        plt.imshow(gen_img[0, :, :, 0] if USE_GRAYSCALE else gen_img[0])
        plt.title(f"Epoch {epoch}")
        plt.show()
