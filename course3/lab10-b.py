# Lab: Develop GANs using Keras - Fixed Version

# Suppress warnings and set environment variables
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Load the MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Normalize the pixel values to the range [-1, 1]
x_train = x_train.astype('float32') / 127.5 - 1.
x_train = np.expand_dims(x_train, axis=-1)

print("Data shape:", x_train.shape)

# Set hyperparameters
latent_dim = 100
batch_size = 64
epochs = 50
sample_interval = 5
learning_rate = 0.0001  # Even lower learning rate for stability
beta_1 = 0.5  # Momentum term in Adam optimizer

# Define the generator model with improved architecture
def build_generator():
    model = Sequential()
    
    # First layer
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.2))  # Reduced dropout
    
    # Second layer
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.2))  # Reduced dropout
    
    # Third layer
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    # Output layer
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    
    return model

# Define the discriminator model with improved architecture
def build_discriminator():
    model = Sequential()
    
    # First layer
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))  # Reduced dropout to prevent underfitting
    
    # Second layer
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))  # Reduced dropout rate
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Build the generator
generator = build_generator()
generator.summary()

# Build the discriminator
discriminator = build_discriminator()
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=learning_rate * 0.5, beta_1=beta_1),  # Slower learning for discriminator
    metrics=['accuracy']
)
discriminator.summary()

# Build the GAN
def build_gan(generator, discriminator):
    # Set the discriminator to not be trainable within the GAN model
    discriminator.trainable = False
    
    # GAN input (noise) and output (discriminator's prediction)
    gan_input = Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    
    # Define and compile the GAN model
    gan = Model(gan_input, gan_output)
    gan.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate, beta_1=beta_1)
    )
    
    return gan

# Build the GAN
gan = build_gan(generator, discriminator)
gan.summary()

# Function to generate and plot sample images
def sample_images(generator, epoch):
    r, c = 5, 5  # row, col
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    generated_images = generator.predict(noise)
    
    # Rescale images from [-1, 1] to [0, 1]
    generated_images = 0.5 * generated_images + 0.5
    
    fig, axs = plt.subplots(r, c, figsize=(10, 10))
    cnt = 0
    
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
            
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f"gan_output_epoch_{epoch}.png")
    plt.close()

# Training loop with improvements
# Labels for real and fake images with milder label smoothing
real = np.ones((batch_size, 1)) * 0.95  # Use 0.95 instead of 0.9
fake = np.zeros((batch_size, 1)) + 0.05  # Use 0.05 instead of 0.1

# Lists to store loss history
d_loss_history = []
g_loss_history = []
d_acc_history = []

# Training loop
for epoch in range(epochs):
    # ---------------------
    #  Train Discriminator
    # ---------------------
    
    # Select a random batch of real images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]
    
    # Generate a batch of fake images
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)
    
    # Train the discriminator (only if not too strong)
    if epoch == 0 or d_loss[1] < 0.8:  # Always train in first epoch or if acc < 80%
        d_loss_real = discriminator.train_on_batch(real_images, real)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Store accuracy history
    d_acc_history.append(d_loss[1])
    
    # ---------------------
    #  Train Generator
    # ---------------------
    
    # Generate new noise for generator training
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    
    # Train the generator (to have the discriminator label generated samples as valid)
    g_loss = gan.train_on_batch(noise, real)
    
    # Store loss history
    d_loss_history.append(d_loss[0])
    g_loss_history.append(g_loss)
    
    # Print the progress and generate sample images
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
        sample_images(generator, epoch)

# Plot accuracy and loss history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(d_loss_history, label='Discriminator Loss')
plt.plot(g_loss_history, label='Generator Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(d_acc_history, label='Discriminator Accuracy')
plt.axhline(y=0.5, color='r', linestyle='-', label='Ideal Balance (50%)')
plt.title('Discriminator Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('gan_training_metrics.png')
plt.show()

# Generate and visualize final images
print("Generating final sample images...")
sample_images(generator, epochs)

# Evaluate the model
print("Evaluating the model...")
noise = np.random.normal(0, 1, (batch_size, latent_dim))
generated_images = generator.predict(noise)

# Evaluate the discriminator on real images
real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
d_loss_real = discriminator.evaluate(real_images, real, verbose=0)

# Evaluate the discriminator on fake images
d_loss_fake = discriminator.evaluate(generated_images, fake, verbose=0)

print(f"Discriminator Accuracy on Real Images: {d_loss_real[1] * 100:.2f}%")
print(f"Discriminator Accuracy on Fake Images: {d_loss_fake[1] * 100:.2f}%")
print(f"Total Discriminator Accuracy: {((d_loss_real[1] + d_loss_fake[1]) / 2) * 100:.2f}%")
