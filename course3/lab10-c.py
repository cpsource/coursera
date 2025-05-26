import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Load and preprocess MNIST data
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype('float32') / 127.5) - 1.0
x_train = np.expand_dims(x_train, axis=-1)
print("Training data shape:", x_train.shape)

# Generator
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model

generator = build_generator()

# Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])  # Lower LR

# GAN setup
discriminator.trainable = False
z = Input(shape=(100,))
img = generator(z)
valid = discriminator(img)
gan = Model(z, valid)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002))

# Training setup
batch_size = 64
epochs = 50
sample_interval = 10
real_labels = np.ones((batch_size, 1)) * 0.9  # Label smoothing
fake_labels = np.zeros((batch_size, 1))

d_losses, g_losses = [], []

# Training loop
for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]

    # Add Gaussian noise to real images
    real_images += 0.05 * np.random.normal(0, 1, real_images.shape)

    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_images = generator.predict(noise, verbose=0)

    # Train discriminator
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    d_losses.append(d_loss[0])

    # Train generator (twice to let it catch up)
    for _ in range(2):
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real_labels)
    g_losses.append(g_loss)

    # Print progress
    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]:.4f}] [D acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.show()

# Sample and display generated images
def sample_images(generator, epoch, n=25):
    noise = np.random.normal(0, 1, (n, 100))
    gen_imgs = generator.predict(noise, verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    count = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    plt.suptitle(f"Generated Digits at Epoch {epoch}")
    plt.show()

sample_images(generator, epochs)

# Evaluate discriminator accuracy
noise = np.random.normal(0, 1, (batch_size, 100))
generated_imgs = generator.predict(noise)
real_imgs = x_train[np.random.randint(0, x_train.shape[0], batch_size)]

acc_real = discriminator.evaluate(real_imgs, np.ones((batch_size, 1)), verbose=0)
acc_fake = discriminator.evaluate(generated_imgs, np.zeros((batch_size, 1)), verbose=0)

print(f"Discriminator Accuracy on Real: {acc_real[1] * 100:.2f}%")
print(f"Discriminator Accuracy on Fake: {acc_fake[1] * 100:.2f}%")

