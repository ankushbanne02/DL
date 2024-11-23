import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from PIL import Image

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((128, 128))  # Resize to 128x128
        img = np.array(img) / 255.0  # Normalize pixel values
        images.append(img)
    return np.array(images)

# Load the dataset
folder_path = '/content/CARS'
x_train = load_images_from_folder(folder_path)

# Split into training and testing sets
x_train, x_test = train_test_split(x_train, test_size=0.2, random_state=42)

# Add noise to the images
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Reshape data to include the channel dimension
x_train_noisy = np.reshape(x_train_noisy, (len(x_train_noisy), 128, 128, 1))
x_test_noisy = np.reshape(x_test_noisy, (len(x_test_noisy), 128, 128, 1))
x_train = np.reshape(x_train, (len(x_train), 128, 128, 1))
x_test = np.reshape(x_test, (len(x_test), 128, 128, 1))

# Build the autoencoder model
def build_autoencoder():
    model = keras.Sequential()
    model.add(layers.Input(shape=(128, 128, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    return model

# Compile and train the autoencoder
autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=4, validation_data=(x_test_noisy, x_test))

# Predict denoised images
denoised_images = autoencoder.predict(x_test_noisy)

# Visualize the original, noisy, and denoised images
n = min(4, len(x_test))
plt.figure(figsize=(15, 5))
for i in range(n):
    # Original image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(128, 128), cmap='gray')
    plt.axis("off")

    # Noisy image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(128, 128), cmap='gray')
    plt.axis("off")

    # Denoised image
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_images[i].reshape(128, 128), cmap='gray')
    plt.axis("off")

plt.show()
