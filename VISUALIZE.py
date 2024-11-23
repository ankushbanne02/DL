import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    MaxPooling2D((2, 2), name='pool1'),
    Conv2D(16, (3, 3), activation='relu', name='conv2'),
    MaxPooling2D((2, 2), name='pool2'),
    Flatten(),
    Dense(10, activation='softmax', name='dense')
])

model.summary()

filters, biases = model.get_layer('conv1').get_weights()
n_filters = filters.shape[-1]
fig = plt.figure(figsize=(10, 5))
for i in range(n_filters):
    f = filters[:, :, :, i]
    plt.subplot(1, n_filters, i + 1)
    plt.imshow(f[:, :, 0], cmap='gray')
    plt.axis('off')
plt.suptitle('Filters of Conv1 Layer')
plt.show()

image_path = "th.jpeg"
image = load_img(image_path, target_size=(28, 28), color_mode="grayscale")
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

layer_name = 'conv2'
feature_model = Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
feature_maps = feature_model.predict(image)

num_features = feature_maps.shape[-1]
grid_size = int(np.ceil(np.sqrt(num_features)))

fig = plt.figure(figsize=(15, 15))
for i in range(num_features):
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    plt.axis('off')
plt.suptitle(f'Feature Maps of {layer_name}')
plt.show()
