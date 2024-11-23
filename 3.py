import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
image = cv2.imread("", cv2.IMREAD_GRAYSCALE)

# Scaling function
def scaling(image, cx, cy):
    (h, w) = image.shape
    scaling_matrix = np.array([[cx, 0, 0],
                               [0, cy, 0]], dtype=np.float32)
    scaled_image = cv2.warpAffine(image, scaling_matrix, (int(w * cx), int(h * cy)))
    return scaled_image

# Reflection function
def reflection(image):
    (h, w) = image.shape
    matrix = np.array([[-1, 0, w],
                       [0, -1, h]], dtype=np.float32)
    reflected_image = cv2.warpAffine(image, matrix, (w, h))
    return reflected_image

# Translation function
def translation(image, tx, ty):
    (h, w) = image.shape
    matrix = np.float32([[1, 0, tx],
                         [0, 1, ty]])
    translated_image = cv2.warpAffine(image, matrix, (w, h))
    return translated_image

# Rotation function
def rotation(image, angle):
    (h, w) = image.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

# Shearing function
def shearing(image, shx, shy):
    (h, w) = image.shape
    shear_matrix = np.float32([[1, shx, 0],
                               [shy, 1, 0]])
    sheared_image = cv2.warpAffine(image, shear_matrix, (w, h))
    return sheared_image

# Apply transformations
scaled_image = scaling(image, 1.5, 1.5)
reflected_image = reflection(image)
translated_image = translation(image, 50, 50)
rotated_image = rotation(image, 45)  # Rotate 45 degrees
sheared_image = shearing(image, 0.5, 0)  # Shear along x-axis

# Display the original and transformed images
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Scaled Image
plt.subplot(2, 3, 2)
plt.title("Scaled Image")
plt.imshow(scaled_image, cmap='gray')
plt.axis('off')

# Reflected Image
plt.subplot(2, 3, 3)
plt.title("Reflected Image")
plt.imshow(reflected_image, cmap='gray')
plt.axis('off')

# Translated Image
plt.subplot(2, 3, 4)
plt.title("Translated Image")
plt.imshow(translated_image, cmap='gray')
plt.axis('off')

# Rotated Image
plt.subplot(2, 3, 5)
plt.title("Rotated Image")
plt.imshow(rotated_image, cmap='gray')
plt.axis('off')

# Sheared Image
plt.subplot(2, 3, 6)
plt.title("Sheared Image")
plt.imshow(sheared_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
