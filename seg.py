import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image():
    image_path = input("Enter the image path: ")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def thresholding(gray):
    threshold_value = int(input("Enter threshold value (0-255): "))
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh, cmap="gray")
    plt.title("Thresholded Image")
    plt.axis("off")
    plt.show()

def watershed_segmentation(image, gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    markers = cv2.connectedComponents(np.uint8(sure_fg))[1] + 1
    markers[binary == 0] = 0
    cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Watershed Segmentation")
    plt.axis("off")
    plt.show()

def region_growing(gray):
    seed_x = int(input("Enter seed point X coordinate: "))
    seed_y = int(input("Enter seed point Y coordinate: "))
    def region_growing_algorithm(img, seed, thresh=10):
        height, width = img.shape
        visited = np.zeros_like(img, dtype=bool)
        region = np.zeros_like(img, dtype=np.uint8)
        stack = [seed]
        seed_value = img[seed]
        while stack:
            x, y = stack.pop()
            if not visited[y, x]:
                visited[y, x] = True
                region[y, x] = 255
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if not visited[ny, nx] and abs(int(img[ny, nx]) - int(seed_value)) < thresh:
                            stack.append((nx, ny))
        return region
    seed = (seed_x, seed_y)
    region_grown = region_growing_algorithm(gray, seed)
    plt.imshow(region_grown, cmap="gray")
    plt.title("Region Growing")
    plt.axis("off")
    plt.show()

image, gray = load_image()

while True:
    print("\nChoose a segmentation method:")
    print("1. Thresholding")
    print("2. Watershed Segmentation")
    print("3. Region Growing")
    print("4. Exit")
    choice = int(input("Enter your choice (1/2/3/4): "))
    if choice == 1:
        thresholding(gray)
    elif choice == 2:
        watershed_segmentation(image.copy(), gray)
    elif choice == 3:
        region_growing(gray)
    elif choice == 4:
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please try again.")

def automatic_thresholding(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Automatic threshold value (Otsu's method): {thresh}")
    plt.imshow(thresh, cmap="gray")
    plt.title("Thresholded Image (Automatic)")
    plt.axis("off")
    plt.show()
