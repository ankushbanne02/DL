from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("C:/Users/AKASH/Downloads/yolov8s.pt")

# Path to the image file
image_path = "C:/Users/AKASH/Downloads/Car.jpeg"  # Replace with your image path
print("hello")
# Read the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not open image.")
    exit()

# Convert the image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform object detection
results = model(image_rgb)

# Get the first result if results is a list
result = results[0] if isinstance(results, list) else results

# Annotate detected objects on the image
annotated_image = result.plot()
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display the annotated image using matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(annotated_image_rgb)
plt.axis('off')  # Hide axes
plt.show()
