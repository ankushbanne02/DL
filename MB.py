from ultralytics import YOLO

# Path to your dataset's data.yaml file
data_yaml_path = "C:/Users/AKASH/Downloads/object detection/Persian_Car_Plates_YOLOV8/data.yaml" # Update this path

# Load YOLOv8 model (choose a version: 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO("yolov8n.pt") # yolov8n.pt = nano (lightweight), yolov8s.pt = small

# Train the model
model.train(
    data=data_yaml_path, # Path to the data.yaml file
    epochs=1, # Number of training epochs
    batch=16, # Batch size
    imgsz=640, # Image size (resolution)
    project="PersianCarPlates", # Project folder to save results
    name="car_plate_train", # Experiment name
    pretrained=True # Use pretrained weights
)

# Validate the model
metrics = model.val()

# Perform inference on test images
results = model.predict(
    source="/path/to/dataset/test/images", # Path to test images
    save=True # Save predictions
)

# Optional: Export the trained model to ONNX, TorchScript, or other formats
model.export(format="onnx") # Options: onnx, torchscript, coreml, etc.