import cv2
import os
import numpy as np

# Load training data
def load_training_data(data_path):
    faces, labels, label_map = [], [], {}
    
    for label, person_name in enumerate(os.listdir(data_path)):
        person_folder = os.path.join(data_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_map[label] = person_name  

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                faces.append(image)
                labels.append(label)
            else:
                print(f"Error loading: {image_path}")

    return faces, labels, label_map

def train_lbph_model(faces, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer

def recognize_face(recognizer, test_image_path, label_map, confidence_threshold=60):
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        raise ValueError(f"Error loading test image: {test_image_path}")

    label, confidence = recognizer.predict(test_image)
    confidence_percentage = 100 - confidence

    if confidence_percentage < confidence_threshold:
        person_name = "Unknown"
    else:
        person_name = label_map.get(label)
    
    return person_name, confidence_percentage, test_image


def display_result(image, name, confidence):
   
    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    text = f"Name: {name}, Confidence: {confidence:.2f}%"
    
    cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction Result", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Menu
if __name__ == "__main__":
    data_path = "C:/Users/AKASH/Desktop/Celebrity Faces Dataset"
    recognizer, label_map = None, {}

    while True:
        print("\nMenu:\n1. Train Model\n2. Recognize Face\n3. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            print("Training model...")
            faces, labels, label_map = load_training_data(data_path)
            recognizer = train_lbph_model(faces, labels)
            print("Model trained!")

        elif choice == '2':
            if not recognizer:
                print("Model not trained. Train it first.")
            else:
                test_image_path = input("Enter test image path: ")
                try:
                    name, confidence, image = recognize_face(recognizer, test_image_path, label_map)
                    print("name:",name,"confindance",confidence)

                    # Display the image with prediction
                    display_result(image, name, confidence)
                except ValueError as e:
                    print(e)

        elif choice == '3':
            print("Exiting.")
            break

        else:
            print("Invalid choice.")


