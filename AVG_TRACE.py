import cv2
import matplotlib.pyplot as plt


img = cv2.imread("", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Blurred Image")


cv2.createTrackbar("Kernel Size", "Blurred Image", 1, 50, lambda:x=none)

while True:
    
    kernel_size = cv2.getTrackbarPos("Kernel Size", "Blurred Image")
    
 
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    img_blur = cv2.blur(img, (kernel_size, kernel_size))
    cv2.imshow("Blurred Image", img_blur)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()





import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv.imread('PL.jpeg', cv.IMREAD_GRAYSCALE)

# Define filtering functions
def apply_average_filtering(image):
    return cv.blur(image, (11, 11))

def median(image, kernel):
    return cv.medianBlur(image, kernel)

def gaussian(image, kernel, sigma):
    return cv.GaussianBlur(image, (kernel, kernel), sigma)

# Define edge detection functions
def sobel(image):
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, 3)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, 3)
    combined = cv.magnitude(sobel_x, sobel_y)
    return np.uint8(combined)

def canny(image):
    blurred = cv.GaussianBlur(image, (5, 5), 1.4)
    edges = cv.Canny(blurred, threshold1=100, threshold2=200)
    return edges

# User menu
def menu():
    while True:
        print("\nMenu:")
        print("1. Image Filtering")
        print("2. Edge Detection")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            print("\nImage Filtering Options:")
            print("1. Average Filter")
            print("2. Median Filter")
            print("3. Gaussian Filter")
            filter_choice = input("Select a filter: ")

            if filter_choice == "1":
                filtered_image = apply_average_filtering(image)
                cv.imshow("Average Filter", filtered_image)
            elif filter_choice == "2":
                filtered_image = median(image, 11)
                cv.imshow("Median Filter", filtered_image)
            elif filter_choice == "3":
                filtered_image = gaussian(image, 5, 0)
                cv.imshow("Gaussian Filter", filtered_image)
            else:
                print("Invalid choice. Please try again.")
        
        elif choice == "2":
            print("\nEdge Detection Options:")
            print("1. Sobel")
            print("2. Canny")
            edge_choice = input("Select an edge detection method: ")

            if edge_choice == "1":
                edges = sobel(image)
                cv.imshow("Sobel Edge Detection", edges)
            elif edge_choice == "2":
                edges = canny(image)
                cv.imshow("Canny Edge Detection", edges)
            else:
                print("Invalid choice. Please try again.")

        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

        # Wait for the user to close the current window
        cv.waitKey(0)
        cv.destroyAllWindows()

# Run the menu
menu()
