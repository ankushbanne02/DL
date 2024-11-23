import cv2 as cv
import numpy as np


image = cv.imread('C:/Users/AKASH/Desktop/download.png', cv.IMREAD_GRAYSCALE)

# 
def apply_average_filtering(image):
    
    kernel_size = 11
    
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    
    filtered_image = np.zeros_like(image)

  
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
           
            roi = padded_image[i:i + kernel_size, j:j + kernel_size]
            
            filtered_image[i, j] = np.mean(roi)

    return filtered_image


def median(image, kernel_size):
    
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    
    filtered_image = np.zeros_like(image)

    # Iterate over the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
           
            roi = padded_image[i:i + kernel_size, j:j + kernel_size]
            
            filtered_image[i, j] = np.median(roi)

    return filtered_image


def gaussian(image, kernel, sigma):
    return cv.GaussianBlur(image, (kernel, kernel), sigma)


def sobel(image):
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, 3)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, 3)
    combined = cv.magnitude(sobel_x, sobel_y)
    return np.uint8(combined)

def canny(image):
    blurred = cv.GaussianBlur(image, (5, 5), 1.4)
    edges = cv.Canny(blurred, threshold1=100, threshold2=200)
    return edges


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

        
        cv.waitKey(0)
        cv.destroyAllWindows()


menu()
