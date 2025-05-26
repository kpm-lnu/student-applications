import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def LoG_filter_opencv(image, sigma, size=None):
    if size is None:
        size = int(6 * sigma + 1) if sigma >= 1 else 7

    if size % 2 == 0:
        size += 1

    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(np.abs(kernel))

    result = cv2.filter2D(image, -1, kernel)

    return result

def visualize_results(original, filtered):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(original, cmap='gray')
    plt.title('Оригінальне зображення')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(filtered, cmap='gray')
    plt.title('LoG відфільтроване зображення')
    plt.axis('off')
    
    plt.tight_layout()

    if not os.path.exists('results'):
        os.makedirs('results')

    plt.savefig('Outputs/log_result.jpg')
    plt.close()

if __name__ == "__main__":
    image_path = "Images/photo_cat.jpg"
    sigma = 2.0

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    filtered_image = LoG_filter_opencv(image, sigma)
    filtered_image = cv2.convertScaleAbs(filtered_image)

    visualize_results(image, filtered_image)
    print("Зображення збережено в папку 'Outputs/log_result.ipg'") 