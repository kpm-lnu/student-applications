import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_edges_dog(image, sigma1=1.0, sigma2=1.6):
   
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    
    dog = blur1 - blur2
    
    _, edges = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY)
    
    return edges

def visualize_results(original, edges, output_path):
    os.makedirs('Outputs', exist_ok=True)
    
    cv2.imwrite(output_path, edges)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Оригінальне зображення')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title('Виявлені контури')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    image = cv2.imread("Images/photo_cat.jpg")
    
    if image is None:
        print("Помилка: Не вдалося завантажити зображення")
        exit()
    
    edges = detect_edges_dog(image)
    
    output_path = os.path.join('Outputs', 'dog_result.jpg')
    
    visualize_results(image, edges, output_path)
    
    print(f"Результат збережено у файл: {output_path}") 