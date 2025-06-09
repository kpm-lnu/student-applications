import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel(img):

    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    filter_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    filter_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    S_x = cv2.filter2D(img, -1, filter_x)
    S_y = cv2.filter2D(img, -1, filter_y)
    
    grad = np.sqrt(S_x ** 2 + S_y ** 2)
    
    grad = np.clip(grad, 0, 255)
    
    return grad.astype(np.uint8)

def main():
    img = cv2.imread('Images/photo_cat.jpg')
    if img is None:
        print("Помилка: Не вдалося завантажити зображення")
        return
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edge = sobel(img_gray)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Оригінальне зображення')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Зображення в градаціях сірого')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(edge, cmap='gray')
    plt.title('Виявлені краї (оператор Собеля)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('Outputs/sobel_edge_detection.png')
    print("Результати збережено як 'sobel_edge_detection.png'")

if __name__ == "__main__":
    main() 