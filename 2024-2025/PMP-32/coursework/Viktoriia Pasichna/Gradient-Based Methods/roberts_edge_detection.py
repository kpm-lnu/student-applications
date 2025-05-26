import cv2 
import numpy as np
from scipy import ndimage

def roberts_cross_edge_detection(image_path, output_path):
    roberts_cross_v = np.array([[1, 0],
                               [0, -1]])
    
    roberts_cross_h = np.array([[0, 1],
                               [-1, 0]])
    img = cv2.imread(image_path, 0).astype('float64')
    img /= 255.0
    
    vertical = ndimage.convolve(img, roberts_cross_v)
    horizontal = ndimage.convolve(img, roberts_cross_h)
    
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    edged_img *= 255
    
    cv2.imwrite(output_path, edged_img)
    return edged_img

if __name__ == "__main__":
    input_path = "Images/photo_cat.jpg"
    output_path = "Outputs/roberts_edge_detection.jpg"
    result = roberts_cross_edge_detection(input_path, output_path)
    print("Edge detection completed. Result saved to:", output_path) 