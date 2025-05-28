import kornia as kn
import torch
import cv2
from kornia.contrib import FaceDetector, FaceDetectorResult, FaceKeypoint
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def mark_keypoint(image: np.ndarray, detection: FaceDetectorResult, keypoint_type: FaceKeypoint) -> np.ndarray:
    keypoint = detection.get_keypoint(keypoint_type).int().tolist()
    return cv2.circle(image, keypoint, 2, (255, 0, 0), 2)

def find_faces(raw_image):
    if raw_image is not None:
        image_tensor = kn.image_to_tensor(raw_image, keepdim=False)
        image_tensor = kn.color.bgr_to_rgb(image_tensor.float())

        face_detector = FaceDetector()
        with torch.no_grad():
            detections = face_detector(image_tensor)
        detections = [FaceDetectorResult(det) for det in detections[0]]

        image_copy = raw_image.copy()
        confidence_threshold = 0.95

        for detection in detections:
            if detection.score < confidence_threshold:
                continue

            image_copy = cv2.rectangle(
                image_copy, detection.top_left.int().tolist(), detection.bottom_right.int().tolist(), (255, 0, 0), 4
            )

            image_copy = mark_keypoint(image_copy, detection, FaceKeypoint.EYE_LEFT)
            image_copy = mark_keypoint(image_copy, detection, FaceKeypoint.EYE_RIGHT)
            image_copy = mark_keypoint(image_copy, detection, FaceKeypoint.NOSE)
            image_copy = mark_keypoint(image_copy, detection, FaceKeypoint.MOUTH_LEFT)
            image_copy = mark_keypoint(image_copy, detection, FaceKeypoint.MOUTH_RIGHT)

        return image_copy

def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

image = np.asarray(Image.open("image1.jpg"))
processed_image = find_faces(image)
show_image(processed_image)
