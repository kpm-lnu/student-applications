import dlib
import cv2
import numpy as np
from PIL import Image

# Ініціалізація детектора облич та моделі визначення ключових точок
face_detector = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def process_image(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    detected_faces = face_detector(img_bgr)
    for face in detected_faces:
        landmarks = landmark_model(img_bgr, face)

        cv2.rectangle(img_bgr, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 3)

        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(img_bgr, (x, y), 2, (255, 0, 0), -1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

image = Image.open("image1.jpg")

processed_image = process_image(image)

cv2.imshow("Face and Key Points", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
