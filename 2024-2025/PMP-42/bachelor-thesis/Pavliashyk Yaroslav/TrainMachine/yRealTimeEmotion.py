import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


MODEL_PATH = r"D:\Programming\Diploma\TrainMachine\best_model_finetuned.keras"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (128, 128)


model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


cap = cv2.VideoCapture(0)
print("🎥 Запуск камери... Натисни 'q' щоб вийти.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_resized = face_pil.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        face_array = img_to_array(face_resized) / 255.0
        face_array = np.expand_dims(face_array, axis=0)


        prediction = model.predict(face_array, verbose=0)
        label = EMOTION_LABELS[np.argmax(prediction)]
        confidence = np.max(prediction)


        label_text = f"{label} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
