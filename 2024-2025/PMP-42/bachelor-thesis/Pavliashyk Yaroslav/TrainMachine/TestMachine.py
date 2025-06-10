import os
import shutil
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Шляхи ===
MODEL_PATH = r"D:\Programming\Diploma\TrainMachine\best_emotion_model_valacc.keras"
TEST_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\test"
OUTPUT_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\test_result"
IMG_SIZE = (128, 128)

# === Класи емоцій ===
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === Завантаження моделі ===
model = load_model(MODEL_PATH)

# === Створення директорій для результатів ===
for label in EMOTION_LABELS:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# === Класифікація зображень ===
for filename in os.listdir(TEST_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    path = os.path.join(TEST_DIR, filename)
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)  # ✅ Новий антиаліасинг
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        prediction = model.predict(arr, verbose=0)
        label_index = np.argmax(prediction)
        label_name = EMOTION_LABELS[label_index]

        # === Зберегти результат ===
        dest_path = os.path.join(OUTPUT_DIR, label_name, filename)
        shutil.copy(path, dest_path)
        print(f"✅ {filename} → {label_name}")

    except Exception as e:
        print(f"❌ Помилка з {filename}: {e}")

print("\n🎉 Класифікацію завершено!")
