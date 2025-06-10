import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

# === Конфігурація ===
TEST_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train_unique"
PRED_JSON = "predictions.json"
FILENAMES = "filenames.json"  # Якщо є
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_EXAMPLES = 20

# === Завантаження даних ===
with open(PRED_JSON, "r") as f:
    data = json.load(f)
y_true = np.array(data["y_true"])
y_pred = np.array(data["y_pred"])

with open(FILENAMES, "r") as f:
    filenames = json.load(f)

# === Індекси правильних / помилкових ===
correct_idx = np.where(y_true == y_pred)[0]
incorrect_idx = np.where(y_true != y_pred)[0]

# === Вивід прикладів ===
def show_image(idx):
    path = os.path.join(TEST_DIR, filenames[idx])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    true_label = CLASS_NAMES[y_true[idx]]
    pred_label = CLASS_NAMES[y_pred[idx]]
    plt.title(f"Факт: {true_label} | Передбачено: {pred_label}")
    plt.axis("off")
    plt.show()

print("\n✅ Правильні передбачення:")
for i in correct_idx[:NUM_EXAMPLES]:
    show_image(i)

print("\n❌ Помилкові передбачення:")
for i in incorrect_idx[:NUM_EXAMPLES]:
    show_image(i)
