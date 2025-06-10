import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
TEST_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train_unique"  
MODEL_PATH = r"D:\Programming\Diploma\TrainMachine\best_emotion_model_valacc.keras"
PREDICTIONS_FILE = "predictions.json"

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

model = load_model(MODEL_PATH)

y_prob = model.predict(test_generator)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

with open(PREDICTIONS_FILE, "w") as f:
    json.dump({"y_true": y_true.tolist(), "y_pred": y_pred.tolist()}, f)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Матриця плутанини")
plt.xlabel("Передбачено")
plt.ylabel("Факт")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nЗвіт класифікації:")
print(classification_report(y_true, y_pred, target_names=class_names))
