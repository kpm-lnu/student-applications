import os
import shutil
from PIL import Image

# Вхідна коренева папка
ROOT_INPUT = r"D:\Programming\Diploma\datasets\black_white_dataset\images"
# Вихідна папка
ROOT_OUTPUT = r"D:\Programming\Diploma\NewDatasetDirectory\train"
# Мінімальний розмір
MIN_SIZE = 128
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Мапа для перейменування емоцій
EMOTION_MAP = {
    "anger": "angry",
    "surprised": "surprise"
    # інші залишаються без змін
}

# Лічильники імен
rename_counters = {}

# Обхід усіх вкладених папок (0, 1, 2, ...)
for folder_name in os.listdir(ROOT_INPUT):
    folder_path = os.path.join(ROOT_INPUT, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith(IMG_EXTENSIONS):
            emotion_raw = os.path.splitext(file)[0]       # Наприклад: "Anger"
            emotion_key = emotion_raw.lower()             # "anger"

            # Визначаємо папку призначення через мапу
            emotion_folder = EMOTION_MAP.get(emotion_key, emotion_key)

            src_file = os.path.join(folder_path, file)
            dest_dir = os.path.join(ROOT_OUTPUT, emotion_folder)

            if not os.path.exists(dest_dir):
                print(f"⚠️ Пропущено: немає папки {dest_dir}")
                continue

            try:
                with Image.open(src_file) as img:
                    width, height = img.size
                    if width < MIN_SIZE or height < MIN_SIZE:
                        print(f"⚠️ Маленьке зображення: {src_file}")
                        continue

                    # Лічильник імен
                    if emotion_folder not in rename_counters:
                        rename_counters[emotion_folder] = 1
                    else:
                        rename_counters[emotion_folder] += 1

                    new_filename = f"{emotion_raw}{rename_counters[emotion_folder]}.jpg"
                    dest_path = os.path.join(dest_dir, new_filename)

                    shutil.copy2(src_file, dest_path)
                    print(f"✅ {file} → {dest_path}")

            except Exception as e:
                print(f"❌ Помилка з {src_file}: {e}")

print("\n🎉 Завершено.")
