import os
import shutil
from PIL import Image

# Вхідні директорії
SOURCE_DIRS = [
    r"D:\Programming\Diploma\datasets\combined_dataset\train",
    r"D:\Programming\Diploma\datasets\Oahega"
]
#SOURCE_DIRS = [r"D:\Programming\Diploma\datasets\black_white_dataset\images"]

# Вихідна директорія
DEST_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train"

# Мінімальний розмір зображення
MIN_SIZE = 128

# Підтримувані формати
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Лічильник для статистики
total_copied = 0
skipped_too_small = 0
skipped_errors = 0

for src_dir in SOURCE_DIRS:
    for emotion in os.listdir(src_dir):
        emotion_path = os.path.join(src_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue

        for filename in os.listdir(emotion_path):
            if filename.lower().endswith(IMG_EXTENSIONS):
                src_file = os.path.join(emotion_path, filename)
                try:
                    with Image.open(src_file) as img:
                        width, height = img.size
                        if width < MIN_SIZE or height < MIN_SIZE:
                            skipped_too_small += 1
                            continue

                        # Вихідна папка
                        dest_emotion_dir = os.path.join(DEST_DIR, emotion.lower())
                        os.makedirs(dest_emotion_dir, exist_ok=True)

                        # Уникнути конфлікту імен
                        base_name = os.path.splitext(filename)[0]
                        extension = os.path.splitext(filename)[1]
                        counter = 1
                        dest_file = os.path.join(dest_emotion_dir, filename)
                        while os.path.exists(dest_file):
                            dest_file = os.path.join(dest_emotion_dir, f"{base_name}_{counter}{extension}")
                            counter += 1

                        shutil.copy2(src_file, dest_file)
                        total_copied += 1

                except Exception as e:
                    print(f"⚠️ Помилка з файлом {src_file}: {e}")
                    skipped_errors += 1

# Підсумки
print(f"\n✅ Готово!")
print(f"➤ Скопійовано зображень: {total_copied}")
print(f"⚠️ Пропущено (занадто малі): {skipped_too_small}")
print(f"❌ Пропущено (помилки): {skipped_errors}")
