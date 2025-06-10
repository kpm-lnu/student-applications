import os
from PIL import Image

#ROOT_DIR = r"D:\Programming\Diploma\datasets\Oahega"
#ROOT_DIR = r"D:\Programming\Diploma\datasets\combined_dataset"
ROOT_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train"

min_width = float('inf')
min_height = float('inf')
image_count = 0
small_image_count = 0 


for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, file)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_count += 1
                    print(f"[{image_count}] 📷 {file} → {width}x{height}")

                    min_width = min(min_width, width)
                    min_height = min(min_height, height)

                    if width < 128 or height < 128:
                        small_image_count += 1

            except Exception as e:
                print(f"⚠️ Проблема з файлом {image_path}: {e}")

if image_count > 0:
    print("\n📊 Статистика:")
    print(f"🔢 Загальна кількість зображень: {image_count}")
    print(f"🔍 Мінімальний розмір: ширина = {min_width}, висота = {min_height}")
    print(f"⚠️ Кількість зображень менше ніж 128×128: {small_image_count}")
else:
    print("❌ Зображення не знайдено.")
