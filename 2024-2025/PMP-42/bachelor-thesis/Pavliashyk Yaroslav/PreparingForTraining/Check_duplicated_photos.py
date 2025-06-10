import os
import shutil
from PIL import Image
import imagehash
from collections import defaultdict
import re

SOURCE_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train"
DEST_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train_unique_modified"
DUPLICATE_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train_duplicates_modified"

HASH_DISTANCE_THRESHOLD = 1  # Чутливість до схожості (зменшити — суворіше)

copied = 0
duplicates = 0

def normalize_name(name):
    # Визначення ключа за іменем: cropped_emotions або color, або просто перші 8 символів
    match1 = re.search(r'cropped_emotions\.(.{6})', name)
    if match1:
        return match1.group(0)

    match2 = re.search(r'color_(.{5})', name)
    if match2:
        return match2.group(0)

    return name[:8]

for root, _, files in os.walk(SOURCE_DIR):
    rel_path = os.path.relpath(root, SOURCE_DIR)
    dest_class_dir = os.path.join(DEST_DIR, rel_path)
    dup_class_dir = os.path.join(DUPLICATE_DIR, rel_path)
    os.makedirs(dest_class_dir, exist_ok=True)
    os.makedirs(dup_class_dir, exist_ok=True)

    image_hashes = {}
    file_to_hash = {}
    name_groups = defaultdict(list)

    # === 1. Групування по normalized name ===
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(root, file)
        try:
            with Image.open(path) as img:
                gray_img = img.convert("L")
                img_hash = imagehash.phash(gray_img)
                image_hashes[file] = img_hash
                file_to_hash[file] = img_hash
                key = normalize_name(file)
                name_groups[key].append(file)
        except Exception as e:
            print(f"⚠️ Проблема з {file}: {e}")
            continue

    processed = set()

    # === 2. Усередині кожної групи перевіряємо схожість ===
    for group_files in name_groups.values():
        visual_groups = []
        used = set()

        for i, file1 in enumerate(group_files):
            if file1 in used:
                continue
            current_group = [file1]
            used.add(file1)
            for file2 in group_files[i+1:]:
                if file2 in used:
                    continue
                if file1 in file_to_hash and file2 in file_to_hash:
                    dist = file_to_hash[file1] - file_to_hash[file2]
                    if dist <= HASH_DISTANCE_THRESHOLD:
                        current_group.append(file2)
                        used.add(file2)
            visual_groups.append(current_group)

        # === 3. Копіювання: перший залишаємо, решта — в дублікатну папку ===
        for group in visual_groups:
            for i, f in enumerate(group):
                src_path = os.path.join(root, f)
                if i == 0:
                    dest_path = os.path.join(dest_class_dir, f)
                    shutil.copy2(src_path, dest_path)
                    copied += 1
                else:
                    dup_path = os.path.join(dup_class_dir, f)
                    shutil.copy2(src_path, dup_path)
                    duplicates += 1

# === РЕЗУЛЬТАТ ===
print("\n🎯 Завершено:")
print(f"✅ Залишено унікальних зображень: {copied}")
print(f"🚫 Виявлено та переміщено дублікатів: {duplicates}")
