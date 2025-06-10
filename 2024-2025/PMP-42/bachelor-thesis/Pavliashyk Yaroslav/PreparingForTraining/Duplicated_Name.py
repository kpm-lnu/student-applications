import os
import re
import shutil
from collections import defaultdict

SOURCE_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train"
DEST_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train_unique_modified"
DUP_DIR = r"D:\Programming\Diploma\NewDatasetDirectory\train_duplicates_modified"
LOG_FILE = r"D:\Programming\Diploma\NewDatasetDirectory\duplicates_log.txt"

def get_key(filename):
    match = re.search(r"cropped_emotions\.\w{6}", filename)
    if match:
        return match.group()
    else:
        return filename[:8]

copied = 0
duplicates = 0
log_lines = []

with open(LOG_FILE, "w", encoding="utf-8") as log_file:
    for root, _, files in os.walk(SOURCE_DIR):
        group_map = defaultdict(list)
        rel_path = os.path.relpath(root, SOURCE_DIR)
        dest_class_dir = os.path.join(DEST_DIR, rel_path)
        dup_class_dir = os.path.join(DUP_DIR, rel_path)
        os.makedirs(dest_class_dir, exist_ok=True)
        os.makedirs(dup_class_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                key = get_key(file)
                group_map[key].append(file)

        for key, file_list in group_map.items():
            for idx, file in enumerate(file_list):
                src_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(src_path, SOURCE_DIR)

                if idx == 0:
                    dest_path = os.path.join(dest_class_dir, file)
                    shutil.copy2(src_path, dest_path)
                    copied += 1
                else:
                    dup_path = os.path.join(dup_class_dir, file)
                    shutil.copy2(src_path, dup_path)
                    duplicates += 1
                    log_file.write(f"{rel_file_path} [KEY: {key}]\n")

print("\n✅ Завершено!")
print(f"✔️ Унікальних зображень: {copied}")
print(f"❌ Дублікатів перенесено: {duplicates}")
print(f"📝 Лог записано у: {LOG_FILE}")
