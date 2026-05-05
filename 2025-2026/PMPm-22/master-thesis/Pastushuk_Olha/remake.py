import os
import pandas as pd
import soundfile as sf
import librosa
import io
from datasets import load_dataset, concatenate_datasets, Audio

# 1. Налаштування шляхів
BASE_AUDIO_DIR = "audio"
OUT_DIR = "data"

for p in [BASE_AUDIO_DIR, OUT_DIR, 
          os.path.join(BASE_AUDIO_DIR, "audio_train"), 
          os.path.join(BASE_AUDIO_DIR, "audio_val")]:
    os.makedirs(p, exist_ok=True)

# 2. Словник VAD
emo2vad = {
    "happiness": [ 0.81,  0.51,  0.46],
    "pride":     [ 0.60,  0.45,  0.85],
    "interest":  [ 0.50,  0.25,  0.20],
    "surprise":  [ 0.40,  0.67, -0.13],
    "neutral":   [ 0.00,  0.00,  0.00],
    "boredom":   [-0.15, -0.65, -0.25],
    "anxiety":   [-0.35,  0.55, -0.45],
    "contempt":  [-0.20,  0.30,  0.70],
    "guilt":     [-0.45,  0.30, -0.40],
    "shame":     [-0.55,  0.10, -0.55],
    "anger":     [-0.51,  0.59,  0.25],
    "disgust":   [-0.60,  0.35,  0.11],
    "sadness":   [-0.63, -0.27, -0.33],
    "fear":      [-0.64,  0.60, -0.43],
}

CAMEO_EMOTIONS = [
    "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral",
    "contempt", "pride", "shame", "guilt", "boredom", "anxiety", "interest"
]

def get_data():
    print("--- Етап 1: Завантаження датасету CAMEO ---")
    try:
        ds_dict = load_dataset("amu-cai/CAMEO")
        full_ds = concatenate_datasets([ds_dict[k] for k in ds_dict.keys()])
        full_ds = full_ds.cast_column("audio", Audio(decode=False))
        
        print(f"Успішно зібрано {len(full_ds)} записів.")
        return full_ds.train_test_split(test_size=0.1, seed=42)
    except Exception as e:
        print(f"Помилка завантаження: {e}")
        return None

def save_data(dataset_split, subdir_name):
    print(f"\n--- Етап 2: Обробка {subdir_name} ---")
    rows = []

    for i, item in enumerate(dataset_split):
        try:
            emo_raw = item['emotion']
            emo_name = str(emo_raw).lower().strip()
            
            if emo_name not in emo2vad:
                continue
                
            v, a, d = emo2vad[emo_name]
            
            filename = f"{i}_{emo_name}.wav"
            target_subdir = f"audio_{subdir_name}"
            full_path = os.path.join(BASE_AUDIO_DIR, target_subdir, filename)
            
            audio_bytes = item['audio']['bytes']
            
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            sf.write(full_path, y, sr)
            
            rows.append([full_path, v, a, d])
            
        except Exception as e:
            if i == 0:
                print(f"DEBUG Помилка на індексі {i}: {e}")
            continue
            
        if i % 500 == 0 and i > 0:
            print(f"Прогрес [{subdir_name}]: {i} файлів оброблено... (в списку вже {len(rows)})")
            
    return rows

def main():
    split_ds = get_data()
    if not split_ds:
        return

    train_rows = save_data(split_ds['train'], "train")
    val_rows = save_data(split_ds['test'], "val")

    print("\n--- Етап 3: Запис CSV ---")
    cols = ["path", "valence", "arousal", "dominance"]
    pd.DataFrame(train_rows, columns=cols).to_csv("data/labels_train.csv", index=False)
    pd.DataFrame(val_rows, columns=cols).to_csv("data/labels_val.csv", index=False)
    print(f"Готово! Оброблено всього {len(train_rows) + len(val_rows)} файлів.")

if __name__ == "__main__":
    main()