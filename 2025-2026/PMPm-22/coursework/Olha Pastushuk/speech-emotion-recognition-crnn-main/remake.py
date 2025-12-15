import os
import pandas as pd

AUDIO_TRAIN = "audio/audio_train"
AUDIO_VAL = "audio/audio_val"
OUT_DIR = "data"

emo2va = {
    "01": [0.0, 0.1],   # neutral
    "02": [0.2, 0.2],   # calm
    "03": [0.8, 0.6],   # happy
    "04": [-0.6, -0.3], # sad
    "05": [-0.7, 0.8],  # angry
    "06": [-0.8, 0.9],  # fear
    "07": [-0.6, 0.5],  # disgust
    "08": [0.3, 0.7],   # surprise
}

def collect_rows(audio_dir):
    rows = []
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.endswith(".wav"):
                parts = f.split("-")
                emo_code = parts[2]
                val, aro = emo2va[emo_code]
                path = os.path.join(root, f)
                rows.append([path, val, aro])
    return rows

# зібрати окремо train і val
train_rows = collect_rows(AUDIO_TRAIN)
val_rows = collect_rows(AUDIO_VAL)

# зберегти CSV
pd.DataFrame(train_rows, columns=["path","valence","arousal"]).to_csv(
    os.path.join(OUT_DIR, "labels_train.csv"), index=False
)
pd.DataFrame(val_rows, columns=["path","valence","arousal"]).to_csv(
    os.path.join(OUT_DIR, "labels_val.csv"), index=False
)
