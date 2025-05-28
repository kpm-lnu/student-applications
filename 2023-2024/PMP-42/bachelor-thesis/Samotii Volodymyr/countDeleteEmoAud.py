import pandas as pd
import os

emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']

csv_file = './train_sent_emo.csv'  
df = pd.read_csv(csv_file)
threshold = 250

audio_folder = './train_set/'

df = pd.read_csv(csv_file)

emotion_counts = df['Emotion'].value_counts().to_dict()

deletions_needed = {emotion: max(0, count - threshold) for emotion, count in emotion_counts.items()}

rows_to_delete = []

for index, row in df.iterrows():
    emotion = row['Emotion']
    if deletions_needed.get(emotion, 0) > 0:
        rows_to_delete.append(index)
        deletions_needed[emotion] -= 1

        audio_file = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"

        audio_file_path = os.path.join(audio_folder, audio_file)

        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

df_filtered = df.drop(rows_to_delete)

filtered_csv_file = './filtered_file.csv' 
df_filtered.to_csv(filtered_csv_file, index=False)

print(f"Filtered CSV file saved to {filtered_csv_file}")
#%%
