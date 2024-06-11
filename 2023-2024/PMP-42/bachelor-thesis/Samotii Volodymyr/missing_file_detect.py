import pandas as pd
import os

def find_missing_and_duplicate_files(csv_path, files_directory):

    df = pd.read_csv(csv_path)

    duplicates = df[df.duplicated(subset=['Dialogue_ID', 'Utterance_ID'], keep=False)]

    missing_files = []
    for index, row in df.iterrows():
        dia = row['Dialogue_ID']
        utt = row['Utterance_ID']
        file_name = f"dia{dia}_utt{utt}.wav"
        file_path = os.path.join(files_directory, file_name)

        if not os.path.isfile(file_path):
            missing_files.append(file_name)

    return missing_files, duplicates


csv_path = './dev_sent_emo.csv'
files_directory = './dev_set'

missing_files, duplicates = find_missing_and_duplicate_files(csv_path, files_directory)

if missing_files:
    print("Missing files:")
    for file in missing_files:
        print(file)
else:
    print("No files are missing.")

if not duplicates.empty:
    print("\nDuplicate rows found:")
    print(duplicates)
else:
    print("\nNo duplicate rows found.")