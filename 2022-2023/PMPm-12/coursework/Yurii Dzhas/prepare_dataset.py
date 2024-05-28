import librosa as lbr
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050  # 1 sec worth of sound


def preprocess_dataset(dataset_path, json_path, num_mfcc=13, hop_length=512, n_fft=2048):
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }
    # loop through all the sub-dirs
    for i, (dirpath, dirnames, filesnames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            # update mappings
            category = dirpath.split("\\")[-1]  # dataset\\down -> [dataset, down]
            data["mappings"].append(category)
            print(f"Processing {category}")
            # loop through all filenames and extract MFCCs
            for f in filesnames:
                # get file path
                file_path = os.path.join(dirpath, f)
                # load audio files
                signal, sr = lbr.load(file_path)
                # ensure the audio file at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    signal = signal[:SAMPLES_TO_CONSIDER]
                    # extract the MFCCs
                    MFCCs = lbr.feature.mfcc(y=signal, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    data["labels"].append(i - 1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}: {i - 1}")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)
