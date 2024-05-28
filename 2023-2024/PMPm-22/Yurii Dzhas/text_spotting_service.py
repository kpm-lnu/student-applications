import os
import warnings
import numpy as np
import tensorflow.keras as keras
import librosa

MODEL_PATH = "model.h5"
TESTSET_PATH = "testset"
NUM_SAMPLES_TO_CONSIDER = 22050

warnings.filterwarnings('ignore')
class _Text_Spotting_Service:
    model = None
    _mappings = [
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "four",
        "go",
        "happy",
        "house",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "wow",
        "yes",
        "zero"
    ]
    _instance = None

    def predict(self, file_path):
        # extract the MFCCs
        MFCCs = self.preprocess(file_path)  # ( # segments, # coefficients)

        # convert 2d MFCCs array to 4d array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs)
        predictions_index = np.argmax(predictions)
        predicted_text = self._mappings[predictions_index]

        return predicted_text

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load the audiofile
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audiofile
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T


def Text_Spotting_Service():
    # ensure that we only have 1 instance of TSS
    if _Text_Spotting_Service._instance is None:
        _Text_Spotting_Service._instance = _Text_Spotting_Service()
        _Text_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Text_Spotting_Service._instance


if __name__ == "__main__":
    tss = Text_Spotting_Service()
    output = []
    for filename in os.listdir(TESTSET_PATH):
        f = os.path.join(TESTSET_PATH, filename)
        if os.path.isfile(f):
            key_text = tss.predict(TESTSET_PATH + '/' + f.split("\\")[-1])
            output.append(f"Audio file: {filename}. Computed word: {key_text}")
    for i in output: print(i)
