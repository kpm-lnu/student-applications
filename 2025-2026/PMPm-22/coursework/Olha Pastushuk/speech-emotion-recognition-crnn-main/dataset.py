import os
import random
import numpy as np
import torch
import torch.utils.data as data
import librosa
import pandas as pd


def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_shift(audio, shift_max=0.2):
    shift = int(random.random() * shift_max * len(audio))
    return np.roll(audio, shift)

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)

def time_stretch(audio, rate=1.0):
    try:
        return librosa.effects.time_stretch(audio, rate)
    except Exception:
        return audio
    

def augment_audio(y, sr):
    # 1. Додаємо випадковий білий шум
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y = y + noise_amp * np.random.normal(size=y.shape[0])

    # 2. Pitch shift (+/- 1 півтону)
    pitch_change = np.random.uniform(-1, 1)
    y = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change)

    # 3. Speed/tempo shift (+/- 10%)
    speed_change = np.random.uniform(0.9, 1.1)
    y = librosa.effects.time_stretch(y, speed_change)

    return y

class EmotionVADDataset(data.Dataset):
    """Dataset that returns mel-spectrograms and valence/arousal targets.

    Expected CSV format: path,valence,arousal
    """

    def __init__(self, csv_path, sr=16000, n_mels=128, duration=4.0, augment=False):
        self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.n_mels = n_mels
        self.samples = int(sr * duration)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr)
        # mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # pad/crop
        if len(audio) < self.samples:
            audio = np.pad(audio, (0, self.samples - len(audio)))
        else:
            audio = audio[:self.samples]
        return audio

    def compute_mel(self, audio):
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=self.n_mels,
                                             n_fft=1024, hop_length=256, power=2.0)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        valence = float(row['valence'])
        arousal = float(row['arousal'])

        audio = self.load_audio(path)

        # augmentation (on-the-fly)
        if self.augment:
            if random.random() < 0.3:
                audio = add_noise(audio, noise_factor=0.005)
            if random.random() < 0.2:
                audio = time_shift(audio, shift_max=0.2)
            if random.random() < 0.2:
                # random pitch between -2 and 2 semitones
                n_steps = random.uniform(-2, 2)
                audio = pitch_shift(audio, self.sr, n_steps=n_steps)
            if random.random() < 0.15:
                rate = random.uniform(0.9, 1.1)
                audio = time_stretch(audio, rate=rate)
                # ensure length
                if len(audio) < self.samples:
                    audio = np.pad(audio, (0, self.samples - self.samples))
                else:
                    audio = audio[:self.samples]

        mel = self.compute_mel(audio)
        mel = torch.tensor(mel).unsqueeze(0)  # (1, n_mels, time)
        target = torch.tensor([valence, arousal], dtype=torch.float32)
        return mel, target