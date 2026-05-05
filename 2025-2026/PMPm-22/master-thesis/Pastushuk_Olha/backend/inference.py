import torch
import math
import librosa
import numpy as np
import torch.nn as nn
import webrtcvad
from model import EmotionCRNN

class EmotionPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Параметри аудіо з вашого файлу dataset.py
        self.sr = 16000
        self.n_mels = 128
        self.duration = 4.0
        self.samples = int(self.sr * self.duration)
        
        # Ініціалізація VAD
        self.vad = webrtcvad.Vad(3)
        
        # Ініціалізація моделі та завантаження ваг
        self.model = EmotionCRNN(num_outputs=3).to(self.device)
        # Додано weights_only=True для безпеки в нових версіях Torch
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def is_speech_present(self, audio):
        """Перевіряє наявність голосу в уже завантаженому масиві"""
        # Конвертація у 16-бітний формат
        audio_int16 = (audio * 32767).astype(np.int16)
        
        samples_per_frame = int(self.sr * 0.03) # 30ms
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_int16) - samples_per_frame, samples_per_frame):
            frame = audio_int16[i:i + samples_per_frame]
            if self.vad.is_speech(frame.tobytes(), self.sr):
                speech_frames += 1
            total_frames += 1
        
        return (speech_frames / total_frames) > 0.1 if total_frames > 0 else False

    def _preprocess_audio(self, audio):
        """Професійна підготовка аудіо для уникнення помилок 'Angry'"""
        # 1. Нормалізація гучності (Критично!) 
        # Це прибирає піки, які нейронка сприймає як крик/гнів
        if np.max(np.abs(audio)) > 0:
            audio = librosa.util.normalize(audio)

        # 2. Видалення пауз (Trim)
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # 3. Доведення до фіксованої довжини
        if len(audio) < self.samples:
            audio = np.pad(audio, (0, self.samples - len(audio)))
        else:
            audio = audio[:self.samples]
            
        # 4. Створення Мел-спектрограми
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels,
            n_fft=1024, hop_length=256, power=2.0
        )
        
        # 5. Логарифмічне перетворення з фіксованим діапазоном
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        
        # Стандартизація значень (Mean/Std) - допомагає стабілізувати передбачення
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

        mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0)
        return mel_tensor.to(self.device)

    def predict(self, audio_path):
        """Основний метод передбачення емоцій"""
        # Завантажуємо аудіо один раз
        audio, _ = librosa.load(audio_path, sr=self.sr)

        # Спершу перевіряємо наявність голосу
        if not self.is_speech_present(audio):
            return {
                "error": "no_speech",
                "interpretation": "Тиша або шум",
                "valence": 0.0, "arousal": 0.0, "dominance": 0.0
            }

        # Препроцесинг
        mel = self._preprocess_audio(audio)
        
        with torch.no_grad():
            output = self.model(mel)
            v, a, d = output[0].cpu().numpy()
            
            # Обмеження значень в діапазоні [-1, 1] для стабільності інтерпретації
            v = np.clip(v, -1.0, 1.0)
            a = np.clip(a, -1.0, 1.0)
            d = np.clip(d, -1.0, 1.0)
            
        return {
            "valence": float(v),
            "arousal": float(a),
            "dominance": float(d),
            "interpretation": self._interpret_vad(v, a, d)
        }

    def _interpret_vad(self, v, a, d):
        """Евклідова відстань до еталонних емоцій"""
        emo2vad = {
            "Радість (Happiness)": [0.81, 0.51, 0.46],
            "Гордість (Pride)": [0.60, 0.45, 0.85],
            "Інтерес (Interest)": [0.50, 0.25, 0.20],
            "Здивування (Surprise)": [0.40, 0.67, -0.13],
            "Нейтральний стан (Neutral)": [0.00, 0.00, 0.00],
            "Нудьга (Boredom)": [-0.15, -0.35, -0.25],
            "Тривога (Anxiety)": [-0.35, 0.55, -0.45],
            "Зневага (Contempt)": [-0.20, 0.30, 0.70],
            "Провина (Guilt)": [-0.45, 0.30, -0.40],
            "Сором (Shame)": [-0.55, 0.10, -0.55],
            "Гнів (Anger)": [-0.51, 0.59, 0.25],
            "Огида (Disgust)": [-0.60, 0.35, 0.11],
            "Сум (Sadness)": [-0.63, -0.25, -0.33],
            "Страх (Fear)": [-0.64, 0.60, -0.43],
        }

        closest_emotion = "Нейтральний стан"
        min_distance = float('inf')

        for emotion, coords in emo2vad.items():
            distance = math.sqrt(
                (v - coords[0])**2 + (a - coords[1])**2 + (d - coords[2])**2
            )
            if distance < min_distance:
                min_distance = distance
                closest_emotion = emotion

        return closest_emotion

if __name__ == "__main__":
    CHECKPOINT = "checkpoints/emotion_crnn_best.pth"
    TEST_FILE = "test_audio.wav"
    
    import os
    if os.path.exists(CHECKPOINT) and os.path.exists(TEST_FILE):
        predictor = EmotionPredictor(CHECKPOINT)
        result = predictor.predict(TEST_FILE)
        
        if "error" in result:
            print(f"\nРезультат: {result['interpretation']}")
        else:
            print(f"\nАналіз файлу: {TEST_FILE}")
            print("-" * 30)
            print(f"Valence:   {result['valence']:.4f}")
            print(f"Arousal:   {result['arousal']:.4f}")
            print(f"Dominance: {result['dominance']:.4f}")
            print(f"Стан:      {result['interpretation']}")
    else:
        print("Перевірте наявність файлу моделі та тестового аудіо.")