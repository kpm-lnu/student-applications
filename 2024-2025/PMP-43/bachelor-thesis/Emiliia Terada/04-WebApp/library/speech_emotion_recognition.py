## Basics ##
import time
import os
import numpy as np

## Audio Preprocessing ##
import pyaudio
import wave
import librosa
from scipy.stats import zscore

## Time Distributed CNN ##
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM


'''
Speech Emotion Recognition
'''
class speechEmotionRecognition:

    def __init__(self, subdir_model=None):
        """
        Якщо вказано шлях subdir_model, будуємо модель і завантажуємо збережені ваги.
        """
        if subdir_model is not None:
            # build_model визначений нижче
            self._model = self.build_model()
            # завантажуємо ваги з файлу або директорії
            # припускаємо, що subdir_model — це шлях до файлу з вагами (.h5 або .ckpt)
            self._model.load_weights(subdir_model)

        # Мапування індексів на назви емоцій
        self._emotion = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
        }


    def build_model(self):
        """
        Будує TimeDistributed CNN + LSTM архітектуру.
        Повертає компільовану Keras-модель.
        """
        # Очищаємо попередню сесію Keras/TensorFlow
        K.clear_session()

        # Вхід: (кількість фрагментів, n_mels, timesteps, 1)
        # У цьому прикладі ми припускаємо, що кожний приклад має форму (5, 128, 128, 1).
        input_y = Input(shape=(5, 128, 128, 1), name='Input_MELSPECT')

        # Перший локальний блок (LFLB)
        y = TimeDistributed(
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            name='Conv_1_MELSPECT'
        )(input_y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
        y = TimeDistributed(
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            name='MaxPool_1_MELSPECT'
        )(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)

        # Другий локальний блок (LFLB)
        y = TimeDistributed(
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            name='Conv_2_MELSPECT'
        )(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
        y = TimeDistributed(
            MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'),
            name='MaxPool_2_MELSPECT'
        )(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)

        # Третій локальний блок (LFLB)
        y = TimeDistributed(
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            name='Conv_3_MELSPECT'
        )(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
        y = TimeDistributed(
            MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'),
            name='MaxPool_3_MELSPECT'
        )(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

        # Четвертий локальний блок (LFLB)
        y = TimeDistributed(
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            name='Conv_4_MELSPECT'
        )(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
        y = TimeDistributed(
            MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'),
            name='MaxPool_4_MELSPECT'
        )(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)

        # Флаттен кожного фрагменту
        y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)

        # LSTM шар
        y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)

        # Повнозв’язний шар з softmax для 7 класів емоцій
        y = Dense(7, activation='softmax', name='FC')(y)

        # Створюємо модель
        model = Model(inputs=input_y, outputs=y)

        # Компілюємо під задачу класифікації
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


    def voice_recording(self, filename, duration=5, sample_rate=16000, chunk=1024, channels=1):
        """
        Запис голосу у WAV-файл тривалістю duration секунд.
        """
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk
        )

        frames = []
        print('* Start Recording *')
        stream.start_stream()
        start_time = time.time()
        current_time = time.time()

        while (current_time - start_time) < duration:
            data = stream.read(chunk)
            frames.append(data)
            current_time = time.time()

        stream.stop_stream()
        stream.close()
        p.terminate()
        print('* End Recording *')

        wf = wave.open(filename, 'w')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()


    def mel_spectrogram(self, y, sr=16000, n_fft=512, win_length=256,
                        hop_length=128, window='hamming', n_mels=128, fmax=4000):
        """
        Обчислення лог-мел-спектрограми з перевіркою на NaN/Inf.
        """
        # 1. Перевірка на NaN/Inf і очищення
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y)

        # 2. Якщо сигнал став повністю нульовим, повертаємо матрицю нулів
        if not np.any(y):
            time_steps = len(y) // hop_length + 1
            return np.zeros((n_mels, time_steps), dtype=np.float32)

        # 3. Обчислюємо STFT
        S = np.abs(
            librosa.stft(y, n_fft=n_fft, window=window,
                         win_length=win_length, hop_length=hop_length)
        ) ** 2

        # 4. Обчислюємо мел-спектрограми
        mel_spect = librosa.feature.melspectrogram(
            S=S,
            sr=sr,
            n_mels=n_mels,
            fmax=fmax
        )

        # 5. Перетворюємо в лог-мел
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        return np.asarray(mel_spect, dtype=np.float32)


    def frame(self, y, win_step=64, win_size=128):
        """
        Розбиває сигнал y (форми (1, 1, T)) на фрейми розміру win_size з кроком win_step.
        """
        nb_frames = 1 + int((y.shape[2] - win_size) / win_step)
        frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)

        for t in range(nb_frames):
            frames[:, t, :, :] = np.copy(
                y[:, :, (t * win_step):(t * win_step + win_size)]
            ).astype(np.float16)

        return frames


    def predict_emotion_from_file(self, filename, chunk_step=16000,
                                  chunk_size=49100, predict_proba=False, sample_rate=16000):
        """
        Розбиває аудіо файл на шматки, обчислює мел-спектрограми, передбачає емоції.
        """
        # 1. Завантажуємо WAV-файл
        y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)

        # 2. Розбиваємо сигнал на шматки
        chunks = self.frame(y.reshape(1, 1, -1), chunk_step, chunk_size)

        # 3. Приводимо до форми (num_chunks, chunk_size)
        chunks = chunks.reshape(chunks.shape[1], chunks.shape[-1])

        # 4. Z-нормалізація кожного фрагмента
        y_chunks = np.asarray(list(map(zscore, chunks)))

        # 5. Обчислюємо мел-спектрограми
        mel_spect_list = np.asarray(list(map(self.mel_spectrogram, y_chunks)))

        # 6. Застосовуємо frame до мел-спектрограм (щоб отримати TimeDistributed-зображення)
        mel_spect_ts = self.frame(mel_spect_list)

        # 7. Переформотовуємо у X для TimeDistributed CNN
        X = mel_spect_ts.reshape(
            mel_spect_ts.shape[0],
            mel_spect_ts.shape[1],
            mel_spect_ts.shape[2],
            mel_spect_ts.shape[3],
            1
        )

        # 8. Передбачуємо
        if predict_proba:
            predictions = self._model.predict(X)
        else:
            raw_preds = self._model.predict(X)
            idxs = np.argmax(raw_preds, axis=1)
            predictions = [self._emotion.get(i) for i in idxs]

        # 9. Очищаємо сесію Keras (усуваємо зайві графи)
        K.clear_session()

        # 10. Обчислюємо часові мітки (у секундах)
        timestamp = np.concatenate(
            [[chunk_size], np.ones((len(predictions) - 1)) * chunk_step]
        ).cumsum()
        timestamp = np.round(timestamp / sample_rate)

        return [predictions, timestamp]


    def prediction_to_csv(self, predictions, filename, mode='w'):
        """
        Зберігає передбачені емоції у CSV-файл.
        """
        with open(filename, mode) as f:
            if mode == 'w':
                f.write("EMOTIONS\n")
            for emo in predictions:
                f.write(str(emo) + '\n')
        # Файл автоматично закривається завдяки контекстному менеджеру

