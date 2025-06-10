import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from data.dataset import load_dataset

def train_model():
    # Завантаження даних
    X, y = load_dataset()

    num_nodes = len(set(y.tolist() + X[:, :2].flatten().astype(int).tolist()))

    # Перетворення y в one-hot
    y_cat = to_categorical(y, num_classes=num_nodes)

    # Побудова моделі
    model = Sequential([
        Dense(256, activation='relu', input_shape=(4,)),
        Dense(256, activation='relu'),
        Dense(num_nodes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Навчання
    model.fit(X, y_cat, epochs=1000, batch_size=8, verbose=1)

    # Збереження
    model.save('model/modelN2(1000).h5')
    print("[✓] Модель збережено в model/model.h5")


if __name__ == '__main__':
    train_model()