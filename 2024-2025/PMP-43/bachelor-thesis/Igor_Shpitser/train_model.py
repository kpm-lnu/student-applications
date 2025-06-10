import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks, regularizers, losses, optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS']   = '0'

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X_raw = np.array([d["features"][4:8] for d in data], dtype=np.float32)
Y_raw = np.array([d["labels"]      for d in data], dtype=np.float32)

scaler_X = StandardScaler().fit(X_raw)
scaler_Y = StandardScaler().fit(Y_raw)
X = scaler_X.transform(X_raw)
Y = scaler_Y.transform(Y_raw)


X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val,   X_test, Y_val,   Y_test   = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

in_dim, out_dim = X_train.shape[1], Y_train.shape[1]

lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

model = models.Sequential([
    layers.Input(shape=(in_dim,)),
    layers.GaussianNoise(0.02),

    layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation("elu"),
    layers.Dropout(0.3),

    layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation("elu"),
    layers.Dropout(0.3),

    layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation("elu"),
    layers.Dropout(0.2),

    layers.Dense(out_dim, activation="linear"),
])


opt = optimizers.Adam(learning_rate=lr_schedule)
model.compile(
    optimizer=opt,
    loss=losses.Huber(delta=1.0),
    metrics=["mae"]
)


history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=1000,
    batch_size=64,
    # callbacks=[lr_plateau, early_stop],
    verbose=2
)


train_loss = history.history["loss"][-1]
val_loss   = history.history["val_loss"][-1]
test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)

print(f"\nFinal train loss: {train_loss:.6f}")
print(f"Final   val loss: {val_loss:.6f}")
print(f"Test  loss/mae: {test_loss:.6f} / {test_mae:.6f}")


model.save("quad_net_improved.keras")


plt.figure(figsize=(8,5))
plt.plot(history.history["loss"],    label="train loss")
plt.plot(history.history["val_loss"], label="val   loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Huber Loss (log)")
plt.legend()
plt.tight_layout()
plt.show()
