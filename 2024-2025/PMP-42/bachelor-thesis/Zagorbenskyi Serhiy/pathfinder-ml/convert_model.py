import tensorflowjs as tfjs
from tensorflow import keras

model = keras.models.load_model("model/model.h5")
tfjs.converters.save_keras_model(model, "export/model_tfjs")
