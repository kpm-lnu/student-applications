from tensorflow.keras.models import load_model

from .standart_model_training import CTCLayer

load_model('../models/pro_model.h5', custom_objects={'CTCLayer': CTCLayer})
