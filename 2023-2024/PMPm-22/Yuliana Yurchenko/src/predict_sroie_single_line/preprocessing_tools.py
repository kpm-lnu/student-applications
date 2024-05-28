import numpy as np
import cv2 as cv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Desired image dimensions
IMG_WIDTH = 256
IMG_HEIGHT = 32


characters = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
              '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'l',
              'r', '{', '|', '}', '~', '·']


# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def encode_single_image(image):
    # 1. Convert grayscale image to 3-dimensional tensor
    image = tf.reshape(image, [image.shape[0], image.shape[1], 1])
    # 2. Convert to float32 in [0, 1] range
    image = tf.image.convert_image_dtype(image, tf.float32)
    # 3. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    image = tf.transpose(image, perm=[1, 0, 2])
    return image


def decode_batch_predictions(pred, max_label_len):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_label_len]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def preprocess_image(image: np.array) -> np.array:
    # 1. Resize to suit the model
    image = cv.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # 2. Some more preprocessing
    # (reshaping, normalizing, transposing (time dimension must correspond to the width of the image))
    image = encode_single_image(image)

    image = tf.reshape(image, [1, IMG_WIDTH, IMG_HEIGHT, 1])
    return image


def predict_text(image: np.array, model) -> str:
    predictions = model.predict(image, verbose=0)
    text_prediction = decode_batch_predictions(predictions, max_label_len=32)[0]

    # Remove unidentified characters
    while text_prediction.find('[UNK]') != -1:
        text_prediction = text_prediction.replace('[UNK]', '')

    # Remove extra spaces from the end
    while len(text_prediction) > 0 and text_prediction[-1] == ' ':
        text_prediction = text_prediction[:-1]

    return text_prediction
