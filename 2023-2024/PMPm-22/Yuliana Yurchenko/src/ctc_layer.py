import tensorflow as tf
from tensorflow import keras
from keras import layers


class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(CTCLayer, self).__init__(name=name)
        super(CTCLayer, self).__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def get_config(self):
        config = super(CTCLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
