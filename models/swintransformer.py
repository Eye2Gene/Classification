import tensorflow as tf
from keras_cv_attention_models import swin_transformer_v2

from .base import ModelBase


class SwinTransformerV2(ModelBase):
    """SwinTransformer Model class."""

    def __init__(self, model_config):
        super().__init__(model_config)

        # self._config['preprocess_func'] = 'tensorflow.keras.applications.imagenet_utils.preprocess_input'

        self.model = swin_transformer_v2.SwinTransformerV2Tiny_window16(
            pretrained="imagenet" if self.use_imagenet_weights else None,
            input_shape=[*self.input_shape, 3],
        )
        self.name = "SwinTransformerV2Tiny_window16"

        self.setup()

    def set_layers(self):
        """Set the bottom layers of the network."""
        dense = self.model.layers[-2].output
        prediction = tf.keras.layers.Dense(len(self.classes), activation="softmax", name="final-output-dense")(dense)
        self.model = tf.keras.Model(self.model.input, prediction)
