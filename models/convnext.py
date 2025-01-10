"""ConvNeXt Model class."""

import tensorflow as tf
from keras_cv_attention_models import convnext

from .base import ModelBase


class ConvNeXt(ModelBase):
    """ConvNeXt Model class."""

    def __init__(self, model_config):
        """Initialize the ConvNeXt model with the given configuration."""
        super().__init__(model_config)

        self._config["preprocess_func"] = "tensorflow.keras.applications.imagenet_utils.preprocess_input"

        self.model = convnext.ConvNeXtTiny(
            pretrained="imagenet" if self.use_imagenet_weights else None,
            input_shape=[*self.input_shape, 3],
        )
        self.name = "ConvNeXtTiny"

        self.setup()

    def set_layers(self):
        """Set the bottom layers of the network."""
        dense = self.model.layers[-2].output
        prediction = tf.keras.layers.Dense(len(self.classes), activation="softmax", name="final-output-dense")(dense)
        self.model = tf.keras.Model(self.model.input, prediction)


class ConvNeXtV2(ModelBase):
    """ConvNeXt Model class."""

    def __init__(self, model_config):
        """Initialize the ConvNeXt model with the given configuration."""
        super().__init__(model_config)

        # Not clear if needed
        # self._config['preprocess_func'] = 'tensorflow.keras.applications.imagenet_utils.preprocess_input'

        self.model = convnext.ConvNeXtV2Tiny(
            pretrained="imagenet" if self.use_imagenet_weights else None,
            input_shape=[*self.input_shape, 3],
        )
        self.name = "ConvNeXtV2Tiny"

        self.setup()

    def set_layers(self):
        """Set the bottom layers of the network."""
        dense = self.model.layers[-2].output
        prediction = tf.keras.layers.Dense(len(self.classes), activation="softmax", name="final-output-dense")(dense)
        self.model = tf.keras.Model(self.model.input, prediction)
