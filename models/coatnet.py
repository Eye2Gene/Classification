"""This module contains the CoAtNET model implementation."""

import tensorflow as tf
from keras_cv_attention_models import coatnet

from .base import ModelBase


class CoAtNET(ModelBase):
    """ConvNeXt Model class."""

    def __init__(self, model_config: dict) -> None:
        """Initialize the CoAtNET model with the given configuration."""
        super().__init__(model_config)

        self._config["preprocess_func"] = "tensorflow.keras.applications.imagenet_utils.preprocess_input"

        self.use_imagenet_weights = model_config.get("use_imagenet_weights", False)
        self.input_shape = model_config.get("input_shape", [224, 224])

        self.model = coatnet.CoAtNet0(
            pretrained="imagenet" if self.use_imagenet_weights else None,
            input_shape=[*self.input_shape, 3],
        )
        self.name = "CoAtNet0"

        self.setup()

    def set_layers(self) -> None:
        """Set the bottom layers of the network."""
        x = self.model.layers[-2].output
        x = tf.keras.layers.Dropout(self.dropout)(x)
        prediction = tf.keras.layers.Dense(len(self.classes), activation="softmax", name="final-output-dense")(x)
        self.model = tf.keras.Model(self.model.input, prediction)
