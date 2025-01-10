"""Class to contain a NASNetLarge Keras model."""

from tensorflow.keras.applications import NASNetLarge as kNASNetLarge
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

from .base import ModelBase


class NASNetLarge(ModelBase):
    """NASNetLarge Model class."""

    def __init__(self, model_config):
        """Initialize the NASNetLarge model with the given configuration."""
        super().__init__(model_config)

        self.input_shape = (331, 331)

        self.name = "NASNetLarge"
        self.model = kNASNetLarge(
            include_top=False,
            weights="imagenet" if self.use_imagenet_weights else None,
            input_shape=(*self.input_shape, 3),
            input_tensor=None,
            pooling="max",
        )

        self.setup()

    def set_layers(self):
        """Set the bottom layers of the network."""
        x = self.model.output
        x = Dropout(self.dropout)(x)
        x = Dense(len(self.classes), activation="softmax")(x)
        self.model = Model(self.model.input, x)
