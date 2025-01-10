"""Class to contain a Keras InceptionResnetV2 model."""

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .base import ModelBase


# Class to hold Inception_ResnetV2
class InceptionResnetV2(ModelBase):
    """InceptionResnetV2 model."""

    def __init__(self, model_config):
        """Initialize the InceptionResnetV2 model with the given configuration."""
        super().__init__(model_config)

        self.name = "Inception_ResnetV2"
        self.model = InceptionResNetV2(
            include_top=False,
            weights="imagenet" if self.use_imagenet_weights else None,
            input_shape=[*self.input_shape, 3],
            input_tensor=None,
            pooling="max",
        )

        self._config["preprocess_func"] = "tensorflow.keras.applications.inception_resnet_v2.preprocess_input"
        self.setup()

    def set_layers(self):
        """Set the bottom layers of the network."""
        x = self.model.output
        x = Dropout(self.dropout)(x)
        x = Dense(len(self.classes), activation="softmax")(x)
        self.model = Model(self.model.input, x)
