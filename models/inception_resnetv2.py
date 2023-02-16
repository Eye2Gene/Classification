"""
Class to contain a Keras InceptionResnetV2 model
"""
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model as kModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from .base import Model

# Class to hold Inception_ResnetV2
class InceptionResnetV2(Model):
    ''' InceptionResnetV2 model '''

    def __init__(self, model_config):
        super(InceptionResnetV2, self).__init__(model_config)

        self.name = 'Inception_ResnetV2'
        self.model = InceptionResNetV2(
            include_top=False,
            weights='imagenet' if self.use_imagenet_weights else None,
            input_shape=self.input_shape + [3,],
            input_tensor=None,
            pooling='max',
        )

        self._config['preprocess_func'] = 'tensorflow.keras.applications.inception_resnet_v2.preprocess_input'
        self.setup()

    def set_layers(self):
        ''' Set the bottom layers of the network '''

        #for layer in self.model.layers:
        #    layer.trainable = False

        x = self.model.output
        x = Dropout(self.dropout)(x)
        x = Dense(len(self.classes), activation='softmax')(x)
        self.model = kModel(self.model.input, x)