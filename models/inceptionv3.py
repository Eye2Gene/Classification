"""
Class to contain an InceptionV3 Keras model
"""

from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.applications import InceptionV3 as kInceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

from .base import ModelBase

class InceptionV3(ModelBase):
    ''' InceptionV3 Model class '''

    def __init__(self, model_config):
        super(InceptionV3, self).__init__(model_config)
        
        self._config['preprocess_func'] = 'tensorflow.keras.applications.inception_v3.preprocess_input'

        self.name = 'InceptionV3'
        self.model = kInceptionV3(
            include_top=False,
            weights='imagenet' if self.use_imagenet_weights else None,
            input_shape=self.input_shape + [3,],
            input_tensor=None,
            pooling='max',
        )

        self.setup()

    def set_layers(self):
        ''' Set the bottom layers of the network '''

        #for layer in self.model.layers:
        #    layer.trainable = False

        x = self.model.output
        x = Dropout(self.dropout)(x)
        x = Dense(len(self.classes), activation='softmax')(x)
        self.model = Model(self.model.input, x)
