import tensorflow as tf
import numpy as np

from .base import ModelBase

class CoAtNET(ModelBase):
    ''' ConvNeXt Model class '''

    def __init__(self, model_config):
        super(CoAtNET, self).__init__(model_config)
        
        self._config['preprocess_func'] = 'tensorflow.keras.applications.imagenet_utils.preprocess_input'
        
        from keras_cv_attention_models import coatnet

        self.model = coatnet.CoAtNet0(pretrained='imagenet' if self.use_imagenet_weights else None,
                                      input_shape=self.input_shape + [3,])
        self.name = 'CoAtNet0'

        self.setup()

    def set_layers(self):
        ''' Set the bottom layers of the network '''
        
        x = self.model.layers[-2].output
        x = tf.keras.layers.Dropout(self.dropout)(x)
        prediction = tf.keras.layers.Dense(len(self.classes),
                                           activation="softmax",
                                           name='final-output-dense')(x)
        self.model = tf.keras.Model(self.model.input, prediction)
