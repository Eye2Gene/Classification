import tensorflow as tf
import numpy as np

from .base import ModelBase

class ConvNeXt(ModelBase):
    ''' ConvNeXt Model class '''

    def __init__(self, model_config):
        super(ConvNeXt, self).__init__(model_config)
        
        self._config['preprocess_func'] = 'tensorflow.keras.applications.imagenet_utils.preprocess_input'
        
        from keras_cv_attention_models import convnext

        self.model = convnext.ConvNeXtTiny(pretrained='imagenet' if self.use_imagenet_weights else None,
                                           input_shape=self.input_shape + [3,])
        self.name = 'ConvNeXtTiny'

        self.setup()

    def set_layers(self):
        ''' Set the bottom layers of the network '''
        
        dense = self.model.layers[-2].output
        prediction = tf.keras.layers.Dense(len(self.classes),
                                           activation = "softmax",
                                           name='final-output-dense')(dense)
        self.model = tf.keras.Model(self.model.input, prediction)
        
class ConvNeXtV2(ModelBase):
    ''' ConvNeXt Model class '''

    def __init__(self, model_config):
        super(ConvNeXtV2, self).__init__(model_config)
        
        #Not clear if needed
        #self._config['preprocess_func'] = 'tensorflow.keras.applications.imagenet_utils.preprocess_input'
        
        from keras_cv_attention_models import convnext

        self.model = convnext.ConvNeXtV2Tiny(pretrained='imagenet' if self.use_imagenet_weights else None,
                                             input_shape=self.input_shape + [3,])
        self.name = 'ConvNeXtV2Tiny'

        self.setup()

    def set_layers(self):
        ''' Set the bottom layers of the network '''
        
        dense = self.model.layers[-2].output
        prediction = tf.keras.layers.Dense(len(self.classes),
                                           activation = "softmax",
                                           name='final-output-dense')(dense)
        self.model = tf.keras.Model(self.model.input, prediction)
