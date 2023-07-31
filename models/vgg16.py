"""
Class to contain a Keras VGG16 model
"""

from tensorflow.keras.applications.vgg16 import VGG16 as keras_VGG16
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

from .base import ModelBase


class VGG16(ModelBase):
    ''' VGG16 Model class '''

    def __init__(self, model_config):
        super(VGG16, self).__init__(model_config)

        self.name = 'vgg16'

        self.model = keras_VGG16(
            include_top=False,
            weights='imagenet' if self.use_imagenet_weights else None,
            input_shape=self.input_shape + (3,),
        )

        self.setup()

    def set_layers(self):
        """Set up layers of network for training"""

        # Freeze all layers up to last conv block
        # https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975#gistcomment-2068023
        for layer in self.model.layers[:15]:
            layer.trainable = False

        x = self.model.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(len(self.classes), activation='softmax')(x)

        self.model = Model(inputs=self.model.input, outputs=x)

    def compile(self):
        ''' Compile the model using an optimiser (gradient descent method) '''
        optimizer = SGD(lr=self.lr, momentum=0.9)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
