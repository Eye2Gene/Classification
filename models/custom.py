
"""
Class to contain a custom Keras model
"""

from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras import Sequential

from .base import Model

class Custom(Model):
    ''' Custom Model class '''

    def __init__(self, model_config):
        super(Custom, self).__init__(model_config)

        self.name = 'Custom'
        self.setup()

    def set_layers(self):
        ''' Create a small conv model '''
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(len(self.classes), activation='softmax'))
