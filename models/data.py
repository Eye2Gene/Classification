""" Data helper functions """
import os
import random
import numpy as np
import PIL

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

def load_data(directory, classes, rescale=True, preprocess=None, verbose=False):
    """ Helper function to load data in a Keras friendly format """

    if not os.path.exists(directory):
        raise FileNotFoundError(directory + ' not found')

    # Check directories
    directories = os.listdir(directory)
    dataset = list()

    for d in directories:
        if d not in classes or not os.path.isdir(os.path.join(directory, d)):
            print('Skipping', d)
            continue
        if verbose:
            print('Loading directory', d)
        for f in os.listdir(os.path.join(directory, d)):

            try:
                # Load image
                img = load_img(
                    os.path.join(os.path.join(directory, d), f),
                    color_mode='rgb',
                    target_size=[256, 256]
                )
            except PIL.UnidentifiedImageError:
                continue

            # Convert to numpy array
            img = img_to_array(img)

            # Apply any preprocess function and rescaling
            if preprocess is not None:
                img = preprocess(img)
            else:
                if rescale:
                    img /= 255

            # Get index number
            class_num = classes.index(d)

            # Append
            dataset.append([img, class_num])

    # Shuffle dataset
    random.shuffle(dataset)

    x = list()
    y = list()

    # Create our x and y arrays
    for img, label in dataset:
        x.append(img)
        y.append(label)

    # Reshape x,y into the required formats
    x = np.array(x).reshape(-1, 256, 256, 3)
    y = to_categorical(y)

    return x, y, 0
