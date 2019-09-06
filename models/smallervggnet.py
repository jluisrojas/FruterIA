# Implementacion de arquitectura de modelo, siguiendo tutorial de pyimagesearch

import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import BatchNormalization
from tf.keras.layers import Conv2D
from tf.keras.layers import MaxPool2D
from tf.keras.layers import Activation
from tf.keras.layers import Flatten
from tf.keras.layers import Dropout
from tf.keras.layers import Dense

class SmallerVGGNet:
    @staticmethod
    def build(input_shape=(100, 100, 3), classes=10):
        model = Sequential()

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3) padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FC => RELU
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifer
        model.add(Dense(classes))
        model.add(Activation("relu"))

        return model
