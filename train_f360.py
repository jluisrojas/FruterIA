import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import shutil
from os import path
import json

import tensorflow as tf
import cv2
import numpy as np

# Importa datasets
from datasets.Fruits360.f360_load_dataset import load_dataset as load_f360_dataset
from datasets.mnist.mnist_dataset import load_mnist_dataset_resize, load_mnist_dataset

# Importa modelos
from models.mobilenetv2 import MobileNetV2
from models.test_models import mnist_model
from models.smallervggnet import SmallerVGGNet
from tensorflow.keras.models import Sequential

# Import funciones para entrenameinto
from training_utils.training import continue_training
from training_utils.training import train_model

def main():
    train_f360()
    #train_mnist()

def f360_train_setup():
    setup = {
        "info": "Entrenando Fruits 360 dataset con SmallerVGGNet y RMSprop, input shape de [100, 100, 3]",
        "path": "trained_models/f360_VGG_02/",
        "num_classes": 3,
        "classes": ["Apple Golden 1", "Banana", "Orange"],
        "input_shape": (100, 100, 3),
        "epochs": 20,
        "batch_size": 16,
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "learning_rate": 0.0001,
        "seed": 123321
    }

    return setup

def train_setup():
    setup = {
        "info": "Entrenando con MobilenetV2 y weights en None",
        "path": "trained_models/mnist_test_02/",
        "num_classes": 10,
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "epochs": 50,
        "batch_size": 128,
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "learning_rate": 0.001,
        "seed": 123321
    }

    return setup

def train_f360():
    setup = f360_train_setup()
    tf.random.set_seed(setup["seed"])
    np.random.seed(setup["seed"])

    w, h, _ = setup["input_shape"]

    train, test = load_f360_dataset(path="datasets/Fruits360/", resize=w)

    train = train.shuffle(492).batch(setup["batch_size"])
    test = test.batch(setup["batch_size"])

    model = SmallerVGGNet.build(input_shape=(100, 100, 3), classes=3)
    #model = tf.keras.applications.MobileNetV2(include_top=True,
    #        weights="imagenet",classes=3, input_shape=(100, 100, 3))

    #model = mnv2_transfer_model(num_classes=3, input_shape=(96, 96, 3))

    train_model(setup, model, (train, test))

def mnv2_transfer_model(num_classes=None, input_shape=None):
    base_model = tf.keras.applications.MobileNetV2(include_top=False,
            weights="imagenet", input_shape=input_shape)
    #base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name="gap")
    prediction_layer = tf.keras.layers.Dense(num_classes, name="dense")
    activation_layer = tf.keras.layers.Activation("softmax", name="activation")

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer,
        activation_layer])

    return model



def train_mnist():
    setup = train_setup()
    tf.random.set_seed(setup["seed"])
    np.random.seed(setup["seed"])

    #train, test = load_mnist_dataset_resize(224)
    train, test = load_mnist_dataset()

    def _resize_images(x, y):
        x = tf.image.resize(x, [224, 224])
        return x, y

    train = train.map(_resize_images)
    test = test.map(_resize_images)

    train = train.shuffle(456).take(10000).batch(setup["batch_size"])
    test = test.shuffle(500).take(1000).batch(setup["batch_size"])

    #model = mnist_model()
    model = tf.keras.applications.MobileNetV2(include_top=True,
            weights=None,classes=10, input_shape=(224, 224, 1))

    #train_model(setup, model, (train, test))
    continue_training("trained_models/mnist_test_02/", (train, test))



main()
