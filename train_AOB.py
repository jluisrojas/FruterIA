import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import shutil
from os import path
import json

import tensorflow as tf
import cv2
import numpy as np

# Importa datasets
from datasets.Fruits360.f360_dataset import f360_load_dataset
from datasets.mnist.mnist_dataset import load_mnist_dataset_resize, load_mnist_dataset
from datasets.AOBDataset.aob_dataset import load_dataset

# Importa modelos
from models.mobilenetv2 import MobileNetV2
from models.test_models import mnist_model
from models.smallervggnet import SmallerVGGNet
from tensorflow.keras.models import Sequential

# Import funciones para entrenameinto
from training_utils.training import continue_training
from training_utils.training import train_model

def main():
    train_AOB()

def AOB_train_setup():
    setup = {
        "info": """Entrenando AOB dataset sin bolsa con MobileNetV2 con weights de imagnet y RMSprop, 
        input shape de [224, 224, 3], se esta entrenando el modelo completo""",
        "path": "trained_models/AOB_MobileNetV2_04/",
        "dataset_path": "datasets/AOBDataset/AOB_TF_NB/",
        "num_classes": 3,
        "classes": [],
        "input_shape": (224, 224, 3),
        "epochs": 20,
        "batch_size": 20,
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "learning_rate": 0.0001,
        "seed": 123321,
        "dataset_info": " "
    }

    return setup

def train_AOB():
    setup = AOB_train_setup()
    tf.random.set_seed(setup["seed"])
    np.random.seed(setup["seed"])

    w, h, _ = setup["input_shape"]

    train, test, info = load_dataset(path=setup["dataset_path"]) 

    train = train.shuffle(int(info["train_size"] / info["num_classes"])).batch(setup["batch_size"])
    test = test.batch(setup["batch_size"])

    setup["dataset_info"] = info
    setup["classes"] = info["categories"]
    setup["num_classes"] = info["num_classes"]

    #model = SmallerVGGNet.build(input_shape=(100, 100, 3), classes=3)
    #model = tf.keras.applications.MobileNetV2(include_top=True,
    #        weights="imagenet",classes=3, input_shape=(100, 100, 3))

    model = mnv2_transfer_model(num_classes=setup["num_classes"],
            input_shape=setup["input_shape"])
    #model = mnv2_finetune_model(num_classes=3, input_shape=(96, 96, 3))

    train_model(setup, model, (train, test))

def mnv2_transfer_model(num_classes=None, input_shape=None):
    # Obtiene el modelo base que proporciona keras
    # este no incluye el top, porque es custom
    base_model = tf.keras.applications.MobileNetV2(include_top=False,
            weights="imagenet", input_shape=input_shape)
    base_model.trainable = True

    # Agrega un classficador al final del modelo
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name="gap")
    prediction_layer = tf.keras.layers.Dense(num_classes, name="dense")
    activation_layer = tf.keras.layers.Activation("softmax", name="activation")

    # Crea un nuevo modelo con el base_model y clasficador
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer,
        activation_layer])

    return model

def mnv2_finetune_model(num_classes=None, input_shape=None):
    # Obtiene el modelo de MobileNetV2 con transfer learning
    base_model = mnv2_transfer_model(num_classes=num_classes, input_shape=input_shape)
    # El numero de layers que se van a congelar
    fine_tune_at = 50

    # Congela las layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    return base_model

main()
