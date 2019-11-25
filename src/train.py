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
from datasets.data_aug import *

# Importa modelos
from models.mobilenetv2 import MobileNetV2
from models.test_models import mnist_model
from models.smallervggnet import SmallerVGGNet
from tensorflow.keras.models import Sequential

# Import funciones para entrenameinto
from training_utils.training import continue_training
from training_utils.training import train_model

def main():
    setup = train_setup()
    train, test = dataset_pipeline(setup)
    save_setup(setup)

def train_setup():
    setup = {
        "info": """Entrenando AOB dataset con bolsa, Entrenado en SmallerVGGNet con weights 
        inicializado aleatoriamente y Adam optimzier, input shape de [224, 224, 3]""",
        "path": "trained_models_FINAL/AOB_SVGG_02/",
        "dataset_path": "datasets/AOBDataset/AOB_TF_NB/",
        "num_classes": 3,
        "classes": [],
        "input_shape": (224, 224, 3),
        "epochs": 50,
        "batch_size": 30,
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "learning_rate": 3e-4,
        "seed": 123321,
        "dataset_info": " "
    }

    return setup

# Creates the enviroment for training
# Args:
#   setup: dictionary with the training setup of the model
def save_setup(setup):
    # Creates the training directory
    if not path.exists(setup["path"]):
        os.makedirs(setup["path"])

    # Saves the setup in JSON file
    with open(setup["path"]+"setup.json", "w") as writer:
        json.dump(setup, writer, indent=4)

# Function for the dataset pipeline
# Args:
#   setup: dictionary with the training setup of the model
def dataset_pipeline(setup):
    # loads the dataset from AOB
    train, test, info = load_dataset(path=setup["dataset_path"])

    # adds ifnormation of the dataset to the training setup
    setup["dataset_info"] = info
    setup["classes"] = info["categories"]
    setup["num_classes"] = info["num_classes"]

    # DATASER PIPELINE
    def _join_inputs(x, c, y):
        return (x, c), y

    #train = train.map(_join_inputs)
    #test = test.map(_join_inputs)

    #train = train.map(color_aug)

    train = train.shuffle(int(info["train_size"] / info["num_classes"])).batch(setup["batch_size"])
    test = test.batch(setup["batch_size"])

    return train, test

def train_AOB(setup=None):
    tf.random.set_seed(setup["seed"])
    np.random.seed(setup["seed"])

    w, h, _ = setup["input_shape"]

    train, test, info = load_dataset(path=setup["dataset_path"]) #, color_data=True) 

    def _join_inputs(x, c, y):
        return (x, c), y

    #train = train.map(_join_inputs)
    #test = test.map(_join_inputs)

    #train = train.map(color_aug)

    train = train.shuffle(int(info["train_size"] / info["num_classes"])).batch(setup["batch_size"])
    test = test.batch(setup["batch_size"])

    setup["dataset_info"] = info
    setup["classes"] = info["categories"]
    setup["num_classes"] = info["num_classes"]

    base_model = tf.keras.applications.MobileNetV2(include_top=False,
            alpha=1.0, weights="imagenet", input_shape=input_shape)
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

    train_model(setup, model, (train, test))

main()
