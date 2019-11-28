#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import os
import shutil
from os import path
import json

import tensorflow as tf
import numpy as np
import cv2

# Importa datasets
from datasets.AOBDataset.aob_dataset import load_dataset
from datasets.data_aug import *

# Import model stuff
from models.mobilenetv2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger, TensorBoard

# Import training utils
from training_utils.training_graphs import graph_confusion_matrix
from training_utils.training_graphs import graph_model_metrics

def main():
    setup = train_setup()
    train, test = dataset_pipeline(setup)
    save_setup(setup)
    train_model(setup=setup, dataset=(train, test))

def train_setup():
    setup = {
        "info": """
            Training MobileNetV2 with ImageNet weights and Adam optimizer.
            first training the last fully connected layer, then using
            fine tunning with the last 100 layers
            """,
        "path": "trained_models_FINAL/AOB_MobileNetV2_finetune_01/",
        "dataset_path": "datasets/AOBDataset/AOB_TF/",
        "num_classes": 3,
        "classes": [],
        "input_shape": (224, 224, 3),
        "epochs": 20,
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
    else:
        # Erase the logs dir
        if path.exists(setup["path"]+"logs"):
            shutil.rmtree(setup["path"]+"logs")

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

def train_model(setup=None, dataset=None):
    tf.random.set_seed(setup["seed"])
    np.random.seed(setup["seed"])

    train, test = dataset

    base_model = tf.keras.applications.MobileNetV2(include_top=False,
            alpha=1.0, weights="imagenet", input_shape=setup["input_shape"])
    base_model.trainable = False

    # Agrega un classficador al final del modelo
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name="gap")
    prediction_layer = tf.keras.layers.Dense(setup["num_classes"], name="dense")
    activation_layer = tf.keras.layers.Activation("softmax", name="activation")

    # Crea un nuevo modelo con el base_model y clasficador
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer,
        activation_layer])

    # Plots the model
    plot_model(model, to_file=setup["path"]+"model_architecture.png",
        show_shapes=True, show_layer_names=True, expand_nested=False)

    # Compiles the model
    opt = Adam(lr=setup["learning_rate"])
    model.compile(loss=setup["loss"], optimizer=opt, metrics=setup["metrics"])

    # Model callbacks
    callbacks = [
        CSVLogger(setup["path"]+"training_log.csv", append=False),
        TensorBoard(log_dir=setup["path"]+"logs")
    ]

    # Trains the model
    print("[INFO] Training model")
    _ = model.fit(train, epochs=setup["epochs"], callbacks=callbacks,
            validation_data=test)

    # Fine tunning the mode
    base_model.trainable = True
    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    opt = Adam(lr=setup["learning_rate"]/10)
    model.compile(loss=setup["loss"], optimizer=opt, metrics=setup["metrics"])

    # Model callbacks
    callbacks = [
        CSVLogger(setup["path"]+"training_log.csv", append=True),
        TensorBoard(log_dir=setup["path"]+"logs")
    ]

    # Trains the model
    fine_tune_epochs = 10
    total_epochs = setup["epochs"] + fine_tune_epochs
    _ = model.fit(train, initial_epoch=setup["epochs"], epochs=total_epochs,  
            callbacks=callbacks, validation_data=test)

main()
