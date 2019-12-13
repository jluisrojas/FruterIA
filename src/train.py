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
            Training MobileNetV2 with ImageNet weights and RMSprop optimizer.
            first training the last fully connected layer, then using
            fine tunning from the 100th layer. 
            """,
        "path": "trained_models2/MNV2_ft_17/",
        "include_bag": True,
        "color_data": True,
        "color_type": "HIST",
        "dataset_path": "datasets/AOBDataset/",
        "num_classes": 3,
        "classes": [],
        "input_shape": (224, 224, 3),
        "epochs": 30,
        "ft_epochs": 20,
        "batch_size": 50,
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "learning_rates": [
            0.001,
            0.0001 / 10],
        "fine_tune_at": 100,
        "seed": 123321,
        "dataset_info": " "
    }

    if setup["color_data"] == True:
        if setup["color_type"] == "RGB":
            setup["dataset_path"] += "AOB_BAG_COLOR/"
        elif setup["color_type"] == "HIST":
            setup["dataset_path"] += "AOB_BAG_HIST/"
    else:
        if setup["include_bag"] == True:
            setup["dataset_path"] += "AOB_TF/"
        else:
            setup["dataset_path"] += "AOB_TF_NB/"


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
    train, test, info = load_dataset(path=setup["dataset_path"],
            color_data=setup["color_data"], color_type=setup["color_type"])

    # adds ifnormation of the dataset to the training setup
    setup["dataset_info"] = info
    setup["classes"] = info["categories"]
    setup["num_classes"] = info["num_classes"]

    # Checks if there is color data to extract it
    if setup["color_data"] == True:
        def _join_inputs(x, c, y):
            return (x, c), y

        train = train.map(_join_inputs)
        test = test.map(_join_inputs)

    #train = train.map(color_aug)

    train = train.shuffle(int(info["train_size"] / info["num_classes"])).batch(setup["batch_size"])
    test = test.batch(setup["batch_size"])

    return train, test

# Function that creates the multi input model
def multi_input_model(setup):
    input_img = tf.keras.Input(shape=setup["input_shape"])
    if setup["color_type"] == "RGB":
        input_col = tf.keras.Input(shape=(3,))
    elif setup["color_type"] == "HIST":
        input_col = tf.keras.Input(shape=(765,))

    base_model = tf.keras.applications.MobileNetV2(include_top=False,
            alpha=1.0, weights="imagenet", input_shape=setup["input_shape"])
    base_model.trainable = False 

    # Adds classifer head at the end of the model
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name="gap")
    conv_dense = tf.keras.layers.Dense(3, activation="relu", name="conv_dense")

    x = base_model(input_img)
    x = global_average_layer(x)
    #x = conv_dense(x)

    # Numerical data layers
    num_dense1 = tf.keras.layers.Dense(500, activation="relu", name="color_dense1")
    #num_dense2 = tf.keras.layers.Dense(100, activation="relu", name="color_dense2")

    #y = num_dense1(input_col)
    #y = num_dense2(y)

    combined = tf.keras.layers.Concatenate()([x, input_col])

    prediction_layer = tf.keras.layers.Dense(setup["num_classes"], name="dense")
    activation_layer = tf.keras.layers.Activation("softmax", name="activation")

    z = prediction_layer(combined)
    z = activation_layer(z)

    # Creates the new model
    model = tf.keras.Model(inputs=[input_img, input_col], outputs=z)

    # Creates layers dictionaire
    layers_dict = {
        "base_cnn": base_model,
        "global_average": global_average_layer,
        "conv_dense": conv_dense,
        "num_dense1": num_dense1,
        #"num_dense2": num_dense2,
        "concat": combined,
        "prediction": prediction_layer,
        "activation": activation_layer }

    return model, layers_dict

# Function that creates the standard model
def std_model(setup):
    base_model = tf.keras.applications.MobileNetV2(include_top=False,
            alpha=1.0, weights="imagenet", input_shape=setup["input_shape"])
    base_model.trainable = False 

    # Adds classifer head at the end of the model
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name="gap")

    prediction_layer = tf.keras.layers.Dense(setup["num_classes"], name="dense")
    activation_layer = tf.keras.layers.Activation("softmax", name="activation")

    # Creates the new model
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer,
        activation_layer])

    # Creates layer dictionaire
    layers_dict = {
        "base_cnn": base_model,
        "global_average": golbal_average_layer,
        "prediction": prediction_layer,
        "activation": activation_layer }

    return model, layers_dict

# Function that trains the model
def train_model(setup=None, dataset=None):
    tf.random.set_seed(setup["seed"])
    np.random.seed(setup["seed"])

    train, test = dataset

    if setup["color_data"] == True:
        model, l_dict = multi_input_model(setup)
    else:
        model, l_dict = std_model(setup)

    base_model = l_dict["base_cnn"]

    # Plots the model
    plot_model(model, to_file=setup["path"]+"model_architecture.png",
        show_shapes=True, show_layer_names=True, expand_nested=False)

    # Compiles the model
    opt = RMSprop(lr=setup["learning_rates"][0])
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
    # Num of layers in the base model: 155
    fine_tune_at = setup["fine_tune_at"]

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    opt = RMSprop(lr=setup["learning_rates"][1])
    model.compile(loss=setup["loss"], optimizer=opt, metrics=setup["metrics"])

    # Model callbacks
    callbacks = [
        CSVLogger(setup["path"]+"training_log.csv", append=True),
        TensorBoard(log_dir=setup["path"]+"logs")
    ]

    # Trains the model
    print("[INFO] Fine tune phase")
    total_epochs = setup["epochs"] + setup["ft_epochs"]
    _ = model.fit(train, initial_epoch=setup["epochs"], epochs=total_epochs,
            callbacks=callbacks, validation_data=test)

    # Saves model
    print("[INFO] Serializing model")
    model.save(setup["path"] + "model.h5")

    # Graph model metrics
    print("[INFO] Graphing metrics")
    graph_model_metrics(csv_path=setup["path"] + "training_log.csv",
            img_path=setup["path"]+"metrics_graph.png")

main()
