import os
from os import path
import json
import shutil

import tensorflow as tf
import numpy as np

# Importa cosas de Keras API
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# Importa callbacks del modelo
from training_utils.callbacks import TrainingCheckPoints
from tensorflow.keras.callbacks import CSVLogger, TensorBoard

# Importa cosas para graficar el entrenameinto
from training_utils.training_graphs import graph_confusion_matrix
from training_utils.training_graphs import graph_model_metrics

# Function that continues the training of a model
# Args:
#   path_to_model: path were to find the model and setup
#   dataset: tuple of tensorflow dataset of (train, test)
def continue_training(path_to_model, dataset):
    if not path.exists(path_to_model):
        print("[ERROR] El path a la carpeta del modelo no existe")
        return

    # carga el setup del modelo
    setup = None
    with open(path_to_model+"setup.json", "r") as data:
        setup = json.load(data)

    # carga el estado de entrenamiento
    state = None
    with open(path_to_model+"checkpoints/"+"training_state.json", "r") as data:
        state = json.load(data)

    print("[INFO] Continuando entrenameinto de modelo.")

    # carga el modelo
    model_name = "model_checkpoint_{}.h5".format(state["epoch"]-1)
    model = tf.keras.models.load_model(path_to_model+"checkpoints/"+model_name)

    # vuelve a compilar el modelo
    opt = Adam(lr=state["learning_rate"])
    model.compile(loss=setup["loss"], optimizer=opt, metrics=setup["metrics"])

    fit_model(compiled_model=model, dataset=dataset, opt=opt,
            epochs=setup["epochs"], initial_epoch=state["epoch"],
            path=setup["path"], continue_train=True, classes=setup["classes"])

# Method that starts the model training
# Args:
#   setup: Dictionary with the model setup
#   model: the keras.Model architecture to train
#   dataset: tuple of tensorflow dataset of (train, test)
def train_model(setup, model, dataset):
    # Asegura que el path sea el correcto
    if not path.exists(setup["path"]):
        os.makedirs(setup["path"])
    else:
        # Borra las carpetas si ya existen
        if path.exists(setup["path"]+"checkpoints"):
            shutil.rmtree(setup["path"]+"checkpoints")

        if path.exists(setup["path"]+"logs"):
            shutil.rmtree(setup["path"]+"logs")

    # crea carpeta donde se van a guardar los checkpoints
    if not path.exists(setup["path"]+"checkpoints"):
        os.mkdir(setup["path"] + "checkpoints")

    # Escribe el setup del entrenamiento
    with open(setup["path"]+"setup.json", "w") as writer:
        json.dump(setup, writer, indent=4)

    print("[INFO] Entrenando modelo.")

    # Dibuja la arquitectura del modelo
    plot_model(model, to_file=setup["path"]+"model_architecture.png",
            show_shapes=True, show_layer_names=True, expand_nested=False)

    # Crea optimizador, por defecto Adam
    #opt = Adam(lr=setup["learning_rate"])
    opt = RMSprop(lr=setup["learning_rate"])

    # Compila el modelo
    model.compile(loss=setup["loss"], optimizer=opt, metrics=setup["metrics"])

    fit_model(compiled_model=model, dataset=dataset, opt=opt, 
            epochs=setup["epochs"], path=setup["path"], classes=setup["classes"])


# Metodo, que entrena un modelo ya compilado, implementa callbacks de
# tensorboard, log a un archivo CSV y creacion de checkpoints cuando ocurre
# mejoras en el loss, tambien grafica y crea matriz de confusion
# Args:
#   compiled_model: keras.Model ya compilado
#   dataset: tuple of tensorflow dataset of (train, test)
#   opt: keras.Optimizer used in training
#   epochs: The number of epochs to train
#   initial_epoch: Epoch to start training, 0 for normal training
#   continue_train: if the model is continuing training
#   classes: array of classes that the model predict
def fit_model(compiled_model=None, # El modelo debe de estar complicado
        dataset=None, 
        opt=None, 
        epochs=None,
        initial_epoch=0,
        path=None, 
        continue_train=False,
        classes=None):
    # obtiene el dataset
    train, test = dataset

    # Callbacks durante entrenamiento
    relative = 0
    if initial_epoch >= 1:
        relative = initial_epoch
    callbacks = [
        TrainingCheckPoints(path+"checkpoints/", relative_epoch=relative),
        CSVLogger(path+"training_log.csv", append=continue_train),
        TensorBoard(log_dir=path+"logs")
    ]

    # Entrena el modelo
    history = compiled_model.fit(train, initial_epoch=initial_epoch, epochs=epochs, 
            callbacks=callbacks, validation_data=test)

    # Guarda el modelo
    print("[INFO] Serializing model.")
    compiled_model.save(path + "model.h5")

    # Crea grafica del entrenamiento
    graph_model_metrics(csv_path=path+"training_log.csv",
            img_path=path+"metrics_graph.png")

    # Crea confusion matrix
    if test != None:
        print("[INFO] Creando matriz de confusion")
        graph_confusion_matrix(model=compiled_model, test_dataset=test, 
                classes=classes, path=path+"confusion_matrix.png")

def load_model(path):
    model = tf.keras.models.load_model(path + "model.h5")
    with open(path + "setup.json", "r") as data:
        setup = json.load(data)

    return model, setup["classes"]
