import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import shutil
from os import path
import json

import tensorflow as tf
import cv2
import numpy as np

# Importa cosas de Keras API
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# Importa datasets
from datasets.Fruits360.f360_load_dataset import load_dataset as load_f360_dataset
from datasets.mnist.mnist_dataset import load_mnist_dataset_resize, load_mnist_dataset

# Importa modelos
from models.mobilenetv2 import MobileNetV2
from models.test_models import mnist_model
from models.smallervggnet import SmallerVGGNet

# Importa callbacks del modelo
from training_utils.callbacks import TrainingCheckPoints
from tensorflow.keras.callbacks import CSVLogger, TensorBoard

# Importa cosas para graficar el entrenameinto
from training_utils.training_graphs import graph_confusion_matrix
from training_utils.training_graphs import graph_model_metrics

def main():
    train_f360()
    #train_mnist()

def f360_train_setup():
    setup = {
        "info": "Entrenando Fruits 360 dataset con MobileNetV2 y RMSprop, los weights inicializados con los de imagenet e input shape de [96, 96, 3], con toda la red entrenable",
        "path": "trained_models/f360_MobileNetV2_04/",
        "num_classes": 3,
        "classes": ["Apple Golden 1", "Banana", "Orange"],
        "input_shape": (96, 96, 3),
        "epochs": 15,
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


# el modelo debe de estar compilado
def fit_model(compiled_model=None, 
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
        #TrainingCheckPoints(path+"checkpoints/", relative_epoch=relative),
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

def train_f360():
    setup = f360_train_setup()
    tf.random.set_seed(setup["seed"])
    np.random.seed(setup["seed"])

    train, test = load_f360_dataset(path="datasets/Fruits360/", resize=96)

    train = train.shuffle(492).batch(setup["batch_size"])
    test = test.batch(setup["batch_size"])

    #model = SmallerVGGNet.build(input_shape=(100, 100, 3), classes=3)
    #model = tf.keras.applications.MobileNetV2(include_top=True,
    #        weights="imagenet",classes=3, input_shape=(100, 100, 3))

    model = mnv2_transfer_model(num_classes=3, input_shape=(96, 96, 3))

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
