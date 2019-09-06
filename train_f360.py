import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import shutil
from os import path
import json

import tensorflow as tf
import cv2
import numpy as np

# Importa cosas de Keras API
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# Importa datasets
from datasets.Fruits360.f360_load_dataset import load_dataset
from datasets.mnist.mnist_dataset import load_mnist_dataset_resize, load_mnist_dataset

# Importa modelos
from models.mobilenetv2 import MobileNetV2
from models.test_models import mnist_model

# Importa cosas para graficar el entrenameinto
from training_utils.training_graphs import graph_confusion_matrix
from training_utils.training_graphs import graph_model_metrics

def main():
    train_mnist()

def train_setup():
    setup = {
        "info": "Entrenada con una red simple, prueba sin validation dataset",
        "path": "trained_models/mnist_test_00/",
        "num_classes": 20,
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "epochs": 20,
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

    # Dibuja la arquitectura del modelo
    plot_model(model, to_file=setup["path"]+"model_architecture.png",
            show_shapes=True, show_layer_names=True, expand_nested=True)

    # Crea optimizador, por defecto Adam
    opt = Adam(lr=setup["learning_rate"])

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
    #graph_model_metrics(csv_path=path+"training_log.csv",
    #        img_path=path+"metrics_graph.png")

    # Crea confusion matrix
    print("[INFO] Creando matriz de confusion")
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if test != None:
        graph_confusion_matrix(model=compiled_model, test_dataset=test, 
                classes=classes, path=path+"confusion_matrix.png")


class TrainingCheckPoints(tf.keras.callbacks.Callback):
    def __init__(self, folder_path, relative_epoch=0):
        super(TrainingCheckPoints, self).__init__()

        self.folder_path = folder_path
        self.relative_epoch = relative_epoch

    def on_train_begin(self, logs=None):
        self.best_loss = np.Inf
        self.checkpoint_num = self.relative_epoch

    def on_epoch_end(self, epoch, logs=None):
        # Verifica que el modelo tenga learning rate
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Checa si mejoro el loss para hacer un checkpoint
        current_loss = logs.get("loss")
        if current_loss < self.best_loss:
            print("[TRAINING] Creating model checkpoint.")

            self.model.save(self.folder_path+"model_checkpoint_{}.h5".format(self.checkpoint_num))

            if self.checkpoint_num > 0:
                os.remove(self.folder_path+"model_checkpoint_{}.h5".format(self.checkpoint_num-1))

            self.checkpoint_num += 1
            self.best_loss = current_loss
        
        # Guarda el estado actual de entrenamiento, por si se quiere continuar
        training_state = {
            "learning_rate": float(tf.keras.backend.get_value(self.model.optimizer.lr)),
            "epoch": self.checkpoint_num
        }

        with open(self.folder_path+"training_state.json", "w") as writer:
            json.dump(training_state, writer, indent=4)

def train_mnist():
    setup = train_setup()
    tf.random.set_seed(setup["seed"])
    np.random.seed(setup["seed"])

    #train, test = load_mnist_dataset_resize(224)
    train, test = load_mnist_dataset()
    
    train = train.shuffle(456).batch(setup["batch_size"])
    test = test.batch(setup["batch_size"])

    model = mnist_model()
    #model = tf.keras.applications.MobileNetV2(include_top=True,
    #        weights=None,classes=10, input_shape=(224, 224, 1))

    train_model(setup, model, (train, None))
    #continue_training("trained_models/mnist_test/", (train, test))



main()
