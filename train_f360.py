import os
import shutil
from os import path
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Importa cosas de Keras API
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, TensorBoard

# Importa datasets
from datasets.Fruits360.f360_load_dataset import load_dataset
from datasets.mnist.mnist_dataset import load_mnist_dataset_resize, load_mnist_dataset

# Importa modelos
from models.mobilenetv2 import MobileNetV2
from models.test_models import mnist_model

def main():
    train_mnist()

def train_setup():
    setup = {
        "path": "trained_models/mnist_test/",
        "num_classes": 5,
        "epochs": 10,
        "batch_size": 128,
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "learning_rate": 0.001
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
            path=setup["path"], continue_train=True)

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

    # Crea optimizador, por defecto Adam
    opt = Adam(lr=setup["learning_rate"])

    # Compila el modelo
    model.compile(loss=setup["loss"], optimizer=opt, metrics=setup["metrics"])

    fit_model(compiled_model=model, dataset=dataset, opt=opt, 
            epochs=setup["epochs"], path=setup["path"])


# el modelo debe de estar compilado
def fit_model(compiled_model=None, 
        dataset=None, 
        opt=None, 
        epochs=None,
        initial_epoch=0,
        path=None, 
        continue_train=False):
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
    graph_model_metrics(history, epochs, path + "grafica.jpg")

    # Crea confusion matrix


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
            "epoch": epoch
        }

        with open(self.folder_path+"training_state.json", "w") as writer:
            json.dump(training_state, writer, indent=4)

def train_mnist():
    setup = train_setup()

    #train, test = load_mnist_dataset_resize(224)
    train, test = load_mnist_dataset()
    
    train = train.shuffle(456).batch(setup["batch_size"])
    test = test.batch(setup["batch_size"])

    model = mnist_model()
    #model = tf.keras.applications.MobileNetV2(include_top=True,
    #        weights=None,classes=10, input_shape=(224, 224, 1))

    #train_model(setup, model, (train, test))
    continue_training("trained_models/mnist_test/", (train, test))



def graph_model_metrics(H, num_epochs, path):

    # plot the training loss and accuracy
    N = num_epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(path)

"""
def train_model_f360():
    print("[INFO] Loading dataset")
    train, test = load_dataset("datasets/Fruits360/")

    # Cambia el tamano de la datset
    def _resize_dataset(x, y):
        x.set_shape([100, 100, 3]) # Estas especificando que realmente el shape
        y.set_shape([1, num_classes])
        y = tf.reshape(y, [-1])
        x = tf.image.resize(x, [224, 224])

        return x, y

    train = train.map(_resize_dataset)
    test = test.map(_resize_dataset)

    train = train.shuffle(buffer_size=492).batch(batch_size)
    test = test.batch(batch_size)

    print("[INFO] Compiling model")
    
    #model = MobileNetV2.build_model(num_classes)
    model = tf.keras.applications.MobileNetV2(weights=None, include_top=True, classes=3)

    
    opt = Adam(lr=3e-4, decay=1e-4 / epochs)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])

    history = model.fit(train, epochs=epochs, validation_data=test)
    graph_model_metrics(history)

    print("[INFO] Serializing network...")
    model.save("f360.model")
"""

main()
