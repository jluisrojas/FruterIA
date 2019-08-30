import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
import cv2

# Importa cosas de Keras API
from tensorflow.keras.optimizers import Adam

# Importa dataset
from datasets.Fruits360.f360_load_dataset import load_dataset

# Importa modelo
from mobilenetv2 import MobileNetV2

num_classes = 3 # Numero de categorias, manzana-naranja-platano
epochs = 50
batch_size = 3

def main():
    print("[INFO] Loading dataset")
    train, test = load_dataset("datasets/Fruits360/")

    # Cambia el tamano de la datset
    def _resize_dataset(x, y):
        x.set_shape([100, 100, 3]) # Estas especificando que realmente el shape
        x = tf.image.resize(x, [224, 224])

        return x, y

    train = train.map(_resize_dataset)
    test = test.map(_resize_dataset)

    train = train.shuffle(100).batch(batch_size)
    test = test.batch(batch_size)

    print("[INFO] Compiling model")
    
    #model = MobileNetV2.build_model(3)
    model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
            weights='imagenet', include_top=False)
    
    opt = Adam(lr=3e-4)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])

    model.fit(train, epochs=epochs, validation_data=test)


main()
