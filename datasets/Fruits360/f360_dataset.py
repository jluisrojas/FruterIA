import tensorflow as tf
import cv2
from glob import glob
import sys
import os
from os import path
import json

from datasets.datasets_features import bytes_feature

# Metodo que regresa el dataset de f360 ya procesado a tfrecord
# Los data set tiene el formato:
#   x: tensor con la imagen normalizada
#   y: tensor con onehot encoding de la categoria
# Returns:
#   train_data: Dataset de entrenameinto
#   test_data: Dataset de pruebas
def f360_load_dataset(path=None, resize=None, num_classes=None):
    train_path = "f360_train.tfrecord"
    test_path = "f360_test.tfrecord"

    if path == None:
        path = ""

    train_raw_data = tf.data.TFRecordDataset(path+train_path)
    test_raw_data = tf.data.TFRecordDataset(path+test_path)

    _format = {
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_example(example):
        ex = tf.io.parse_single_example(example, _format)
        x = tf.io.parse_tensor(ex["x"], tf.float32)
        y = tf.io.parse_tensor(ex["y"], tf.float32)
        y = tf.reshape(y, [-1])

        data_dict = {
            "x": x,
            "y": y
        }

        return x, y

    train_data = train_raw_data.map(_parse_example)
    test_data = test_raw_data.map(_parse_example)

    def _set_dataset_shape(x, y):
        x.set_shape([100, 100, 3])

        return x, y

    train_data = train_data.map(_set_dataset_shape)
    test_data = test_data.map(_set_dataset_shape)

    if resize != None:
        def _resize_dataset(x, y):
            x = tf.image.resize(x, [resize, resize])

            return x, y

        train_data = train_data.map(_resize_dataset)
        test_data = test_data.map(_resize_dataset)

    with open(path+"dataset_info.json", "r") as data:
        info = json.load(data)


    return train_data, test_data, info

# Metodo que convierte el dataset de Fruits 360 a tfrecord, para despues usarlo
# con el Dataset API de tensorflow
# Args:
#   training_path: el path al dataset de training
#   test_path: el path al dataset de pruebas
#   num_imgs: numero de images a obtener, -1 para todas
#   result_path: el path donde se guarda el resultado
def f360_create_dataset(training_path=None, test_path=None, num_imgs=-1,
        result_path=None):
    # Crea la carpeta por si no existe donde se va a guardar el resultado
    if not path.exists(result_path):
        os.makedirs(result_path)

    #process_cats = ["Apple Golden 1", "Banana", "Orange"]
    process_cats = ["Apple Braeburn", "Apple Golden 1", "Avocado", "Lemon",
        "Limes", "Lychee", "Mandarine", "Banana", "Onion White", "Onion White",
        "Pear", "Orange", "Pineapple", "Potato White", "Strawberry", "Tomato 4"]

    onehot_depth = len(process_cats)
    onehot_dict = { }
    for i in range(len(process_cats)):
        cat = process_cats[i]
        onehot_dict[cat] = i

    # Obtiene todas las categorias que existen
    cats = [x[1] for x in os.walk(training_path)][0]

    # Writer al tfrecord
    train_writer = tf.io.TFRecordWriter(result_path+"f360_train.tfrecord")
    test_writer = tf.io.TFRecordWriter(result_path+"f360_test.tfrecord")

    train_size = 0
    test_size = 0

    # funcion que escribe una imagen al tfrecord
    def encode_image_info(image, category, writer):
        # Convierte la imagen a un tensor y lo normaliza 
        image_tensor = tf.convert_to_tensor(image)
        image_tensor /= 255

        category = tf.one_hot([onehot_dict[category]], onehot_depth)

        # Genera los features para el example
        data = {
            "x": bytes_feature(tf.io.serialize_tensor(image_tensor)),
            "y": bytes_feature(tf.io.serialize_tensor(category))
        }

        example = tf.train.Example(features=tf.train.Features(feature=data))
        writer.write(example.SerializeToString())

    print("[INFO] Writing dataset to tfrecord")
    # itera sobre todas las categorias a procesar
    for cat in process_cats:
        # si la categoria existe
        if cat in cats:
            print("[INFO] Writing {}...".format(cat))
            # obtiene los paths
            train_img_path = glob(training_path+cat+"/*.jpg")
            test_img_path = glob(test_path+cat+"/*.jpg")

            # el numero de imagenes a que se van a ciclar
            n_train = n_test = num_imgs
            if n_train == -1:
                n_train = len(train_img_path)
                n_test = len(test_img_path)

            # escribe training images
            for i in range(n_train):
                img_path = train_img_path[i]
                image = cv2.imread(img_path)
                encode_image_info(image, cat, train_writer)
                train_size += 1

            # escribe test images
            for j in range(n_test):
                img_path = test_img_path[j]
                image = cv2.imread(img_path)
                encode_image_info(image, cat, test_writer)
                test_size += 1

    train_writer.close()
    test_writer.close()

    dataset_info = {
        "name": "Fruits 360 dataset",
        "num_classes": len(process_cats),
        "categories": process_cats,
        "train_size": train_size,
        "test_size": test_size
    }

    # Escribe el info del dataset
    with open(result_path+"dataset_info.json", "w") as writer:
        json.dump(dataset_info, writer, indent=4)

