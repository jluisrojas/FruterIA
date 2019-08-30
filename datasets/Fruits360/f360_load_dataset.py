import tensorflow as tf
import cv2

train_path = "f360_train.tfrecord"
test_path = "f360_test.tfrecord"

# Metodo que regresa el dataset de f360 ya procesado a tfrecord
# Los data set tiene el formato:
#   x: tensor con la imagen normalizada
#   y: tensor con onehot encoding de la categoria
# Returns:
#   train_data: Dataset de entrenameinto
#   test_data: Dataset de pruebas
def load_dataset(path=None):
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

        data_dict = {
            "x": x,
            "y": y
        }

        return x, y

    train_data = train_raw_data.map(_parse_example)
    test_data = test_raw_data.map(_parse_example)

    return train_data, test_data
