import tensorflow as tf
import cv2
from glob import glob
import sys
import os

training_path = "training/"
test_path = "test/"
num_imgs = -1 # -1 es para todas

process_cats = ["Apple Golden 1", "Banana", "Orange"]
onehot_depth = len(process_cats)
onehot_dict = { }
for i in range(len(process_cats)):
    cat = process_cats[i]
    onehot_dict[cat] = i

def main():
    # Obtiene todas las categorias que existen
    cats = [x[1] for x in os.walk("training/.")][0]

    # Writer al tfrecord
    train_writer = tf.io.TFRecordWriter("f360_train.tfrecord")
    test_writer = tf.io.TFRecordWriter("f360_test.tfrecord")

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

            # escribe test images
            for j in range(n_test):
                img_path = test_img_path[j]
                image = cv2.imread(img_path)
                encode_image_info(image, cat, test_writer)

    train_writer.close()
    test_writer.close()

# Returns a bytes_list from a string / byte
def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

main()
