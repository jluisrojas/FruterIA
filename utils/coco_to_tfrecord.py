#
# Script que convierte una subparte del dataset de COCO a un archivo
# TFRecord para obtener mejor desempeño a la hora de entrenamiento.
# El formato del archivo es el siguiente:
#   image_string: cadena conteniendo la imagen
#   []:
#       category: string
#       x: int
#       y: int
#       h: int
#       w: int
import os
import glob
import json
import cv2
import tensorflow as tf
import argparse
from datasets_features import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True, 
                    help="path to dataset folder")
    args = vars(ap.parse_args())
    path = args["folder"]

    coco_tfrecord(path)

def coco_tfrecord(path):
    img_dirs = glob.glob(path + "/*.jpg")
    anns_dirs = glob.glob(path +  "/*.json")
    num_imgs = len(img_dirs)

    if num_imgs > 0:
        for i in range(3):
            _image_example(img_path=img_dirs[i], ann_path=anns_dirs[i]) 

    return None

# Creates a tf.Example from a image with annotation
def _image_example(img_path=None, ann_path=None):
    image = cv2.imread(img_path)
    image_str = cv2.imencode(".jpg", image)[1].tostring()
    with open(ann_path) as json_text:
        ann = json.loads(json_text.read())

    data = {
        "img/filename": bytes_feature([str.encode(ann["filename"])]),
        "img/width": int64_feature([ann["width"]]),
        "img/height": int64_feature([ann["height"]]),
        #"img/str": bytes_feature([image_str]),
        "img/bboxes/category": bytes_feature([str.encode(bbox["category_id"]) for bbox
            in ann["bboxes"]])
    }
    print(data)
    return tf.train.Example(features=tf.train.Features(feature=data))



main()
