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
    ap.add_argument("-r", "--result", required=True,
                    help="path were the TFRecord will be saved")
    args = vars(ap.parse_args())
    path = args["folder"]
    res = args["result"]

    coco_tfrecord(path, res)

def coco_tfrecord(path, res):
    print("[INFO] Loading images paths")
    img_dirs = glob.glob(path + "/*.jpg")
    num_imgs = len(img_dirs)

    if num_imgs > 0:
        print("[INFO] Serializing dataset")
        with tf.io.TFRecordWriter(res) as f:
            for i in range(num_imgs):
                print("Copied {} of {}".format(i+1, num_imgs))
                example = _image_example(img_path=img_dirs[i], ann_path=img_dirs[i]+".json") 
                f.write(example.SerializeToString())

    return None

# Creates a tf.Example from a image with annotation
def _image_example(img_path=None, ann_path=None):
    image = cv2.imread(img_path)
    image_str = cv2.imencode(".jpg", image)[1].tostring()
    with open(ann_path) as json_text:
        ann = json.loads(json_text.read())

    cat_s = []
    x_s = []
    y_s = []
    w_s = []
    h_s = []
    for bbox in ann["bboxes"]:
        cat_s.append(str.encode(bbox["category_id"]))
        x_s.append(bbox["center_x"])
        y_s.append(bbox["center_y"])
        w_s.append(bbox["width"])
        h_s.append(bbox["height"])

    data = {
        "img/filename": bytes_feature(str.encode(ann["filename"])),
        "img/width": int64_feature(ann["width"]),
        "img/height": int64_feature(ann["height"]),
        "img/str": bytes_feature(image_str),
        "img/bboxes/category": bytes_list_feature(cat_s),
        "img/bboxes/x": float_list_feature(x_s),
        "img/bboxes/y": float_list_feature(y_s),
        "img/bboxes/width": float_list_feature(w_s),
        "img/bboxes/height": float_list_feature(h_s)

    }

    return tf.train.Example(features=tf.train.Features(feature=data))



main()
