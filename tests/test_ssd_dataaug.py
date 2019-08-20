import tensorflow as tf
import numpy as np
import cv2
import json
import random

# Regresar un directoria para poder acceder modulo de otra carpeta
import sys
sys.path.append("..")
from ops.SSD import iou_batch, intersection_over_union
sys.path.append("tests/")

from test_bboxes import draw_bbox

def main():
    path_to_image = "test_images/000000000670.jpg"
    path_to_ann = "test_images/000000000670.jpg.json"

    with open(path_to_ann) as json_text:
        ann = json.loads(json_text.read())

    image = cv2.imread(path_to_image)
    bboxes_numpy = np.ones((len(ann["bboxes"]), 4))
    for i in range(len(ann["bboxes"])):
        bbox = ann["bboxes"][i]
        x = bbox["center_x"]
        y = bbox["center_y"]
        w = bbox["width"]
        h = bbox["height"]
        bboxes_numpy[i, :] = [x, y, w, h]
        #bboxes_tensor[i, 0] = x
        draw_bbox(img=image, bbox=(x, y, w, h))

    cv2.imshow("test_data_aug", image)
    cv2.waitKey(0)

    img_w = image.shape[1]
    img_h = image.shape[0]

    aug_w = random.uniform(0.1*img_w, img_w)
    aug_h = random.uniform(0.1*img_h, img_h)
    
    aug_x = random.uniform(0, img_w-aug_w)
    aug_y = random.uniform(0, img_h-aug_h)

    draw_bbox(img=image, bbox=(aug_x, aug_y, aug_w, aug_h), color=(255,0,0))
    cv2.imshow("test_data_aug", image)
    cv2.waitKey(0)

    #print(bboxes_numpy)
    patch = np.array([[aug_x, aug_y, aug_w, aug_h]])
    print(patch)
    iou_batch(bboxes_numpy, patch)
    for i in range(bboxes_numpy.shape[0]):
        intersection_over_union(bboxes_numpy[i], patch[0])

main()
