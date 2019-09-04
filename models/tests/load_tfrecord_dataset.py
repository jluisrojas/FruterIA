import sys
import numpy as np
import tensorflow as tf
import cv2

# Regresar un directoria para poder acceder modulo de otra carpeta
sys.path.append("..")
from utils import tfrecord_coco as dataset 
sys.path.append("tests/")

data = dataset.parse_dataset("../utils/prueba.tfrecord")

for img in data:
    # image_string = np.fromstring(img["img/str"].numpy(), np.uint8)
    image_string = np.frombuffer(img["img/str"].numpy(), np.uint8)
    decoded_image = cv2.imdecode(image_string, cv2.IMREAD_COLOR) 
    print(type(decoded_image))
    cv2.imshow("coco_image", decoded_image)
    cv2.waitKey(0)

