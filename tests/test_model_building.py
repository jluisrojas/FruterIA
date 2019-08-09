import tensorflow as tf
import sys
import cv2

# Regresar un directoria para poder acceder modulo de otra carpeta
sys.path.append("..")
from mobilenetv2 import MobileNetV2_SSD
sys.path.append("tests/")

model = MobileNetV2_SSD(3)

image = cv2.imread("test.jpg")
image = cv2.resize(image, (320, 320))

image_tensor = tf.convert_to_tensor(image)
image_tensor /= 255
image_tensor = tf.expand_dims(image_tensor, 0)
#print(image_tensor)
model(image_tensor, training=True)

