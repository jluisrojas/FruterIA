import tensorflow as tf
import sys
import cv2

# Regresar un directoria para poder acceder modulo de otra carpeta
sys.path.append("..")
from mobilenetv2 import MobileNetV2
sys.path.append("tests/")

model = MobileNetV2(3)

image = cv2.imread("test.jpg")
image = cv2.resize(image, (224, 224))

image_tensor = tf.convert_to_tensor(image)
image_tensor /= 255
image_tensor = tf.expand_dims(image_tensor, 0)
#print(image_tensor)
x = model(image_tensor, training=False)
print(x)

