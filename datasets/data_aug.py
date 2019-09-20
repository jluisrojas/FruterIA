import tensorflow as tf
import numpy as np
import cv2
import random

# Metodos para Data Augmentation siguiendo el Dataset API
# de tensorflow, donde
# x, y in dataset:
#   x: tensor con la imagen de shape [w, h, 3]
#   y: tenosr con one_hot encoding de las classes

# Realiza un flip aleatorio a la image
def random_flip(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x, y

# Aplica augmentacion al color de las imagenes
def color_aug(x, y):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)

    return x, y

def expand_image(x, y):
    image = np.array(x)

    height, width, depth = image.shape

    ratio = random.uniform(1, 2)

    left = random.uniform(0, width*ratio - width)
    top = random.uniform(0, height*ratio - height)

    expand_image = np.zeros((int(height*ratio), int(width*ratio), depth),
        dtype=image.dtype)
    expand_image[:, :, :] = np.mean(image)
    expand_image[int(top):int(top+height), int(left):int(left+width)] = image
    image = tf.convert_to_tensor(expand_image)

    return image, y

# Aplica una expansion a la image
class ImageExpand():
    def __init__(self, ratio, fill_type="w"):
        self.ratio = ratio
        self.fill_type = fill_type

    def expand_image(self, x, y):
        image = np.array(x)

        height, width, depth = image.shape

        ratio = random.uniform(1, self.ratio)

        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros((int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = np.mean(image)
        expand_image[int(top):int(top+height), int(left):int(left+width)] = image
        image = expand_image

        return tf.convert_to_tensor(image), y
