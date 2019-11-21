import tensorflow as tf
import numpy as np
import cv2

def plot_image(image_tensor):
    image = image_tensor.numpy()
    cv2.imshow("Image", image)
    cv2.waitKey()
