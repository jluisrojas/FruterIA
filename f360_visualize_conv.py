import os
from os import path
import tensorflow as tf
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

model_path = "trained_models/f360_MobileNetV2_04/"
result_folder = "conv_features/"

def main():
    model = tf.keras.models.load_model(model_path+"model.h5")
    model = model.layers[0] # we only care the block of MobileNetV2
    conv_output = model.get_layer("block_3_project").output

    model = tf.keras.Model(inputs=model.input, outputs=conv_output)

    noise_image = tf.random.uniform([96, 96, 3], minval=150, maxval=180,
            name="image") / 255
    noise_image = tf.expand_dims(noise_image, 0)
    noise_image = tf.Variable(noise_image)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    iterations = 20

    for _ in range(iterations):
        with tf.GradientTape() as t:
            t.watch(noise_image)
            activation = model(noise_image)
            loss = -tf.math.reduce_mean(activation)

        grads = t.gradient(loss, noise_image)
        print(type(grads))
        optimizer.apply_gradients([(grads, noise_image)])

    noise_image *= 255
    noise_image = tf.reshape(noise_image, [96, 96, 3])

    cv2.imshow("image", np.array([noise_image]))
    cv2.waitKey(0)


main()
