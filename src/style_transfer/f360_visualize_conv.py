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
    if not path.exists(model_path+result_folder):
        os.makedirs(model_path+result_folder)

    model = tf.keras.models.load_model(model_path+"model.h5")
    model = model.layers[0] # we only care the block of MobileNetV2
    conv_output = model.get_layer("block_15_project").output

    model = tf.keras.Model(inputs=model.input, outputs=conv_output)

    noise_image = tf.random.uniform([96, 96, 3], minval=150, maxval=180,
            name="image") / 255
    noise_image = tf.expand_dims(noise_image, 0)
    noise_image = tf.Variable(noise_image)

    iterations = 10

    for i in range(iterations):
        print("Iteration: {}".format(i+1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        with tf.GradientTape() as t:
            t.watch(noise_image)
            activation = model(noise_image)[0, :, :, 10]
            loss = -tf.math.reduce_mean(activation)

        grads = t.gradient(loss, noise_image)
        optimizer.apply_gradients([(grads, noise_image)])
        noise_image.assign(tf.clip_by_value(noise_image, 0.0, 1.0))

    noise_image = noise_image.read_value()
    noise_image = tf.squeeze(noise_image, 0)

    image = np.array(noise_image)
    image = cv2.resize(image, (400, 400))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.imwrite(model_path+result_folder+"filter.jpg", image)


main()
