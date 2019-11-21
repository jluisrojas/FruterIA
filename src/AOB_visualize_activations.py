import os
from os import path
import tensorflow as tf
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

#image_path = "datasets/Fruits360/test/Banana/86_100.jpg"
image_path = "datasets/AOBDataset/test/apple/noBag/1-VA-NB-1.JPG"
model_path = "trained_models/AOB_MobileNetV2_04/"
result_folder = "activations01/"

def main():
    print("[INFO] Ploting images activation")

    if not path.exists(model_path+result_folder):
        os.makedirs(model_path+result_folder)

    image = cv2.imread(image_path)
    image_tensor = tf.convert_to_tensor(image)
    #image_tensor = tf.image.resize(image_tensor, [96, 96])
    image_tensor = tf.expand_dims(image_tensor, 0) / 255

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(model_path+"model.h5")
    mnetBlock = model.layers[0]

    layers_output = [layer.output for layer in mnetBlock.layers]
    activationModel = tf.keras.Model(inputs=mnetBlock.input, outputs=layers_output)

    activations = activationModel.predict(image_tensor)

    layer_names = []
    for layer in mnetBlock.layers[:]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
        
    images_per_row = 16
    count = 0

    for layer_name, layer_activation in zip(layer_names, activations[:]): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

        if n_cols != 0:
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                     :, :,
                                                     col * images_per_row + row]
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size, # Displays the grid
                                 row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            #print(display_grid.shape)
            plt.imsave(model_path+result_folder+str(count)+"_"+layer_name+".png", display_grid, format="png", cmap='viridis')
            count += 1

    print("Finished")

main()
