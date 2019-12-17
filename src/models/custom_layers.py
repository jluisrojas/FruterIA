import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# Keras layer that extracts the colors from an image
# it uses KMeans clustering algorithm to achieve this
# it works with batches 
class ColorExtractor(layers.Layer):
    # Args:
    #   num_clusters: amount of random initial clusters
    #   num_iterations: amout of update loops
    def __init__(self, num_clusters, num_iterations):
        super(ColorExtractor, self).__init__()
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations

    def build(self, input_shape):
        # Obtains how it will resize the image and the depth of the image
        self.resize = input_shape[1] * input_shape[2]
        self.dims = input_shape[-1]

    def call(self, inputs):
        # Extracts the color of one image
        def extract_color(input_tensor):
            input_tensor = tf.cast(input_tensor, tf.float32)

            input_tensor = tf.reshape(input_tensor, [self.resize, self.dims])

            # Obtains random centroids from the input
            input_shuffle = tf.random.shuffle(input_tensor)
            centroids = tf.constant(tf.slice(input_shuffle, [0, 0], [self.num_clusters, -1]))

            input_expanded = tf.expand_dims(input_tensor, 0)

            # Updates the centroids 
            for _ in range(self.num_iterations):
                centroids, assignments = self.update_centroids(input_tensor, input_expanded, centroids)

            # Flattens the output
            centroids = tf.reshape(centroids, [self.dims * self.num_clusters])

            return centroids

        # Maps every element of the batch 
        return tf.map_fn(extract_color, inputs, dtype=tf.float32)

    # Updates the centroids respectevily to reduce the distance to the centroid
    @tf.function
    def update_centroids(self, input_tensor, input_expanded, centroids):
        centroids_expanded = tf.expand_dims(centroids, 1)

        # Calculate the distance of the centroids in respect with all the inputs
        distances = tf.reduce_sum(tf.square(tf.subtract(input_expanded, centroids_expanded)), 2)
        # Assing a input point to a cluster
        assignments = tf.argmin(distances, 0)

        # Iterates through each cluster
        means = []
        for c in range(self.num_clusters): 
            # Obtains the input points assigned to the cluster
            ruc = tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])
            ruc = tf.gather(input_tensor, ruc)
            # Obtains the mean of the points in order to update the centroid
            ruc = tf.reduce_mean(ruc, axis=[1])
            means.append(ruc)
            new_centroids = tf.concat(means, 0)

        return new_centroids, assignments
