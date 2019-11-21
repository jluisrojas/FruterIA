import tensorflow as tf
import tensorflow_datasets as tfds

def load_mnist_dataset():
    train = tfds.load(name="mnist", split=tfds.Split.TRAIN)
    test = tfds.load(name="mnist", split=tfds.Split.TEST)

    def _preprocess_mnist(features):
        image = features["image"]
        label = features["label"]

        one_hot = tf.one_hot(label, 10)
        
        return image, one_hot

    train = train.map(_preprocess_mnist)
    test = test.map(_preprocess_mnist)

    return train, test

def load_mnist_dataset_resize(resize):
    train = tfds.load(name="mnist", split=tfds.Split.TRAIN)
    test = tfds.load(name="mnist", split=tfds.Split.TEST)

    def _preprocess_mnist(features):
        image = features["image"]
        label = features["label"]

        image = tf.image.resize(image, [resize, resize])
        one_hot = tf.one_hot(label, 10)
        
        return image, one_hot

    train = train.map(_preprocess_mnist)
    test = test.map(_preprocess_mnist)

    return train, test

