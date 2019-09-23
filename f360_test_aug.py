import tensorflow as tf
import numpy as np

from datasets.data_aug import *
from datasets.data_visualization import plot_image
from datasets.Fruits360.f360_dataset import f360_load_dataset

def main():
    dataset_path = "datasets/Fruits360/F360-3/"

    train, test, info = f360_load_dataset(path=dataset_path)
    datasets_cats = []

    for cat in info["categories"]:
        train_size, _ = info["categories_size"][cat]
        datasets_cats.append(train.take(train_size))
        train = train.skip(train_size)

    for dataset in datasets_cats:
        for x, y in dataset.take(5):
            plot_image(x)
            x, _ = color_aug(x, y)
            x, _ = expand_image(x, y)
            x, _ = random_flip(x, y)
            plot_image(x)

main()
