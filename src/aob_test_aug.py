import tensorflow as tf
import numpy as np

from datasets.data_aug import *
from datasets.data_visualization import plot_image
from datasets.AOBDataset.aob_dataset import load_dataset

def main():
    dataset_path = "datasets/AOBDataset/AOB_TF/"

    train, test, info = load_dataset(path=dataset_path)

    for x, y in train.take(5):
        plot_image(x)
        x, _ = color_aug(x, y)
        x, _ = random_flip(x, y)
        plot_image(x)

main()
