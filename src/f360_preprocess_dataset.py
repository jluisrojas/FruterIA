import tensorflow as tf
from datasets.Fruits360.f360_dataset import f360_create_dataset

def main():
    # Inicializa los paths
    training_path = "datasets/Fruits360/training/"
    test_path = "datasets/Fruits360/test/"
    result_path = "datasets/Fruits360/F360-3-N/"

    print("[INFO] Creating Fruits 360 Dataset")
    # Crea el dataset
    f360_create_dataset(training_path=training_path, test_path=test_path,
        result_path=result_path, delta=85, offset=0)

main()
