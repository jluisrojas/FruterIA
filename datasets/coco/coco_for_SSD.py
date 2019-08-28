#
# Script que procesa 
import tensorflow as tf
import cv2
# Regresar un directoria para poder acceder modulo de otra carpeta
import sys
sys.path.append("..")
from ops.SSD import SSD_data_pipeline, SSD_load_dataset
import mobilenetv2
sys.path.append("datasets/")
print(sys.path)

def main():
    model = mobilenetv2.MobileNetV2_SSD
    fmaps_arr = model.get_fmaps_array()
    cats = ["orange", "apple", "banana"]
    aspect_ratios = [1, 2, 3, 1/2, 1/3]
    img_size = model.get_input_size()

    process_for_ssd(fmaps_arr, cats, img_size, aspect_ratios)

def process_for_ssd(fmaps_array, categories_array, img_size, aspect_ratios):
    pipeline = SSD_data_pipeline(feature_maps=fmaps_array,
            categories_arr=categories_array, img_size=img_size,
            aspect_ratios=aspect_ratios)
    pipeline.preprocess_tfrecord_coco("cocofruits.tfrecord",
            "ssd_preprocess.tfrecord")

    dataset = SSD_load_dataset("ssd_preprocess.tfrecord")

main()


