#
# Funciones que desiralizan la base de datos de COCO en el formato de
# TFRecord
import tensorflow as tf

def _image_feature_proto():
    image_feature_description = {
        "img/filename": tf.io.FixedLenFeature([], tf.string),
        "img/width": tf.io.FixedLenFeature([], tf.int64),
        "img/height": tf.io.FixedLenFeature([], tf.int64),
        "img/str": tf.io.FixedLenFeature([], tf.string),
        "img/bboxes/category": tf.io.VarLenFeature(tf.string),
        "img/bboxes/x": tf.io.VarLenFeature(tf.float32),
        "img/bboxes/y": tf.io.VarLenFeature(tf.float32),
        "img/bboxes/width": tf.io.VarLenFeature(tf.float32),
        "img/bboxes/height": tf.io.VarLenFeature(tf.float32)
    }

    return image_feature_description

def parse_dataset(path):
    raw_data = tf.data.TFRecordDataset(path)
    proto = _image_feature_proto()
    
    def _parse_function(example):
        return tf.io.parse_single_example(example, proto)
    
    data = raw_data.map(_parse_function)

    return data
