import tensorflow as tf
import cv2
import os
import json
import sys
from glob import glob
import fire

def load_dataset(path=""):
    train_path = path+"aob_train.tfrecord"
    test_path = path+"aob_test.tfrecord"

    train_raw_data = tf.data.TFRecordDataset(train_path)
    test_raw_data = tf.data.TFRecordDataset(test_path)

    _format = {
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_example(example):
        ex = tf.io.parse_single_example(example, _format)
        x = tf.io.parse_tensor(ex["x"], tf.float32)
        y = tf.io.parse_tensor(ex["y"], tf.float32)
        y = tf.reshape(y, [-1])

        data_dict = {
            "x": x,
            "y": y
        }

        return x, y

    train_data = train_raw_data.map(_parse_example)
    test_data = test_raw_data.map(_parse_example)

    def _set_dataset_shape(x, y):
        x.set_shape([224, 224, 3])

        return x, y

    train_data = train_data.map(_set_dataset_shape)
    test_data = test_data.map(_set_dataset_shape)

    with open(path+"dataset_info.json", "r") as data:
        info = json.load(data)


    return train_data, test_data, info

def encode_image(image, category, writer):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor /= 255
    data = {
        "x": bytes_feature(tf.io.serialize_tensor(image_tensor)),
        "y": bytes_feature(tf.io.serialize_tensor(category))
    }

    example = tf.train.Example(features=tf.train.Features(feature=data))
    writer.write(example.SerializeToString())

def info_template(domains, categories):
    info = {"about": "Apple Orange Banana Dataset"}
    info["categories"] = categories
    info["num_classes"] = len(categories)

    for domain in domains:
        info[domain+"_size"] = 0
        for category in categories:
            if not category in info.keys():
                info[category] = { }

            info[category][domain+"_size"] = 0

    return info

def create_dataset(path="AOB_TF", include_bag=True):
    domains = ["train", "test"]
    categories = ["apple", "orange", "banana"]
    types = ["noBag"]
    if include_bag: types.append("bag")

    info = info_template(domains, categories)

    onehot_dict = { }
    for i in range(len(categories)):
        onehot_dict[categories[i]] = i

    if not os.path.exists(path):
        os.makedirs(path)

    for domain in domains:
        writer = tf.io.TFRecordWriter(path+"/aob_"+domain+".tfrecord")
        for category in categories:
            c_onehot = tf.one_hot([onehot_dict[category]], len(categories))

            for typ in types:
                p = os.path.join(domain, category, typ)
                img_paths = glob(p+"/*")

                for img in img_paths:
                    image = cv2.imread(img)
                    encode_image(image, c_onehot, writer)

                info[domain+"_size"] += len(img_paths)
                info[category][domain+"_size"] += len(img_paths)

        writer.close()

    with open(path+"/dataset_info.json", "w") as writer:
        json.dump(info, writer, indent=4)

def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == "__main__":
    fire.Fire(create_dataset)
