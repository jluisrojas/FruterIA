import tensorflow as tf
import numpy as np
import cv2
import os
import json
import sys
from glob import glob
import fire

# Function that loads the AOB dataset from a tfrecord
# Args:
#   path: path to the folder that contains the tfrecord's
# Returns:
#   train_data: Tf Dataset API object, with the training data
#   test_data: Tf Dataset API object, with the test data
#   info: dictionary containing information about the data set
def load_dataset(path="", color_data=False, color_type="RGB"):
    train_path = path+"aob_train.tfrecord"
    test_path = path+"aob_test.tfrecord"

    train_raw_data = tf.data.TFRecordDataset(train_path)
    test_raw_data = tf.data.TFRecordDataset(test_path)

    _format = {
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string)
    }

    if color_data and not color_type=="KMeans":
        _format["color"] = tf.io.FixedLenFeature([], tf.string)

    def _parse_example(example):
        ex = tf.io.parse_single_example(example, _format)
        x = tf.io.parse_tensor(ex["x"], tf.float32)
        y = tf.io.parse_tensor(ex["y"], tf.float32)
        y = tf.reshape(y, [-1])

        if color_data and not color_type == "KMeans":
            c = tf.io.parse_tensor(ex["color"], tf.float32)
            if color_type == "RGB":
                c = tf.reshape(c, [3])
            elif color_type == "HIST":
                c = tf.reshape(c, [765])
            return x, c, y

        return x, y

    train_data = train_raw_data.map(_parse_example)
    test_data = test_raw_data.map(_parse_example)

    # Sets the images shape (extrange things of tensorflow)
    def _set_dataset_shape(x, y):
        x.set_shape([224, 224, 3])
        return x, y
    def _set_dataset_shape_c(x, c, y):
        x.set_shape([224, 224, 3])
        return x, c, y

    if color_data and not color_type == "KMeans":
        train_data = train_data.map(_set_dataset_shape_c)
        test_data = test_data.map(_set_dataset_shape_c)
    else:
        train_data = train_data.map(_set_dataset_shape)
        test_data = test_data.map(_set_dataset_shape)


    with open(path+"dataset_info.json", "r") as data:
        info = json.load(data)

    return train_data, test_data, info

# Function that encodes information, and writes it to the TFRecord Writer
# Args:
#   image: numpy array with the image
#   category: tensor with the onehot encoding
#   writer: object of type TFRecordWriter
def _encode_image(image, category, writer, include_color, color):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor /= 255
    data = {
        "x": bytes_feature(tf.io.serialize_tensor(image_tensor)),
        "y": bytes_feature(tf.io.serialize_tensor(category))
    }

    if include_color:
        tf.cast(color, dtype=tf.float32)
        data["color"] = bytes_feature(tf.io.serialize_tensor(color))

    example = tf.train.Example(features=tf.train.Features(feature=data))
    writer.write(example.SerializeToString())

# Creates a template (dict) for the dataset information
# Args
#   domains: list of domains in the dataset
#   categories: list with the categories of the dataset
# Returns:
#   info: dictionary to store the information
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

# Creates TFRecords of the AOB dataset, in order to use it with the
# tensorflow Dataset API
# Args:
#   path: path of the result folder to store the tfrecords
#   include_bag: True if you want to include the photos with bag
#   include_color: if add color information to the dataset
#   color_type: if include color, what type of color to include
def create_dataset(path="AOB_TF", include_bag=True, include_color=True,
        color_type="RGB"):
    domains = ["train", "test"]
    categories = ["apple", "orange", "banana"]
    types = ["noBag"]
    if include_bag: types.append("bag")

    info = info_template(domains, categories)

    info["include_color"] = include_color
    if include_color:
        info["color_type"] = color_type

    onehot_dict = { }
    for i in range(len(categories)):
        onehot_dict[categories[i]] = i

    if not os.path.exists(path):
        os.makedirs(path)

    for domain in domains:
        writer = tf.io.TFRecordWriter(path+"/aob_"+domain+".tfrecord")
        for category in categories:
            color = None
            if include_color:
                # If color data is of type RGB, hard code the color depending
                # on the category.
                if color_type == "RGB":
                    color = np.zeros((3, 1))
                    if category == "apple":
                        color[0] = 255 / 255
                        color[1] = 255 / 255
                    if category == "banana":
                        color[0] = 255 / 255
                        color[1] = 255 / 255
                    if category == "orange":
                        color[0] = 255 / 255
                        color[1] = 165 / 255

                    color = tf.convert_to_tensor(color)

            c_onehot = tf.one_hot([onehot_dict[category]], len(categories))

            for typ in types:
                p = os.path.join(domain, category, typ)
                img_paths = glob(p+"/*")

                for img in img_paths:
                    image = cv2.imread(img)

                    # if include color and color type, obtain the histogram
                    # information of the color
                    if include_color:
                        if color_type == "HIST":
                            h_red = cv2.calcHist([image],[2],None,[255],[0,255])
                            h_green = cv2.calcHist([image],[1],None,[255],[0,255])
                            h_blue = cv2.calcHist([image],[0],None,[255],[0,255])

                            # Normalizes the histogram
                            hist = np.concatenate((h_red, h_green, h_blue), 0)
                            hist /= np.max(hist)

                            color = tf.convert_to_tensor(hist)

                    _encode_image(image, c_onehot, writer, include_color, color)

                info[domain+"_size"] += len(img_paths)
                info[category][domain+"_size"] += len(img_paths)

        writer.close()

    with open(path+"/dataset_info.json", "w") as writer:
        json.dump(info, writer, indent=4)

def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# If it is executing as main, create the dataset
if __name__ == "__main__":
    fire.Fire(create_dataset)
