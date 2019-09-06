import tensorflow as tf
import numpy as np

import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def graph_confusion_matrix(model=None, test_dataset=None, classes=None, path=None):
    # Desbatchea el dataset
    test_unbatch = test_dataset.unbatch()

    y_predict = None
    for x, y in test_unbatch:
        x_e = tf.expand_dims(x, 0)
        x_e = tf.cast(x_e, tf.float32)
        temp = model.predict(x_e)
        temp = tf.expand_dims(tf.argmax(temp, axis=-1), axis=0)
        if isinstance(y_predict, tf.Tensor):
            y_predict = tf.concat([y_predict, temp], 0)
        else:
            y_predict = tf.identity(temp)


    y_true = None
    for x, y in test_unbatch:
        temp = tf.expand_dims(tf.argmax(y), axis=0)
        if isinstance(y_true, tf.Tensor):
            y_true = tf.concat([y_true, temp], 0)
        else:
            y_true = tf.identity(temp)

    confusion_mtrx = tf.math.confusion_matrix(labels=y_true,
            predictions=y_predict).numpy()

    confusion_mtrx_df = pd.DataFrame(confusion_mtrx, index=classes,
            columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_mtrx_df, annot=True,cmap=plt.cm.Blues)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)


def graph_model_metrics(csv_path=None, img_path=None):
    data = pd.read_csv(csv_path)
    data = data.loc[:, data.sub(data["epoch"], axis=0)]
    figure = plt.figure()
    data.plot()
    plt.savefig(img_path)

