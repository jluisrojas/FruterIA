import tensorflow as tf
from tensorflow.keras import Model
from ops.conv_ops import normal_conv
from ops.conv_blocks import D1x1Block

#
# Implementacion de MobilenetV1, suponiendo un input size de 224x224x3
#

class MobileNetV1(Model):
    def __init__(self, classes, width_multiplier=1):
        super(MobileNetV1, self).__init__()
        self.m_layers = []
        a = width_multiplier
        self.m_layers.append(normal_conv(int(a*32), (3, 3), strides=[1, 2, 2, 1]))
        self.m_layers.append(tf.keras.layers.BatchNormalization())
        self.m_layers.append(tf.keras.layers.Activation("relu"))
        self.m_layers.append(D1x1Block(int(a*64), 1))
        self.m_layers.append(D1x1Block(int(a*128), 2))
        self.m_layers.append(D1x1Block(int(a*128), 1))
        self.m_layers.append(D1x1Block(int(a*256), 2))
        self.m_layers.append(D1x1Block(int(a*256), 1))
        self.m_layers.append(D1x1Block(int(a*512), 2))

        for _ in range(5):
            self.m_layers.append(D1x1Block(int(a*512), 1))

        self.m_layers.append(D1x1Block(int(a*1024), 2))
        self.m_layers.append(D1x1Block(int(a*1024), 1))

        self.m_layers.append(tf.keras.layers.AveragePooling2D(pool_size=(7,7), strides=(1,1)))
        self.m_layers.append(tf.keras.layers.Flatten())
        self.m_layers.append(tf.keras.layers.Dense(1024))
        self.m_layers.append(tf.keras.layers.Dropout(0.5, name="dropout"))
        self.m_layers.append(tf.keras.layers.Dense(classes))
        self.m_layers.append(tf.keras.layers.Activation("softmax"))

    def call(self, inputs, training=False):
        x = inputs
        for l in self.m_layers:
            # print(x.get_shape().as_list())
            if (l.name == "dropout" and training == False) == False:
                x = l(x)

        return x
