import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
# from ops.conv_ops import normal_conv, depthwise_conv, pointwise_conv

"""
class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                  trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

x = tf.ones((2, 2))
linear_layer = Linear(64)
y = linear_layer(x)
print(linear_layer.get_config())

conv = pointwise_conv(20)
x = tf.ones((1, 3, 3, 3))
res = conv(x)
print(res)
"""


x = np.array([[1, 2, 3, 4],
              [1, 2, 3, 4]])

y = np.zeros((4, 4, 3 * 4))

z = np.array([[[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]],

              [[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]], 
              
              [[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]],

              [[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
               [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]])
print(z.shape)

tensor = tf.convert_to_tensor(z)

print(tensor)

reshaped_tensor = tf.reshape(tensor, [4, 4, 3, 4])

print(reshaped_tensor)



