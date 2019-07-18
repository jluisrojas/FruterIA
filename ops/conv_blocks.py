import tensorflow as tf
from tensorflow.keras import layers
from ops import conv_ops as ops


#
# Bloque de capas de mobilenetV1 que realiza lo siguiente
#   > 3x3 Depthwise conv, stride=(1|2)
#   > Batch Normalization
#   > ReLU Activation
#   > 1x1xfilters Conv (Pointwise conv)
#   > Batch Normalization
#   > ReLU Activations
#
class D1x1Block(layers.Layer):
    #
    # Crea el bloque segun los argumentos
    # Args:
    #   filters: numero de filtros que realizara la Pointwise Conv
    #   stride: stride de la layer Depthwise Conv, 1 o 2
    #   name: nombre del bloque
    #
    def __init__(self, filters, stride, name="D1x1Block", **kwargs):
        super(D1x1Block, self).__init__(name=name, **kwargs)
        # deptwise operation
        self.dwise = ops.depthwise_conv((3,3), strides=[1, stride, stride, 1])
        self.dwbn = layers.BatchNormalization()
        self.dwrelu = layers.Activation("relu")

        #point wise operation
        self.pwise = ops.pointwise_conv(filters)
        self.pwbn = layers.BatchNormalization()
        self.pwrelu = layers.Activation("relu")

    def call(self, inputs):
        x = self.dwise(inputs)
        x = self.dwbn(x)
        x = self.dwrelu(x)

        x = self.pwise(x)
        x = self.pwbn(x)
        x = self.pwrelu(x)

        return x

#
# Bloque basico para MobileNetV2
class BottleneckResidualBlock(layers.Layer):

    def __init__(self, input_channels, filters, stride=1, t=6, name="BottleneckResidualBlock", **kwargs):
        super(BottleneckResidualBlock, self).__init__(name=name, **kwargs)
        self.stride = stride

        self.pw_exp = ops.pointwise_conv(input_channels * t)
        self.dwise =  ops.depthwise_conv((3,3), strides=[1, stride, stride, 1])
        self.pw_bottleneck = ops.ops.pointwise_conv(filters)

    def call(self, inputs):
        residual = inputs

        x = self.pw_exp(inputs)
        x = self.dwise(x)
        x = self.pw_bottleneck(x)

        if self.stride == 1:
            x = tf.add(x, residual)

        return x
