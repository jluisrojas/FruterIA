import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.nn import relu6
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

        # Asegura de que el filters sea un entero
        if type(filters) is float:
            filters = int(filters)

        # deptwise operation
        self.dwise = ops.depthwise_conv((3,3), strides=[1, stride, stride, 1])
        self.dwbn = layers.BatchNormalization()
        self.dwrelu = layers.Activation("relu")

        #point wise operation
        self.pwise = ops.pointwise_conv(filters)
        self.pwbn = layers.BatchNormalization()
        self.pwrelu = layers.Activation("relu")

    def call(self, inputs):
        # Operacion depth wise
        x = self.dwise(inputs)
        x = self.dwbn(x)
        x = self.dwrelu(x)

        # Luego point wise convolution
        x = self.pwise(x)
        x = self.pwbn(x)
        x = self.pwrelu(x)

        return x

#
# Bloque basico para MobileNetV2, realiza lo siguiente:
#   > (1x1xinput_channels*t) conv
#   > Batch Normalization
#   > ReLU6
#   > 3x3 Depthwise conv, stride=(1|2)
#   > Batch Normalization
#   > ReLU6
#   > (1x1xoutput_channels) conv
#   > Si stride == 1 entonces residual = output + input
#
class BottleneckResidualBlock(layers.Layer):
    #
    # Crea el bloque segun los argumentos
    # Args:
    #   input_channels: numero de channels que entraran al bloque
    #   filters: numero de filtros del volumen final
    #   stride: stride de la layer Depthwise Conv, 1 o 2
    #   t: expansion factor, por defecto 6
    #   dropout: cantidad de dropout que se realizara
    #   name: nombre del bloque
    #
    def __init__(self, input_channels, filters, stride=1, t=6, dropout=0.25, name="BottleneckResidualBlock", **kwargs):
        super(BottleneckResidualBlock, self).__init__(name=name, **kwargs)

        # Asegura de que el input_channels sea un entero
        if type(input_channels) is float:
            input_channels = int(input_channels)
        # Asegura de que el filters sea un entero
        if type(filters) is float:
            filters = int(filters)

        self.input_channels = input_channels
        self.output_channels = filters
        self.stride = stride
        self.dropout = dropout

        self.pw_exp = ops.pointwise_conv(input_channels * t)
        self.bn_exp = layers.BatchNormalization()

        self.dwise =  ops.depthwise_conv((3,3), strides=[1, stride, stride, 1])
        self.bn_dwise = layers.BatchNormalization()

        self.pw_bottleneck = ops.pointwise_conv(self.output_channels)
        self.bn_bottleneck = layers.BatchNormalization()

        # En caso de que el input y output no concuerden,
        # se realiza un 1x1 conv para que concuerdes
        # if self.input_channels != self.output_channels:
        #     self.pw_residual = ops.pointwise_conv(self.output_channels)

    def call(self, inputs, training=None):
        residual = inputs

        # Expansion de los channels de entrada
        x = self.pw_exp(inputs)
        x = self.bn_exp(x)
        x = relu6(x)
        if training == True:
            x = layers.Dropout(self.dropout)

        # Realisamos la depthwise convolution
        x = self.dwise(x)
        x = self.bn_dwise(x)
        x = relu6(x)
        if training == True:
            x = layers.Dropout(self.dropout)

        # Bottleneck para reducir los channels de salida
        x = self.pw_bottleneck(x)
        x = self.bn_bottleneck(x)

        # checa si hay que sumar el residual
        if self.stride == 1:
            if self.input_channels == self.output_channels:
                x = x + residual
                #residual = self.pw_residual(residual)

            #x = x + residual

        if training == True:
            x = layers.Dropout(self.dropout)

        return x
