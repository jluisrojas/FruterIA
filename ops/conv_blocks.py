import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.nn import relu6
from ops import conv_ops as ops

# Bloque comun de convolucion que consiste:
#   > conv2d
#   > batch normalization
#   > activation
#   > dropout     
class basic_conv_block(layers.Layer):
    # Args:
    #   filters: entero con el numero de filtros de la convolucion
    #   kernel: par de numeros con tamaño del kernel
    #   strides: arreglo de strides
    #   dropout: fraccion a la cual se le va aplicar dropout
    #   activation: el topo de activacion de la capa
    def __init__(self, 
            filters, 
            kernel, 
            stride=1, 
            dropout=0.25, 
            activation="ReLU", 
            name="conv_block", **kwargs):
        super(basic_conv_block, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = [1, stride, stride, 1]
        self.dropout = dropout
        self.activation = activation

        self.conv = ops.normal_conv(filters, kernel, strides=self.strides, name=name+"_conv2d")
        self.bn = layers.BatchNormalization(name=name+"_bn")

        if self.activation == "ReLU":
            self.activation = layers.Activation("relu", name=name+"_relu")
        if self.activation == "ReLU6":
            self.activation = ops.ReLU6(name=name+"_relu6")

        self.dropout = layers.Dropout(dropout, name=name+"_dropout")
    #
    # serializa la configuracion de la capa
    def get_config(self):
        config = super(basic_conv_block, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel": self.kernel,
            "stride": self.strides[1],
            "dropout": self.dropout,
            "activation": self.activation
        })

    def call(self, inputs, training=None):
        # Operacion depth wise
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation(x)
        if training == True:
            x = self.dropout(x)

        return x

# Bloque de pointwise convolution
#   > pointwise
#   > batch normalization
#   > activation
#   > dropout     
class pwise_conv_block(layers.Layer):
    # Args:
    #   filters: entero con el numero de filtros de la convolucion
    #   strides: arreglo de strides
    #   dropout: fraccion a la cual se le va aplicar dropout
    #   activation: el topo de activacion de la capa
    def __init__(self, 
            filters, 
            stride=1, 
            dropout=0.25, 
            activation="ReLU", 
            name="pwise_conv_block", **kwargs):
        super(pwise_conv_block, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.strides = [1, stride, stride, 1]
        self.dropout = dropout
        self.activation = activation

        self.conv = ops.pointwise_conv(filters, strides=self.strides, name=name+"_pwise_conv")
        self.bn = layers.BatchNormalization(name=name+"_bn")

        if self.activation == "ReLU":
            self.activation = layers.Activation("relu", name=name+"_relu")
        if self.activation == "ReLU6":
            self.activation = ops.ReLU6(name=name+"_relu6")

        self.dropout = layers.Dropout(dropout, name=name+"_dropout")
    #
    # serializa la configuracion de la capa
    def get_config(self):
        config = super(pwise_conv_block, self).get_config()
        config.update({
            "filters": self.filters,
            "stride": self.strides[1],
            "dropout": self.dropout,
            "activation": self.activation
        })

    def call(self, inputs, training=None):
        # Operacion depth wise
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation(x)
        if training == True:
            x = self.dropout(x)

        return x


#
# Bloque de capas de mobilenetV1 que realiza lo siguiente
#   > 3x3 Depthwise conv, stride=(1|2)
#   > Batch Normalization
#   > ReLU Activation
#   > 1x1xfilters Conv (Pointwise conv)
#   > Batch Normalization
#   > ReLU Activations
#
class separable_conv_block(layers.Layer):
    #
    # Crea el bloque segun los argumentos
    # Args:
    #   filters: numero de filtros que realizara la Pointwise Conv
    #   stride: stride de la layer Depthwise Conv, 1 o 2
    #   name: nombre del bloque
    #
    def __init__(self, 
            filters,
            stride,
            dropout=0.25,
            name="separable_conv_block", **kwargs):
        super(separable_conv_block, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.stride = stride
        self.dropout = dropout

        # Asegura de que el filters sea un entero
        if type(filters) is float:
            filters = int(filters)

        # deptwise operation
        self.dwise = ops.depthwise_conv((3,3), strides=[1, stride, stride, 1])
        self.dwbn = layers.BatchNormalization()
        self.dwrelu = layers.Activation("relu")
        self.dwdo = layers.Dropout(dropout)

        #point wise operation
        self.pwise = ops.pointwise_conv(filters)
        self.pwbn = layers.BatchNormalization()
        self.pwrelu = layers.Activation("relu")
        self.pwdo = layers.Dropout(dropout)

    #
    # serializa la configuracion de la capa
    def get_config(self):
        config = super(separable_conv_block, self).get_config()
        config.update({
            "filters": self.filters,
            "stride": self.stride,
            "dropout": self.dropout
        })

    def call(self, inputs, training=None):
        # Operacion depth wise
        x = self.dwise(inputs)
        x = self.dwbn(x)
        x = self.dwrelu(x)
        if training == True:
            x = self.dwdo(x)

        # Luego point wise convolution
        x = self.pwise(x)
        x = self.pwbn(x)
        x = self.pwrelu(x)
        if training == True:
            x = self.pwdo(x)

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
    def __init__(self, 
                 input_channels, 
                 filters, 
                 stride=1, 
                 t=6, 
                 dropout=0.25, 
                 store_output=False,
                 name="BottleneckResidualBlock", **kwargs):
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
        self.t = t
        self.dropout = dropout
        self.store_output = store_output

        self.expansion_output = None
        self.block_output = None

        self.pw_exp = ops.pointwise_conv(input_channels * t, name=name + "_expansion_conv")
        self.bn_exp = layers.BatchNormalization(name=name+"_expansion_bn")
        self.do_exp = layers.Dropout(self.dropout, name=name+"_expansion_do")

        self.dwise =  ops.depthwise_conv((3,3), strides=[1, stride, stride, 1], name=name+"_depthwise_conv")
        self.bn_dwise = layers.BatchNormalization(name=name+"_depthwise_bn")
        self.do_dwise = layers.Dropout(self.dropout, name=name+"_depthwise_do")

        self.pw_bottleneck = ops.pointwise_conv(self.output_channels, name=name+"_bottleneck_conv")
        self.bn_bottleneck = layers.BatchNormalization(name=name+"_bottleneck_bn")
        self.do_bottleneck = layers.Dropout(self.dropout, name=name+"_bottleneck_do")

        # En caso de que el input y output no concuerden,
        # se realiza un 1x1 conv para que concuerdes
        # if self.input_channels != self.output_channels:
        #     self.pw_residual = ops.pointwise_conv(self.output_channels)

    #
    # serializa la configuracion de la capa
    def get_config(self):
        config = super(BottleneckResidualBlock, self).get_config()
        config.update({
            "input_channels": self.input_channels,
            "filters": self.output_channels,
            "stride": self.stride,
            "t": self.t,
            "dropout": self.dropout,
            "store_output": self.store_output
        })

    def call(self, inputs, training=None):
        residual = inputs

        # Expansion de los channels de entrada
        x = self.pw_exp(inputs)
        x = self.bn_exp(x)
        x = relu6(x)
        if training == True:
            x = self.do_exp(x)
        res_expansion = x

        # Realisamos la depthwise convolution
        x = self.dwise(x)
        x = self.bn_dwise(x)
        x = relu6(x)
        if training == True:
            x = self.do_dwise(x)
        res_depthwise = x

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
            x = self.do_bottleneck(x)

        res_bottleneck = x

        return [res_expansion, res_depthwise, res_bottleneck]
