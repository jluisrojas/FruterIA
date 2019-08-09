import tensorflow as tf

from tensorflow.nn import depthwise_conv2d, conv2d, bias_add, relu6
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import l2

#
# Layer que realiza activacion ReLU6
#
class ReLU6(layers.Layer):
    def __init__(self, name="ReLU6", **kwargs):
        super(ReLU6, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return relu6(inputs)

#
# Layer que realiza una convulucion estandar
#
class normal_conv(layers.Layer):
    #
    # Asigna los parametros de layer para realizar la convulucion
    # Args:
    #   name: nombre de la layer
    #   f_kernel: tamaño del kernel que realiza la convulucion
    #   num_filters: el numero de filtros de la convulucion
    #   strides: el stride de la convolucion
    #   padding: el padding que se aplicara, por defecto 'SAME'
    #   intializer: para los pesos, por defecto GlorotNormal (Xavier)
    #   regularizer: para los pesos, por defecto L2
    #   use_bias: si se aplica bias despues de la convulucion
    #   weight_decay: hyperparametro para regularizacion
    #
    def __init__(self,
                 num_filters,
                 f_kernel,
                 name="normal_conv",
                 strides=[1,1,1,1],
                 padding="SAME",
                 initializer=None,
                 regularizer=None,
                 use_bias=False,
                 weight_decay=1e-4,
                 **kwargs):

        super(normal_conv, self).__init__(name=name, **kwargs)

        # Asegura de que el num_filters sea un entero
        if type(num_filters) is float:
            num_filters = int(num_filters)

        self.f_kernel = f_kernel
        self.num_filters = num_filters
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

        if initializer == None:
            self.w_initializer = GlorotNormal()
        else:
            self.w_initializer = initializer

        if regularizer == None:
            self.w_regularizer = l2(weight_decay)
        else:
            selw.w_regularizer = regularizer

    #
    # Serializa las propiedades de la capa
    def get_config(self):
        config = super(normal_conv, self).get_config()
        config.update({
            "num_filters": self.num_filters,
            "f_kernel": self.f_kernel,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias
        })

        return config

    #
    # Crea weights y biases dependiendo del input_shape de call()
    #
    def build(self, input_shape):
        # dimensiones de la convulucion
        nc_h, nc_w = self.f_kernel
        in_channels = input_shape[-1]
        self.w = self.add_weight(shape=[nc_h, nc_w, in_channels, self.num_filters],
                                 initializer=self.w_initializer,
                                 regularizer=self.w_regularizer,
                                 trainable=True,
                                 name=self.name + "_w")
        if self.use_bias:
            self.b = self.add_weight(shape=[self.num_filters],
                                     initializer="zeros",
                                     trainable=True,
                                     name=self.name+"_b")
    #
    # Realiza la operacion al argumento inputs
    # Args:
    #   inputs: tensor de shape (batch, heigh, width, channels)
    #
    def call(self, inputs, training=None):
        conv = conv2d(inputs, self.w, self.strides, self.padding)
        if self.use_bias:
            return bias_add(conv, self.b)

        return conv

#
# Layer que realiza una depthwise convolution, a la
# cual solo se le aplica un filtro a cada channel del input
#
class depthwise_conv(layers.Layer):
    #
    # Asigna los parametros de layer para realizar la depthwsie convulution
    # Args:
    #   name: nombre de la layer
    #   f_kernel: tamaño del kernel que realiza la convulucion
    #   channel_multiplier: el numero de filtros por channel del input
    #   strides: el stride de la convolucion
    #   padding: el padding que se aplicara, por defecto 'SAME'
    #   intializer: para los pesos, por defecto GlorotNormal (Xavier)
    #   regularizer: para los pesos, por defecto L2
    #   use_bias: si se aplica bias despues de la convulucion
    #   weight_decay: hyperparametro para regularizacion
    #
    def __init__(self,
                 f_kernel,
                 name="depthwise_conv",
                 channel_multiplier=1,
                 strides=[1,1,1,1],
                 padding="SAME",
                 initializer=None,
                 regularizer=None,
                 use_bias=False,
                 weight_decay=1e-4,
                 **kwargs):
        super(depthwise_conv, self).__init__(name=name, **kwargs)

        self.f_kernel = f_kernel
        self.channel_multiplier = channel_multiplier
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.weight_decay = weight_decay

        if initializer == None:
            self.w_initializer = GlorotNormal()
        else:
            self.w_initializer = initializer
        if regularizer == None:
            self.w_regularizer = l2(weight_decay)
        else:
            self.w_regularizer = regularizer
    #
    # Serializa las propiedades de la capa
    def get_config(self):
        config = super(depthwise_conv, self).get_config()
        config.update({
            "f_kernel": self.f_kernel,
            "channel_multiplier": self.channel_multiplier,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "weight_decay": self.weight_decay
        })

    #
    # Crea weights y biases dependiendo del input_shape de call()
    #
    def build(self, input_shape):
        # print("Input shape: {}".format(input_shape.as_list()))
        in_channels = input_shape[-1]
        nc_h, nc_w = self.f_kernel

        self.w = self.add_weight(shape=[nc_h, nc_w, in_channels, self.channel_multiplier],
                                 initializer=self.w_initializer,
                                 regularizer=self.w_regularizer,
                                 trainable=True,
                                 name=self.name+"_w")

        if self.use_bias:
            self.b = self.add_weight(shape=[in_channels * self.channel_multiplier],
                                     initializer="zeros",
                                     trainable=True,
                                     name=self.name+"_b")

    #
    # Realiza la operacion al argumento inputs
    # Args:
    #   inputs: tensor de shape (batch, heigh, width, channels)
    #
    def call(self, inputs, training=None):
        conv = depthwise_conv2d(inputs, self.w, self.strides, self.padding)
        if self.use_bias:
            return bias_add(conv, self.b)

        return conv

#
# Layer que realiza una pointwise convolution, a la
# cual solo se le convolutions de 1x1
#
class pointwise_conv(layers.Layer):
    #
    # Asigna los parametros de layer para realizar la pointwise convulution
    # Args:
    #   name: nombre de la layer
    #   num_filters: numero de filtros del volumen final
    #   strides: el stride de la convolucion
    #   padding: el padding que se aplicara, por defecto 'SAME'
    #   intializer: para los pesos, por defecto GlorotNormal (Xavier)
    #   regularizer: para los pesos, por defecto L2
    #   use_bias: si se aplica bias despues de la convulucion
    #   weight_decay: hyperparametro para regularizacion
    #
    def __init__(self,
                 num_filters,
                 name="pointwise_conv",
                 strides=[1,1,1,1],
                 padding="VALID",
                 initializer=None,
                 regularizer=None,
                 use_bias=False,
                 weight_decay=1e-4,
                 **kwargs):
        super(pointwise_conv, self).__init__(name=name, **kwargs)

        # Asegura de que el num_filters sea un entero
        if type(num_filters) is float:
            num_filters = int(num_filters)

        self.f_kernel = (1, 1)
        self.num_filters = num_filters
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        if initializer == None:
            self.w_initializer = GlorotNormal()
        else:
            self.w_initializer = initializer
        if regularizer == None:
            self.w_regularizer = l2(weight_decay)
        else:
            self.w_regularizer = regularizer

    #
    # Serializa las propiedades de la capa
    def get_config(self):
        config = super(pointwise_conv, self).get_config()
        config.update({
            "num_filters": self.num_filters,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "weight_deacay": self.weight_decay
        })

    def build(self, input_shape):
        in_channels = input_shape[-1]
        nc_h, nc_w = self.f_kernel

        self.w = self.add_weight(shape=[nc_h, nc_w, in_channels, self.num_filters],
                                 initializer=self.w_initializer,
                                 regularizer=self.w_regularizer,
                                 trainable=True,
                                 name=self.name+"_w")

        if self.use_bias:
            self.b = self.add_weight(shape=[self.num_filters],
                                     initializer="zeros",
                                     trainable=True,
                                     name=self.name+"_b")


    def call(self, inputs, training=None):
        conv = conv2d(inputs, self.w, self.strides, self.padding)
        if self.use_bias:
            return bias_add(conv, self.b)

        return conv
