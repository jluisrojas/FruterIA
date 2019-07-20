import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from ops.conv_ops import normal_conv, ReLU6, pointwise_conv
from ops.conv_blocks import BottleneckResidualBlock

#
# Implementacion de MobilenetV2, suponiendo un input size de 224x224x3
#
class MobileNetV2(Model):
    #
    # Args:
    #   classes: el numero de classes que realizara predicciones
    #   width_multiplier: numero para controlar la complejidad del modelo
    #
    def __init__(self, classes, width_multiplier=1):
        super(MobileNetV2, self).__init__()
        #self.classes = classes
        a = width_multiplier
        self.m_layers = []

        # convolucion inicial
        self.m_layers.append(normal_conv(int(a*32), (3, 3), strides=[1, 2, 2, 1], name="1_conv2d"))
        self.m_layers.append(layers.BatchNormalization(name="1_conv2d_bn"))
        self.m_layers.append(ReLU6(name="1_conv2d_relu6"))
        self.m_layers.append(layers.Dropout(0.25, name="dropout"))

        l = 2
        # los bloques de bottleneck intermedios
        l = self.crearBloques(32, 1, a*16, 1, 1, l)
        l = self.crearBloques(16, 6, a*24, 2, 2, l)
        l = self.crearBloques(24, 6, a*32, 3, 2, l)
        l = self.crearBloques(32, 6, a*64, 4, 2, l)
        l = self.crearBloques(64, 6, a*96, 3, 1, l)
        l = self.crearBloques(96, 6, a*160, 3, 2, l)
        l = self.crearBloques(160, 6, a*320, 1, 1, l)

        # ultima convolucion
        self.m_layers.append(pointwise_conv(int(a*1280), name="{}_conv2d1x1".format(l)))
        self.m_layers.append(layers.BatchNormalization())
        self.m_layers.append(ReLU6())
        self.m_layers.append(layers.Dropout(0.25, name="dropout"))

        # Average Pooling y Fully Connected
        self.m_layers.append(layers.AveragePooling2D(pool_size=(7,7), strides=(1,1)))
        self.m_layers.append(layers.Flatten())
        self.m_layers.append(layers.Dense(1280))
        self.m_layers.append(layers.Dropout(0.5, name="dropout"))
        self.m_layers.append(layers.Dense(classes))
        self.m_layers.append(layers.Activation("softmax"))


    # Crea BottleneckResidualBlock n veces
    def crearBloques(self, input_channels, t, c, n, s, l):
        for i in range(n):
            # Solo el primer bloque tiene stride 2
            # a partir del segundo bottleneck el numero de input_channels es igual al output_channels
            if i > 0:
                s = 1
                input_channels = c

            self.m_layers.append(
                BottleneckResidualBlock(input_channels,
                                        int(c),
                                        stride=s,
                                        t=t,
                                        name="{}_BottleneckResidualBlock".format(l))
            )
            l = l + 1

        return l

    def call(self, inputs, training=False):
        x = inputs
        for l in self.m_layers:
            print(x.get_shape().as_list())
            # Asegura de solo realisar dropout durante entrenamiento
            if (l.name == "dropout" and training == False) == False:
                x = l(x)

        return x
