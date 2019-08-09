import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from ops.conv_ops import normal_conv, ReLU6, pointwise_conv
from ops.conv_blocks import BottleneckResidualBlock, basic_conv_block
from ops.conv_blocks import pwise_conv_block, separable_conv_block
from ops.model_layers import LayerList

#
# Implementacion de MobilenetV2, suponiendo un input size de 224x224x3
#
# ATENCION
# ahorita no funciona hay que cambiar a implementacion con LayerList
# TODO: implementar LayerList
class MobileNetV2(Model):
    #
    # Args:
    #   classes: el numero de classes que realizara predicciones
    #   width_multiplier: numero para controlar la complejidad del modelo
    #
    def __init__(self, classes, width_multiplier=1):
        super(MobileNetV2, self).__init__()
        print("ATENCION, MODELO NO FUNCIONA NECECITA MANTENIMIENTO")
        print("CHECAR CODIGO FUENTE PARA MANTENIMIENTO")
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
        l = self.crearBloques(69, 6, a*96, 3, 1, l)
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


# Implementacion de SSD framework para object detection con arquitectura
# de MobileNetV2, SSD esta configurado de la siguiente manera segun paper:
#   - first SSD layer: expansion de layer 15 stride=16
#   - second and rest SSD layer: ultima layer stride=32 
class MobileNetV2_SSD(Model):
    def __init__(self, classes, width_multiplier=1):
        super(MobileNetV2_SSD, self).__init__()
        #self.classes = classes
        a = width_multiplier
        self.m_layers = LayerList()
        self.saved_block = 13 # output que guarda para ssd_lite

        # convolucion inicial
        l = basic_conv_block(int(a*32), (3, 3), stride=2,
                dropout=0.25, activation="ReLU6", name="layer_0")
        self.m_layers.add(l)

        # los bloques de bottleneck intermedios
        self.crearBloques(32, 1, a*16, 1, 1)
        self.crearBloques(16, 6, a*24, 2, 2)
        self.crearBloques(24, 6, a*32, 3, 2)
        self.crearBloques(32, 6, a*64, 4, 2)
        self.crearBloques(69, 6, a*96, 3, 1)
        self.crearBloques(96, 6, a*160, 3, 2)
        self.crearBloques(160, 6, a*320, 1, 1)

        # ultima convolucion
        l_num = len(self.m_layers)
        l = pwise_conv_block(int(a*1280), dropout=0.25, activation="ReLU6",
                name="layer_{}_conv1x1".format(l_num))
        self.m_layers.add(l)

        # SSD extra feature layers
        l = separable_conv_block(512, 2, name="ssd_feature_layer_1")
        self.m_layers.add(l, save_as=l.name)
        l = separable_conv_block(256, 2, name="ssd_feature_layer_2")
        self.m_layers.add(l, save_as=l.name)
        l = separable_conv_block(256, 2, name="ssd_feature_layer_3")
        self.m_layers.add(l, save_as=l.name)
        l = separable_conv_block(128, 2, name="ssd_feature_layer_4")
        self.m_layers.add(l, save_as=l.name)

        # SSD classifier

    # Crea BottleneckResidualBlock n veces
    def crearBloques(self, input_channels, t, c, n, s):
        for i in range(n):
            # Solo el primer bloque tiene stride 2
            # a partir del segundo bottleneck el numero de input_channels es igual al output_channels
            if i > 0:
                s = 1
                input_channels = c

            l_num = len(self.m_layers)
            l = BottleneckResidualBlock(input_channels, int(c), stride=s, t=t,
                    name="layer_{}_BottleneckResidualBlock".format(l_num))
            save_as = None
            if l_num == self.saved_block:
                save_as = "layer_{}".format(l_num)
            self.m_layers.add(l, save_as=save_as)


    def call(self, inputs, training=False):
        x = self.m_layers.feed_forward(inputs, training)
        return x


