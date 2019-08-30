import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from ops.conv_ops import normal_conv, ReLU6, pointwise_conv
from ops.conv_blocks import BottleneckResidualBlock, basic_conv_block
from ops.conv_blocks import pwise_conv_block, separable_conv_block
from ops.model_layers import LayerList
from ops.SSD import SSD_layer

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
        a = width_multiplier
        self.classes = classes
        self.m_layers = LayerList()

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
        l = pwise_conv_block(int(a*1280), dropout=0.25, activation="ReLU6",
                name="layer_{}_conv1x1".format(len(self.m_layers)))
        self.m_layers.add(l)


        # Average Pooling y Fully Connected
        self.m_layers.add(layers.AveragePooling2D(pool_size=(7,7), strides=(1,1)))
        self.m_layers.add(layers.Flatten())
        self.m_layers.add(layers.Dense(1280))
        self.m_layers.add(layers.Dropout(0.5, name="dropout"), only_training=True)
        self.m_layers.add(layers.Dense(classes))
        self.m_layers.add(layers.Activation("softmax"))

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
            self.m_layers.add(l)

    def call(self, inputs, training=False):
        x = self.m_layers.feed_forward(inputs, training)
        return x

    @staticmethod
    def get_input_size():
        return 224




# Implementacion de SSD framework para object detection con arquitectura
# de MobileNetV2, SSD esta configurado de la siguiente manera segun paper:
#   - first SSD layer: expansion de layer 15 stride=16
#   - second and rest SSD layer: ultima layer stride=32 
class MobileNetV2_SSD(Model):
    def __init__(self, classes, width_multiplier=1):
        super(MobileNetV2_SSD, self).__init__()
        #self.classes = classes
        a = width_multiplier
        self.classes = classes
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
        self.m_layers.add(l, save_as="last_layer")

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
        l = SSD_layer(classes=self.classes, num_fmap=1, total_fmaps=5,
                img_size=320, name="ssd_layer_1")
        self.m_layers.add(l, save_as=l.name, custom_input="layer_13",
                custom_input_index=0)

        l = SSD_layer(classes=self.classes, num_fmap=2, total_fmaps=5,
                img_size=320, name="ssd_layer_2")
        self.m_layers.add(l, save_as=l.name, custom_input="last_layer")

        l = SSD_layer(classes=self.classes, num_fmap=3, total_fmaps=5,
                img_size=320, name="ssd_layer_3")
        self.m_layers.add(l, save_as=l.name, custom_input="ssd_feature_layer_1")

        l = SSD_layer(classes=self.classes, num_fmap=4, total_fmaps=5,
                img_size=320, name="ssd_layer_4")
        self.m_layers.add(l, save_as=l.name, custom_input="ssd_feature_layer_2")

        l = SSD_layer(classes=self.classes, num_fmap=5, total_fmaps=5,
                img_size=320, name="ssd_layer_5")
        self.m_layers.add(l, save_as=l.name, custom_input="ssd_feature_layer_4")

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

    @staticmethod
    def get_fmaps_array():
        return [(20, 20), (10, 10), (5, 5), (3, 3),  (1, 1)]

    @staticmethod
    def get_input_size():
        return 320



