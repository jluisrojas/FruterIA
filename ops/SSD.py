

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import l2
from ops.conv_ops import normal_conv

# Idea general de la layer
#   - recibe input de un feature map
#   - con el input realiza las convoluciones para obtener los sig:
#       - confidence
#       - bbox loc
#       - prior box ?

class SSD_layer(layers.Layer):
    def __init__(self,
                 classes=3,
                 priors=4,
                 initializer=None,
                 regularizer=None,
                 weight_decay=1e-4,
                 name="SSD_layer", **kwargs):
        super(SSD_layer, self).__init__(name=name, **kwargs)
        
        self.classes = classes
        self.priors = priors 

        if initializer == None:
            self.w_initializer = GlorotNormal()
        else:
            self.w_initializer = initializer

        if regularizer == None:
            self.w_regularizer = l2(weight_decay)
        else:
            selw.w_regularizer = regularizer

        # Realiza la prediccion de la seguriada de la clase y del tipo
        # de bounding box
        self.conv_conf = normal_conv(self.classes*self.priors, (3, 3),
                                     name=name+"_conv_conf",
                                     padding="SAME")

        # Realiza la prediccion del offset de las default box,
        # el numero de filtros es de num_priors * 4(cx,cy,w,h)
        self.conv_loc = normal_conv(self.priors*4, (3, 3),
                                     name=name+"_conv_loc",
                                     padding="SAME")

    def get_config(self):
        config = super(SSD_layer, self).get_config()
        config.update({
            "classes": self.classes,
            "priors": self.priors
        })

    
    # Recive el feature map y regresa lo siguiente:
    #   conf: tensor shape (batch, features, features, classes, priors)
    #   loc: tensor shape (batch, features, features, priors, 4(cx,cy,w,h)
    #   priors:
    def call(self, inputs):
        b_size = inputs.get_shape().as_list()[0]
        features = inputs.get_shape().as_list()[1]

        conf = self.conv_conf(inputs)
        loc = self.conv_loc(inputs)
        
        conf = tf.reshape(conf, [b_size, features, features,
            self.classes, self.priors])
        loc = tf.reshape(loc, [b_size, features, features,
            self.priors, 4])

        return conf, loc

def PriorsBoxes(features,
                num_priors=4,
                aspect_ratios=[1, 2, 3, 1/2, 1/3]):
    s_min = 0.2
    s_max = 0.9
