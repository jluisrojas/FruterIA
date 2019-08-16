import math
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import l2
from ops.conv_ops import normal_conv, depthwise_conv, pointwise_conv, ReLU6
# Regresar un directoria para poder acceder modulo de otra carpeta
import sys
sys.path.append("..")
from utils import tfrecord_coco
from utils.datasets_features import bytes_feature
from tests.test_bboxes import draw_bbox
sys.path.append("ops/")

class ssd_lite_conv(layers.Layer):
    # Args:
    #   filters: numero de filtros que se aplica en total
    #   kernel: tamaño del kernel
    def __init__(self, filters, kernel=(3, 3), name="ssd_lite_conv", **kwargs):
        super(ssd_lite_conv, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel

        self.dwise = depthwise_conv(self.kernel, strides=[1, 1, 1, 1],
                padding="SAME", name=name+"_dwise_conv")
        self.dwbn = layers.BatchNormalization(name=name+"_dwise_bn")
        self.dwrelu6 = ReLU6(name=name+"_dwise_relu6")

        self.pwise = pointwise_conv(self.filters)

    def get_config(self):
        config = super(ssd_lite_conv, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel": self.kernel})

        return config

    def call(self, inputs, training=None):
        x = self.dwise(inputs)
        x = self.dwbn(inputs)
        x = self.dwrelu6(inputs)
        x = self.pwise(inputs)

        return x


class SSD_layer(layers.Layer):
    #
    # Constructor de la layer
    # Args:
    #   classes: cantidad de categorias a clasificar
    #   priors: el numero de priors de cada feature
    def __init__(self,
                 classes=3,
                 aspect_ratios=[1, 2, 3, 1/2, 1/3],
                 num_fmap=1,
                 total_fmaps=3,
                 img_size=224,
                 initializer=None,
                 regularizer=None,
                 weight_decay=1e-4,
                 name="SSD_layer", **kwargs):
        super(SSD_layer, self).__init__(name=name, **kwargs)
        
        self.classes = classes
        self.aspect_ratios = aspect_ratios

        # calcula el numero de priors dependiendo de los aspect ratios
        # siguiendo la implemetacion del paper
        self.priors = compute_num_priors(aspect_ratios)
        
        self.num_fmap = num_fmap
        self.total_fmaps = total_fmaps
        self.img_size = img_size

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
        self.conv_conf = ssd_lite_conv(self.priors*self.classes)
        """
        self.conv_conf = normal_conv(self.priors*self.classes, (3, 3), 
            name=name+"_conv_conf",
            padding="SAME")
        """

        # Realiza la prediccion del offset de las default box,
        # el numero de filtros es de num_priors * 4(dx,dy,dw,dh)
        self.conv_loc = ssd_lite_conv(self.priors*4)
        """
        self.conv_loc = normal_conv(self.priors*4, (3, 3),
                name=name+"_conv_loc",
                padding="SAME")
        """

    def get_config(self):
        config = super(SSD_layer, self).get_config()
        config.update({
            "classes": self.classes,
            "aspect_ratios": self.aspect_ratios,
            "num_fmap": self.num_fmap,
            "total_fmaps": self.total_fmaps,
            "img_size": self.img_size
        })

        return config

    
    # Recive el feature map y calcula lo siguiente:
    #   conf: tensor shape (batch, features, features, priors, classes)
    #   loc: tensor shape (batch, features, features, priors, 4(dx,dy,dw,dh)
    #   priors: tensor shape (features, features, priors, 4(cx, cy, w, h))
    # con eso se puede obtener, una version con todo junto para el loss
    #  shape[batch, features*features, priors, classes+4(dx, dy, dw, dh)+4(cx, cy, w h)]
    def call(self, inputs, training=None):
        b_size = inputs.get_shape().as_list()[0]
        features = inputs.get_shape().as_list()[1]

        conf = self.conv_conf(inputs)
        loc = self.conv_loc(inputs)
        bpriors = PriorsBoxes(batch_size=b_size, features=features, num_fmap=self.num_fmap,
                total_fmaps=self.total_fmaps, aspect_ratios=self.aspect_ratios,
                img_size=self.img_size)
        
        # reshape clasification de las convoluciones
        conf = tf.reshape(conf, [b_size, features*features,
            self.priors, self.classes])
        loc = tf.reshape(loc, [b_size, features*features,
            self.priors, 4])

        bpriors = tf.cast(bpriors, tf.float32)
        prediction = tf.concat([conf, loc, bpriors], -1)

        return prediction

#
# Metodo que calcula el numero de priors dependiendo de cuantos aspect ratios
# se usen
# Args:
#   aspect_ratios: arreglo de proporciones
# Returns:
#   priors: entero que representa el numero de default boxes
def compute_num_priors(aspect_ratios):
    priors = 0
    for ratio in aspect_ratios:
        priors += 1
        # en caso de ratio == 1, se agrega otro ratio
        if ratio == 1:
            priors += 1

    return priors

# Metodo que calcula los priorboxes de un feature map
# Args:
#   features: entero mxm de un feature map
#   num_fmap: number of feature map of m feature maps
#   total_fmaps: the total number of feature maps of the network
#   aspect_ratios: arreglo de proporciones
#   img_size: tamaño de la imagen original
# Returns:
#   Tensor with boxes loc of shape (features, features, priors, 4(cx, cy, w, h))
def PriorsBoxes(batch_size=None,
                features=None,
                num_fmap=None,
                total_fmaps=None,
                aspect_ratios=None,
                img_size=None):

    # metodo de calcula la escala de las cajas
    def compute_scale(k, m):
        s_min = 0.15
        s_max = 0.9
        s_k = s_min + (((s_max - s_min)/(m - 1))*(k - 1))

        return s_k
    
    # calcula el ancho y alto de una caja segun su escala y proporcion
    def box_size(scale, aspect_ratio):
        h = scale/math.sqrt(aspect_ratio)
        w = scale*math.sqrt(aspect_ratio)

        return h, w

    s_k = compute_scale(num_fmap, total_fmaps)
    priors = 0
    heights = []
    widths = []

    # Calcula los tamaños de las bounding boxes
    for ar in aspect_ratios:
        priors += 1
        bh, bw = box_size(s_k, ar)
        heights.append(bh)
        widths.append(bw)

        # cuando el ratio es 1, se calcual otro segun el paper
        if ar == 1:
            priors += 1
            s_k_p = compute_scale(num_fmap+1, total_fmaps)
            bh, bw = box_size(s_k_p, ar)
            heights.append(bh)
            widths.append(bw)


    default_boxes = np.zeros((features, features, priors, 4))

    cell_size = 1 / features
    cell_center = cell_size / 2
    for i in range(features):
        for j in range(features):
            for p in range(priors):
                h, w = heights[p], widths[p]
                x = j*cell_size + cell_center
                y = i*cell_size + cell_center

                default_boxes[i, j, p, 0] = x
                default_boxes[i, j, p, 1] = y
                default_boxes[i, j, p, 2] = w
                default_boxes[i, j, p, 3] = h

    default_boxes *= img_size
    default_boxes = tf.convert_to_tensor(default_boxes)
   
    # Checa si se especifico un batch_size en los parametros
    # si si, agrega una dimension y duplica las otras mediante tiling
    if batch_size == None:
        return default_boxes 
    else:
        default_boxes = tf.reshape(default_boxes, [features*features, priors, 4])
        default_boxes = tf.expand_dims(default_boxes, 0)
        default_boxes = tf.tile(default_boxes, [batch_size, 1, 1, 1])

        return default_boxes

#
# Convirte cordenadas (cx,cy,w,h) a (x,y,w,h)
# Args:
#   loc: tensorf of shape [4]
# Returns:
#   x, y, w, h: posicion y tamaño como enteros
def bbox_center_to_rect(loc):
    w = loc[2]
    h = loc[3]
    x = loc[0] - (w/2)
    y = loc[1] - (h/2)

    return int(x), int(y), int(w), int(h)

#
# Converte cordenadas (x, y, w, h) a (cx, cy, w, h)
# Args:
#   loc: tensor of shape [4]
# Returns:
#   cx, cy, w, h: posicion y tamaño como enteros
def bbox_rect_to_center(loc):
    w = loc[2]
    h = loc[3]
    cx = loc[0] + (w/2)
    cy = loc[1] + (h/2)

    return int(x), int(y), int(w), int(h)
#
# Converte cordenadas (x, y, w, h) a (cx, cy, w, h)
# Args:
#   loc: tensor of shape [4]
# Returns:
#   tensor of shape [4] (cx, cy, w, h)
def tbbox_rect_to_center(loc):
    w = loc[2]
    h = loc[3]
    cx = loc[0] + (w/2)
    cy = loc[1] + (h/2)

    return tf.convert_to_tensor(np.array([cx, cy, w, h]))

def rect_to_coord(box):
    _box = np.copy(box)
    box[0] = _box[0]
    box[1] = _box[1]
    box[2] = box[0] + _box[2]
    box[3] = box[1] + _box[3]

    return box

#
# Calcula el jaccard overlap o intesection over union IOU
# entre dos bounding boxes
# Args:
#   t_boxA: tensor of shape [4] de (x, y, w, h)
#   t_boxB: tensor of shape [4] de (x, y, w, h)
# Returns:
#   iou = float de 0.0 a 1.0
def intersection_over_union(t_boxA, t_boxB):
    # Se convierte los tensores  a numpy arrays
    boxA = np.array(t_boxA)
    boxB = np.array(t_boxB)

    boxA = rect_to_coord(boxA)
    boxB = rect_to_coord(boxB)
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


# Pipeline encargada de procesar un batch de training set, para
# realizar la estrategia de matching para crear la informacion de
# ground truth
class SSD_data_pipeline(object):
    # Metodos del objeto
    #   init: iniicalizacion con los parametros de la arquitectura
    #   process: procesa un batch de imagenes

    # Inicializacion de los parametros acerca de la arquitectura de la red
    # Argumentos:
    #   aspect_ratios: arreglo conteniendo los aspect ratios segun el paper
    #   feature_maps: arreglo conteniendo pares con los tamaños de los f maps
    #   categories_arr: arreglo de strings con los nombre de las categorias
    #   img_size: entero que contiene el numero de pixeles de un lado de la img
    def __init__(self, 
                 aspect_ratios=[1, 2, 3, 1/2, 1/3], 
                 feature_maps=None, 
                 categories_arr=None,
                 img_size=None):
        self.aspect_ratios = aspect_ratios
        self.feature_maps = feature_maps
        self.categories_arr = categories_arr
        self.num_categories = len(self.categories_arr)
        self.img_size = img_size
        self.num_priors = compute_num_priors(aspect_ratios)

        self.categories_index = {}
        # Creacion de indices de las categorias
        for i in range(len(self.categories_arr)):
            self.categories_index[self.categories_arr[i]] = i

    # Procesa un batch de imagenes del data set de coco para convertirlos a training data
    # Argumentos:
    #   path_to_tfrecord: string al dataset de coco en formato tfrecord
    def preprocess_tfrecord_coco(self, path_to_tfrecord, res_path):
        total_fmaps = len(self.feature_maps)
        dataset_tfrecord_coco = tfrecord_coco.parse_dataset(path_to_tfrecord)
        
        it = iter(dataset_tfrecord_coco)

        writer = tf.io.TFRecordWriter(res_path)
        i = 0

        for img_data in dataset_tfrecord_coco:
            print("Processing image {}".format(i+1))
            i += 1

            # Decodificacion de imagen
            image_string = np.frombuffer(img_data["img/str"].numpy(), np.uint8)
            decoded_image = cv2.imdecode(image_string, cv2.IMREAD_COLOR)

            # tamaños original de la imagen
            y_, x_ = decoded_image.shape[0], decoded_image.shape[1]

            # resize de la imagen y la convierte a un tensor
            decoded_image = cv2.resize(decoded_image, (self.img_size, self.img_size))
            image_tensor = tf.convert_to_tensor(decoded_image)
            image_tensor /= 255 # normaliza entre 0-1

            # rescale de bbounding box
            x_scalar = self.img_size / x_
            y_scalar = self.img_size / y_

            # Decodificacion de anotaciones
            cats, locs = self.decode_bboxes(img_data["img/bboxes/category"],
                    img_data["img/bboxes/x"], img_data["img/bboxes/y"],
                    img_data["img/bboxes/width"],
                    img_data["img/bboxes/height"], x_scalar, y_scalar)

            # Crea mask de los indices correctos
            mask = self.mask_indices(img_data["img/bboxes/category"])
            
            # Aplica mask
            cats = tf.boolean_mask(cats, mask)
            locs = tf.boolean_mask(locs, mask)

            num_bboxes = locs.get_shape().as_list()[0]
            y_true = []
            num_matches = 0
            
            """
            # debugging
            for loc in locs:
                draw_bbox(img=decoded_image, bbox=loc)
            """
                
            for f in range(total_fmaps):
                m = self.feature_maps[f][0]
                priors = PriorsBoxes(features=m, num_fmap=f+1, total_fmaps=total_fmaps, 
                        aspect_ratios=self.aspect_ratios, img_size=self.img_size)
                feature_y = np.zeros((m, m, self.num_priors, 1 + self.num_categories + 4))

                for i in range(m):
                    for j in range(m):
                        for p in range(self.num_priors):
                            prior = priors[i][j][p]
                            prior = bbox_center_to_rect(prior)
                            for b in range(num_bboxes):
                                iou = intersection_over_union(prior, locs[b])
                                if iou > 0.5:
                                    num_matches += 1
                                    match = tf.ones([1, 1])

                                    # Se obtiene la categoria y se convierte a one hot
                                    cat = cats[b].numpy().decode("UTF-8")
                                    cat_one_hot = [self.categories_index[cat]]
                                    cat_one_hot = tf.one_hot(cat_one_hot, self.num_categories)

                                    # se calcula la diferencia del prior al  ground truth
                                    diff = tf.cast(tf.abs(tbbox_rect_to_center(prior) - tbbox_rect_to_center(locs[b])),
                                            tf.float32)
                                    diff = tf.expand_dims(diff, 0)
                                    
                                    match_y = tf.concat([match, cat_one_hot, diff], -1)
                                    feature_y[i][j][p] = match_y

                                    """
                                    draw_bbox(img=decoded_image, bbox=prior,
                                            color=(255, 0, 0))
                                    """
                
                feature_y = tf.convert_to_tensor(feature_y)
                if f == 0:
                    y_true = tf.identity(tf.reshape(feature_y, [m*m, self.num_priors, 1 +
                        self.num_categories + 4]))
                else:
                    feature_y = tf.reshape(feature_y, [m*m, self.num_priors, 1 +
                        self.num_categories + 4])
                    y_true = tf.concat([y_true, tf.identity(feature_y)], 0)

            """
            cv2.imshow("test", decoded_image)
            cv2.waitKey(0)
            """
            if num_matches > 0:
                y_true = tf.cast(y_true, tf.float32)
                image_tensor = tf.cast(image_tensor, tf.float32)
                data = {
                    "x": bytes_feature(tf.io.serialize_tensor(image_tensor)),
                    "y": bytes_feature(tf.io.serialize_tensor(y_true))
                }
                example = tf.train.Example(features=tf.train.Features(feature=data))
                writer.write(example.SerializeToString())

        writer.close()

    # proces y cambia el formato de las annotacions de las images de tfrecord
    # Args:
    #   cats: sparse tensor de strings con las categorias
    #   x: sparse tensor con las coordenadas x del bbox
    #   y: sparse tensor con las coordenadas y del bbox
    #   width: sparse tensor con en ancho del bbox
    #   height: sparse tensor con la altura del bbox
    #   x_scalar: scalar horizontal que se le aplica al bbox por el resize de la img
    #   y_scalar: scalar vertical que se le aplica al bbox por el resize de la img
    def decode_bboxes(self, cats, x, y, width, height, x_scalar, y_scalar):
        cats_tensor = []
        loc_tensor = []

        for i in cats.indices:
            cat = cats.values[i[0]].numpy().decode("UTF-8")
            _x = x.values[i[0]].numpy() * x_scalar
            _y = y.values[i[0]].numpy() * y_scalar
            _w = width.values[i[0]].numpy() * x_scalar
            _h = height.values[i[0]].numpy() * y_scalar
            
            cats_tensor.append(cat)
            loc_tensor.append([_x, _y, _w, _h])

        return tf.convert_to_tensor(cats_tensor), tf.convert_to_tensor(loc_tensor)

    # Funcion que regresa un mask booleano de los bbox que se van usar para el
    # modelo, segun las categirias a clasificar
    # Args:
    #   sparse_tensor: sparse tensor con las cadenas de las categorias
    def mask_indices(self, sparse_tensor):
        indices = sparse_tensor.indices
        mask = []

        for i in indices:
            index = i.numpy()[0]
            cat = sparse_tensor.values[index]
            cat = cat.numpy().decode("UTF-8")
            mask.append(cat in self.categories_arr)

        return mask

def SSD_load_dataset(path_to_tfrecord):
    raw_data = tf.data.TFRecordDataset(path_to_tfrecord)
    format_ = {
        "x": tf.io.FixedLenFeature([],
            tf.string),
        "y": tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_function(example):
        return tf.io.parse_single_example(example, format_)

    data = raw_data.map(_parse_function)


    def _parse_tensors(example):
        x = tf.io.parse_tensor(example["x"], tf.float32)
        y_true = tf.io.parse_tensor(example["y"], tf.float32)

        return x, y_true

    tensor_data = data.map(_parse_tensors)

    return tensor_data



