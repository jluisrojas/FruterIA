import tensorflow as tf

# Serie de funciones que permite la serializacion de datasets medinate
# TFRecords para el manejo eficiente de las bases de datos, estas funciones
# permiten que el formato de la informacion se pueda guardar de manera lineal

# Returns a bytes_list from a string / byte
def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Returns a float_list from a float / double
def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# Returns an int64_list from a bool / enum / int / uint
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Returns a bytes_list from a list of strings / bytes
def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

# Returns a float_list from a list of floats / doubles
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Returns an int64_list from a list of  bool / enum / int / uint
def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
