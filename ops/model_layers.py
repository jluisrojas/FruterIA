import tensorflow as tf
# Script con classes para el manejo de las layer de un modelo
# TODO:
#   * Permitir extrar partes de la red para crear nuevas redes

# Clase para elementos de la lista con las capas del modelo
class LayerRef():
    # Args:
    #   layer: keras layer class que computa la entrada
    #   only_training: si solo se llama cuando esta en fase de entrenamiento
    #   save_output: si se va a guardar el output de la layer
    #   output_index: el indice de la lista de outputs
    #   custom_input: cadena para diccionario de layers guardadas
    #   custom_input_index: indice del output de una layer guardada
    def __init__(self, layer, 
            only_training=False, 
            save_output=False,
            output_index=-1,
            custom_input=None,
            custom_input_index=-1,
            training_arg=True):
        self.layer = layer
        self.only_training = only_training
        self.save_output = save_output
        self.output_index = output_index
        self.custom_input = custom_input
        self.custom_input_index = custom_input_index
        self.training_arg = training_arg
        self.outputs = []

    # Args:
    #   inputs: tensor con input al layer
    #   training: si el modelo esta en fase de entrenamiento
    # Returns:
    #   x: tensor con resultado de la layer
    def __call__(self, inputs, training=False):
        print(inputs)
        if not training and self.only_training:
            return inputs
        else:
            x = self.layer(inputs, training=training)
            if type(x) is not list:
                x = [x]

            if self.save_output:
                self.outputs = x

            out = tf.identity(x[self.output_index])
            print("Layer output: {}".format(out))

            return out

    # Args:
    #   inputs: tensor con input al layer
    # Returns:
    #   x: tensor con resultado de la layer
    def call2(self, inputs):
        print(inputs)
        x = self.layer(inputs)
        if type(x) is not list:
            x = [x]

        if self.save_output:
            self.outputs = x

        out = tf.identity(x[self.output_index])
        print("Layer output: {}".format(out))

        return out


# Estructura de dato que nos permite guardar las referencias a las layers
# de un modelo de una red neuronal, todo la implementacion es siguiendo el
# API de keras para las layers. Permite las siguientes funcionalidades
#   - Guardar los outputs de ciertas layers
#   - Que el input de una layer sea el output de una layer guardada
#   - Tener acceso a las layers mediante indices y diccionarios
#   - Feedforward
class LayerList():
    def __init__(self):
        self.layers_list = []
        self.layers_dict = {}
        self.saved_ref = {}

    # Agrega un layer a la lista
    # Args:
    #   layers: keras Layer que realiza la computacion
    #   only_training: si solo se llama durante entrenamiento
    #   save_as: string que guarda en un dict el output, None no guarda
    #   output_index: del output del layer(lista) cual es el indice que se usa
    #   custom_input: string del nombre del output que se guarda para input
    #   custom_input_index: del input guarda cual es el indice
    def add(self, layer, 
            only_training=False, 
            save_as=None,
            output_index=-1,
            custom_input=None,
            custom_input_index=-1,
            training_arg=True):
        save_output = False if save_as == None else True
        l = LayerRef(layer, only_training=only_training,
                save_output=save_output,
                output_index=output_index,
                custom_input=custom_input,
                custom_input_index=custom_input_index,
                training_arg=training_arg)
        
        self.layers_list.append(l)
        if layer.name != None:
            self.layers_dict[layer.name] = (len(self.layers_list)-1, l)
        if save_output:
            self.saved_ref[save_as] = l

    # Realiza feed forward en las layers
    def feed_forward(self, inputs, training=None):
        x = inputs
        ##print(x.get_shape())

        for layer in self.layers_list:
            print(layer.layer.name)
            # Si hay input especial
            if layer.custom_input != None:
                x = self.saved_ref[layer.custom_input].outputs[layer.custom_input_index]

            if layer.training_arg:
                x = layer(x, training)
            else:
                x = layer.call2(x)

            print("En feed forward {}".format(x))
            #print(x.get_shape())

        return x

    # Regresa cuantas layers tiene la lista
    def __len__(self):
        return len(self.layers_list)



