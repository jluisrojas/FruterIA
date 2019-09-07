import tensorflow as tf
import os
import json

class TrainingCheckPoints(tf.keras.callbacks.Callback):
    def __init__(self, folder_path, relative_epoch=0):
        super(TrainingCheckPoints, self).__init__()

        self.folder_path = folder_path
        self.relative_epoch = relative_epoch

    def on_train_begin(self, logs=None):
        self.best_loss = np.Inf
        self.checkpoint_num = self.relative_epoch

    def on_epoch_end(self, epoch, logs=None):
        # Verifica que el modelo tenga learning rate
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Checa si mejoro el loss para hacer un checkpoint
        current_loss = logs.get("loss")
        if current_loss < self.best_loss:
            print("[TRAINING] Creating model checkpoint.")

            self.model.save(self.folder_path+"model_checkpoint_{}.h5".format(self.checkpoint_num))

            if self.checkpoint_num > 0:
                os.remove(self.folder_path+"model_checkpoint_{}.h5".format(self.checkpoint_num-1))

            self.checkpoint_num += 1
            self.best_loss = current_loss
        
        # Guarda el estado actual de entrenamiento, por si se quiere continuar
        training_state = {
            "learning_rate": float(tf.keras.backend.get_value(self.model.optimizer.lr)),
            "epoch": self.checkpoint_num
        }

        with open(self.folder_path+"training_state.json", "w") as writer:
            json.dump(training_state, writer, indent=4)


