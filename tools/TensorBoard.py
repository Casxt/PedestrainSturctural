from keras.callbacks import TensorBoard
import keras.backend as K


class MultiLabelTensorBoard(TensorBoard):
    def __init__(self, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
