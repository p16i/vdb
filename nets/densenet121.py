import tensorflow as tf

from nets.base import BaseNet

from efficientnet.tfkeras import EfficientNetB7
from tensorflow.keras.applications.densenet import DenseNet121

class Net(BaseNet):
    def __init__(self, architecture, in_shape, num_class, cov_type,  batch_norm=True, beta=1e-3, M=1):
        super(Net, self).__init__(architecture, cov_type, num_class, beta, M)

        self.encoder = tf.keras.Sequential(
            [
                DenseNet121(weights=None, include_top=False),
                tf.keras.layers.Dense(self.parameters_for_latent),
            ]
        )