import tensorflow as tf

from nets.base import BaseNet

from efficientnet.tfkeras import EfficientNetB7
from tensorflow.keras.applications.densenet import DenseNet121

output_size = {
    (32, 32, 3): (1024,)
}

class Net(BaseNet):
    def __init__(self, architecture, in_shape, num_class, cov_type,  batch_norm=True, beta=1e-3, M=1):
        super(Net, self).__init__(architecture, cov_type, num_class, beta, M)

        self.encoder = tf.keras.Sequential(
            [
                DenseNet121(input_shape=in_shape, weights=None, include_top=False),
                tf.keras.layers.Reshape(output_size[in_shape]),
                tf.keras.layers.Dense(self.parameters_for_latent),
            ]
        )