import tensorflow as tf
import tensorflow_probability as tfp

import losses
from nets import resnet20

from nets.base import BaseNet

class Net(BaseNet):
    """
    Please see resnet20.py for more information about the implementation
    """

    def __init__(self, architecture, in_shape, num_class, cov_type,  batch_norm=True, beta=1e-3, M=1):
        super(Net, self).__init__(architecture, cov_type, num_class, beta, M)

        self.encoder = tf.keras.Sequential(
            [
                resnet20.ResNet(
                    resnet20.BasicBlock, [7, 7, 7],
                    output_dim=self.parameters_for_latent,
                    batch_norm=batch_norm
                ), #  out_dims (-1, 64)
            ]
        )