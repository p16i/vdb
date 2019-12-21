import tensorflow as tf
import tensorflow_probability as tfp

import losses

from nets.base import BaseNet

class Net(BaseNet):
    def __init__(self, architecture, input_shape, cov_type, beta=1e-3, M=1):
        super(Net, self).__init__(architecture, cov_type)

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(units=architecture["e1"], activation=tf.nn.relu),
                tf.keras.layers.Dense(units=architecture["e2"], activation=tf.nn.relu),
                tf.keras.layers.Dense(self.parameters_for_latent),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=10),
            ]
        )

        self.beta = beta
        self.M = M