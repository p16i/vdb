import tensorflow as tf
import tensorflow_probability as tfp

import losses

from nets.base import BaseNet

class Net(BaseNet):
    def __init__(self, architecture, input_shape, beta=1e-3, M=1):
        super(Net, self).__init__()

        latent_dim = architecture["z"]
        self.latent_dim = latent_dim

        num_cov_entries = latent_dim

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(units=architecture["e1"], activation=tf.nn.relu),
                tf.keras.layers.Dense(units=architecture["e2"], activation=tf.nn.relu),
                tf.keras.layers.Dense(latent_dim + num_cov_entries),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=10),
            ]
        )

        self.prior = tfp.distributions.Normal(0, 1)

        self.info_loss = losses.compute_info_loss_diag_cov
        self.class_loss = losses.compute_vdb_class_loss_tf2

        self.beta = beta
        self.M = M

    def encode(self, x):
        entries = self.encoder(x)
        mean = entries[:, :self.latent_dim]
        cov_entries = entries[:, self.latent_dim:]

        cov_entries  = tf.nn.softplus(cov_entries - 5.)

        return tfp.distributions.Normal(mean, cov_entries)