import tensorflow as tf
import tensorflow_probability as tfp

import losses

from nets.base import BaseNet

class Net(BaseNet):
    def __init__(self, architecture, input_shape, beta=1e-3, M=1):
        super(Net, self).__init__()

        latent_dim = architecture["z"]
        self.latent_dim = latent_dim

        num_cov_entries = int(latent_dim * (latent_dim + 1) / 2)

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(units=architecture["e1"], kernel_initializer="he_uniform", bias_initializer="he_uniform"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Lambda(lambda x: tf.nn.elu(x)),
                tf.keras.layers.Dense(units=architecture["e2"], kernel_initializer="he_uniform", bias_initializer="he_uniform"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Lambda(lambda x: tf.nn.elu(x)),
                tf.keras.layers.Dense(latent_dim + num_cov_entries, kernel_initializer="he_uniform", bias_initializer="he_uniform"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=10),
            ]
        )

        self.prior = tfp.distributions.MultivariateNormalDiag(
            tf.zeros(latent_dim), tf.ones(latent_dim)
        )

        self.class_loss = losses.compute_vib_class_loss
        self.compute_info_loss = losses.compute_info_loss_diag_cov

        self.beta = beta
        self.M = M

    def encode(self, x):
        entries = self.encoder(x)
        mean = entries[:, :self.latent_dim]
        cov_entries = entries[:, self.latent_dim:]

        # build lower triangular matrix for Cholesky Decomposition
        tril_raw = tfp.math.fill_triangular(cov_entries)
        diag_entries = tf.nn.softplus(tf.eye(self.latent_dim) * tril_raw - 5.)

        # use indepedent gaussians only when K > 2
        factor = tf.where(self.latent_dim == 2, 0.01, 0)
        off_diag_entries = (1-tf.eye(self.latent_dim)) * tril_raw * factor

        tril = diag_entries + off_diag_entries

        return tfp.distributions.MultivariateNormalTriL(mean, tril)

    def call(self, x, L=1):
        q_zgx = self.encode(x)

        # shape: (M, batch_size, 10)
        z = q_zgx.sample(L)

        # shape: (M, batch_size, 10)
        logits = tf.dtypes.cast(self.decode(z), tf.float64)

        return q_zgx, logits