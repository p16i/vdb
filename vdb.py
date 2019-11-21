import tensorflow as tf
import tensorflow_probability as tfp

class VDB(tf.keras.Model):
    def __init__(self, architecture, input_shape, beta=1e-3, M=1):
        super(VDB, self).__init__()

        latent_dim = architecture["z"]
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(units=architecture["e1"], activation=tf.nn.relu),
                tf.keras.layers.Dense(units=architecture["e2"], activation=tf.nn.relu),
                tf.keras.layers.Dense(latent_dim + int(latent_dim * (latent_dim + 1) / 2)),
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

        self.beta = beta
        self.M = M

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        entries = self.encoder(x)
        mean = entries[:, :self.latent_dim]
        tril_entries = entries[:, self.latent_dim:]

        # build lower triangular matrix for Cholesky Decomposition
        tril_raw = tfp.math.fill_triangular(entries[:, self.latent_dim:])
        diag_entries = tf.nn.softplus(tf.eye(self.latent_dim) * tril_raw - 5.)
        off_diag_entries = (1-tf.eye(self.latent_dim)) * tril_raw * 0.01

        tril = diag_entries + off_diag_entries

        return tfp.distributions.MultivariateNormalTriL(mean, tril)

    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def call(self, x, L=1):
        q_zgx = self.encode(x)

        # shape: (M, batch_size, 10)
        z = q_zgx.sample(L)

        # shape: (M, batch_size, 10)
        logits = tf.dtypes.cast(self.decode(z), tf.float64)

        # shape: (M, batch_size, 10)
        sm = tf.nn.softmax(logits - tf.reduce_max(logits, 2, keepdims=True))

        # shape: (batch_size, 10)
        mean_sm = tf.reduce_mean(sm, 0)

        return q_zgx, mean_sm