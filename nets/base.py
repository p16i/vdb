import tensorflow as tf
import tensorflow_probability as tfp

from losses import compute_info_loss_diag_cov, \
    compute_info_loss_full_cov, \
    mean_softmax_from_logits

class BaseNet(tf.keras.Model):
    def __init__(self, architecture, cov_type, num_class, beta, M):
        super(BaseNet, self).__init__()

        # this will be used by all architectures inheriting BaseNet
        self.latent_dim = architecture["z"]

        # compute parameters for z's layer and set up prior
        print(f"Using {cov_type} covariance")
        if cov_type == "diag":
            self.parameters_for_latent = self.latent_dim * 2

            self.prior = tfp.distributions.Normal(0, 1)

            self.info_loss = compute_info_loss_diag_cov
            self._build_z_dist = _build_multivariate_normal_with_diag_cov
        elif cov_type == "full":
            # for building lower triangular matrix for Cholesky's decomposition
            num_tril_entries = int(self.latent_dim * (self.latent_dim + 1) / 2)

            self.parameters_for_latent = self.latent_dim + num_tril_entries

            self.prior = tfp.distributions.MultivariateNormalDiag(
                tf.zeros(self.latent_dim), tf.ones(self.latent_dim)
            )

            self.info_loss = compute_info_loss_full_cov

            self._build_z_dist = _build_multivariate_normal_with_full_cov
        else:
            raise NotImplementedError("{cov_type} not implemented")

        print(f"Latent dims: {self.latent_dim}")
        print(f"Parameters for latent: {self.parameters_for_latent}")

        self.beta = beta
        self.M = M

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=num_class),
            ]
        )
    
    def encode(self, x, training=False):
        entries = self.encoder(x, training=training)
        mu = entries[:, :self.latent_dim]
        cov_entries = entries[:, self.latent_dim:]

        return mu, cov_entries

    @tf.function
    def decode(self, z):
        logits = self.decoder(z)

        return logits

    @tf.function
    def call(self, x, L=1, training=False):
        mu, cov_entries = self.encode(x, training=training)

        q_zgx = self._build_z_dist(mu, cov_entries)

        # shape: (M, batch_size, 10)
        z = q_zgx.sample(L)

        # shape: (M, batch_size, 10)
        logits = tf.dtypes.cast(self.decode(z), tf.float64)

        return (mu, cov_entries), logits

    @tf.function
    def compute_acc(self, x, y, L, training=False):
        _, logits = self(x, L=L, training=training)

        mean_sm = mean_softmax_from_logits(logits)

        pred = tf.dtypes.cast(tf.math.argmax(mean_sm, axis=1), tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

        return acc

def _build_multivariate_normal_with_diag_cov(mu, cov_entries):
    # this bias -5 comes from VIB paper
    cov_entries  = tf.nn.softplus(cov_entries - 5.)

    return tfp.distributions.Normal(mu, cov_entries)

def _build_multivariate_normal_with_full_cov(mu, cov_entries):
    latent_dim = mu.shape[-1]

    # build lower triangular matrix for Cholesky Decomposition
    tril_raw = tfp.math.fill_triangular(cov_entries)
    diag_entries = tf.nn.softplus(tf.eye(latent_dim) * tril_raw - 5.)

    factor = 0.01 # this factor comes from the VIB paper
    off_diag_entries = (1-tf.eye(latent_dim)) * tril_raw * factor

    tril = diag_entries + off_diag_entries

    return tfp.distributions.MultivariateNormalTriL(mu, tril)