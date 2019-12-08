import tensorflow as tf
import tensorflow_probability as tfp

class BaseNet(tf.keras.Model):
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        entries = self.encoder(x)
        mean = entries[:, :self.latent_dim]
        cov_entries = entries[:, self.latent_dim:]

        cov_entries  = tf.nn.softplus(cov_entries - 5.)

        return tfp.distributions.Normal(mean, cov_entries)

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

        return q_zgx, logits