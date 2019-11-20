import tensorflow as tf
import tensorflow_probability as tfp

class VDB(tf.keras.Model):
  def __init__(self, input_shape, latent_dim, beta=1e-3, M=1):
    super(VDB, self).__init__()
    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential(
      [
          tf.keras.layers.Flatten(input_shape=input_shape),
          tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
          tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
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

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits

clz_loss = tf.keras.losses.SparseCategoricalCrossentropy()

@tf.function
def compute_loss(model, x, y):
    q_zgx = model.encode(x)
    
    z = q_zgx.sample()
    logits = model.decode(z)
    pred = tf.dtypes.cast(tf.math.argmax(logits, axis=1), tf.int32)

    class_loss = clz_loss(
            y,
            logits,
        ) / tf.math.log(2.)

    info_loss = tf.reduce_sum(
        tf.reduce_mean(
          tfp.distributions.kl_divergence(q_zgx, model.prior), 0
        )
    ) / tf.math.log(2.)

    IZY_bound = tf.math.log(10.0) / tf.math.log(2.) - class_loss
    IZX_bound = info_loss

    acc = tf.reduce_mean(tf.cast(pred == y, tf.float32))

    return class_loss + model.beta*info_loss, IZY_bound, IZX_bound, acc

@tf.function
def compute_apply_oneshot_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        x, y = x
        metrics = compute_loss(model, x, y)
        loss = metrics[0]

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return metrics