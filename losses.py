import math

import tensorflow as tf
import tensorflow_probability as tfp

import utils

def get_optimizer(strategy, lr):
    if strategy == "oneshot":
        print("using oneshot strategy")
        return [tf.keras.optimizers.Adam(lr)], strategy, {}
    elif strategy[:3] == "seq":
        slugs = strategy.split("/")
        print(f"using {slugs[0]} strategy with {slugs[1]}")

        # one for encoder and decoder
        return (
            tf.keras.optimizers.Adam(lr), 
            tf.keras.optimizers.Adam(lr),
        ), slugs[0], utils.parse_arch(slugs[1])

@tf.function
def compute_loss(model, x, y, M=1):
    q_zgx = model.encode(x)
    
    # shape: (M, batch_size, 10)
    z = q_zgx.sample(model.M)

    # shape: (M, batch_size, 10)
    logits = model.decode(z)

    # shape: (batch_size, 10)
    one_hot = tf.one_hot(y, depth=10)

    # shape: (M, batch_size, 10)
    sm = tf.nn.softmax(logits)

    # shape: (batch_size, 10)
    mean_sm = tf.reduce_mean(sm, 0)
    pred = tf.dtypes.cast(tf.math.argmax(mean_sm, axis=1), tf.int32)

    class_loss = tf.reduce_mean( # average across all samples in batch
       -tf.reduce_sum(
           one_hot * tf.math.log(mean_sm),
           1 # sum over all classes
       )
    ) / math.log(2.)

    info_loss = tf.reduce_mean(
        tfp.distributions.kl_divergence(q_zgx, model.prior)
    ) / math.log(2.)

    IZY_bound = math.log(10, 2) - class_loss
    IZX_bound = info_loss

    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

    return class_loss + model.beta*info_loss, IZY_bound, IZX_bound, acc

@tf.function
def compute_apply_oneshot_gradients(model, batch, optimizers, epoch, opt_params):
    optimizer = optimizers[0]

    with tf.GradientTape() as tape:
        x, y = batch
        metrics = compute_loss(model, x, y)
        loss = metrics[0]

    gradients = tape.gradient(
        loss,
        model.encoder.trainable_variables + model.decoder.trainable_variables
    )
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return metrics

def apply_gradient(tape, loss, variables, optimizer):
    gradients = tape.gradient(
        loss,
        variables 
    )

    optimizer.apply_gradients(
        zip(gradients, variables)
    )

# @tf.function
def compute_apply_seq_gradients(model, batch, optimizers, epoch, opt_params):
    enc_opt, dec_opt = optimizers

    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
        x, y = batch
        metrics = compute_loss(model, x, y)
        loss = metrics[0]

    if epoch % opt_params["d"] == 0:
        apply_gradient(dec_tape, loss, model.decoder.trainable_variables, dec_opt)

    if epoch % opt_params["e"] == 0:
        apply_gradient(enc_tape, loss, model.encoder.trainable_variables, enc_opt)

    return metrics