import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

import vdb

import tensorflow_probability as tfp

import plot_helper

# Data Preparation
TRAIN_BUF = 60000
BATCH_SIZE = 100

TEST_BUF = 10000

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

train_labels = train_labels.astype('int32')
test_labels = test_labels.astype('int32')

## Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

## Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)) \
  .shuffle(TRAIN_BUF).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)) \
  .shuffle(TEST_BUF).batch(BATCH_SIZE)

@tf.function
def compute_loss(model, x, y):
    q_zgx = model.encode(x)
    
    z = q_zgx.sample()
    logits = model.decode(z)

    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      y, logits
    ) / tf.math.log(2.)

    info_loss = tf.reduce_sum(
        tf.reduce_mean(
          tfp.distributions.kl_divergence(q_zgx, model.prior), 0
        )
    ) / tf.math.log(2.)

    return class_loss + model.beta*info_loss

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    x, y = x
    loss = compute_loss(model, x, y)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
 

optimizer = tf.keras.optimizers.Adam(1e-4)

epochs = 5
latent_dim = 2

model = vdb.VDB(train_images.shape[1:], latent_dim, beta=1e-2)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    compute_apply_gradients(model, train_x, optimizer)
  end_time = time.time()

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, *test_x))
    loss = loss.result()
    print('Epoch: {}, Test set loss: {}, '
          'time elapse for current epoch {}'.format(epoch, loss, end_time - start_time))

indices = np.random.choice(test_labels.shape[0], 1000, replace=False)
selected_labels = test_labels[indices]
selected_images = test_images[indices, :]

plot_helper.plot_2d_representation(
  "./figures/2d-latent.png",
  model, (selected_images, selected_labels))
