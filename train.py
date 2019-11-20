import time
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# ours
import vdb
import plot_helper
import utils

# Parameter Setting
ARTIFACT_DIR = "./artifacts"
TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000


# todo: create data module for this
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32")

train_labels = train_labels.astype("int32")
test_labels = test_labels.astype("int32")

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
 
# todo: make it parameter
epochs = 20
latent_dim = 2
beta = 1e-4
dataset = "mnist"
lr = 1e-3

optimizer = tf.keras.optimizers.Adam(lr)
experiment_name = utils.get_experiment_name(f"vdb-{dataset}")

print(f"Experiment name: {experiment_name}")
artifact_dir = f"{ARTIFACT_DIR}/{experiment_name}"

os.makedirs(f"{artifact_dir}/figures")

train_log_dir = f"{artifact_dir}/logs/train"
test_log_dir = f"{artifact_dir}/logs/test"

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

model = vdb.VDB(train_images.shape[1:], latent_dim, beta=beta)

# for 2d plot
indices = np.random.choice(test_labels.shape[0], 1000, replace=False)
selected_labels = test_labels[indices]
selected_images = test_images[indices, :]

for epoch in range(1, epochs + 1):
    start_time = time.time()
    print(f"Epoch {epoch}")
    m = tf.keras.metrics.MeanTensor("train_metrics")
    for train_x in train_dataset:
        metrics = vdb.compute_apply_oneshot_gradients(model, train_x, optimizer)
        m.update_state(metrics)

    print(
        ":: Train >> Loss: %.4f | I(Z; Y) >= %.4f | I(X; Z) <= %.4f | acc = %.4f"
        % tuple(m.result().numpy())
    )

    end_time = time.time()

    if latent_dim == 2 and epoch % 5 == 0:
        plot_helper.plot_2d_representation(
            f"{artifact_dir}/figures/2d-latent-%04d-epoch.png" % epoch,
            model,
            (selected_images, selected_labels),
            title=f"Epoch %04d" % epoch
        )

    m = tf.keras.metrics.MeanTensor("test_metrics")
    for test_x in test_dataset:
        metrics = vdb.compute_loss(model, *test_x)
        m.update_state(metrics)

    print(
        ":: Test  >> Loss: %.4f | I(Z; Y) >= %.4f | I(X; Z) <= %.4f | acc = %.4f"
        % tuple(m.result().numpy())
    )

    print(f"--- Time elapse for current epoch {end_time - start_time}")

print(f"Please see artifact at: {artifact_dir}")