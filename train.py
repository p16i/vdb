"""
Usage: train.py --epoch=<epoch> --dataset=<dataset> --beta=<beta> -M=<M> --lr=<lr> <model>

Options:
  -h --help                 Show this screen.
  --epoch=<epoch>           Number of epochs [default: 10]
  --dataset=<dataset>       Dataset for training [default: mnist]
  --beta=<beta>             Value of β [default: 0.001]
  --lr=<lr>                 Learning rate [default: 0.0001]
  -M=<M>                    Value of M [default: 1]
"""

import time
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from docopt import docopt

# ours
import vdb
import plot_helper
import utils
import tfutils

def train(model, dataset, epochs, beta, M, lr):

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
    
    print(f"Training with {model} on {dataset} for {epochs} epochs (lr={lr})")
    print(f"Params: beta={beta} M={M} lr={lr}")

    optimizer = tf.keras.optimizers.Adam(lr)
    experiment_name = utils.get_experiment_name(f"vdb-{dataset}")

    print(f"Experiment name: {experiment_name}")
    artifact_dir = f"{ARTIFACT_DIR}/{experiment_name}"

    os.makedirs(f"{artifact_dir}/figures")

    train_log_dir = f"{artifact_dir}/logs/train"
    test_log_dir = f"{artifact_dir}/logs/test"

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    _, architecture = model.split("/")
    architecture = utils.parse_arch(architecture)
    model = vdb.VDB(architecture, train_images.shape[1:], beta=beta, M=M)

    # todo: get from dataset module
    indices = np.random.choice(test_labels.shape[0], 1000, replace=False)
    selected_labels = test_labels[indices]
    selected_images = test_images[indices, :]

    metric_labels = ["loss", "I_YZ", "I_XZ", "accuracy"]

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        print(f"Epoch {epoch}")
        m = tf.keras.metrics.MeanTensor("train_metrics")
        for train_x in train_dataset:
            metrics = vdb.compute_apply_oneshot_gradients(model, train_x, optimizer)
            m.update_state(metrics)

        m = m.result().numpy()
        print(utils.format_metrics("Train", m))

        tfutils.log_metrics(train_summary_writer, metric_labels, m, epoch)

        end_time = time.time()

        if model.latent_dim == 2:
            img_buff = plot_helper.plot_2d_representation(
                model,
                (selected_images, selected_labels),
            )

            tfutils.summary_image(
                test_summary_writer,
                img_buff,
                "latent-representation",
                epochs
            )

        m = tf.keras.metrics.MeanTensor("test_metrics")
        for test_x in test_dataset:
            metrics = vdb.compute_loss(model, *test_x)
            m.update_state(metrics)

        m = m.result().numpy()
        print(utils.format_metrics("Test", m))

        tfutils.log_metrics(test_summary_writer, metric_labels, m, epoch)

        print(f"--- Time elapse for current epoch {end_time - start_time}")

    print(f"Please see artifact at: {artifact_dir}")

if __name__ == "__main__":
    arguments = docopt(__doc__, version="VDB Experiment 1.0")

    # todo: write a function to do magic type cast
    model = arguments["<model>"]
    beta = float(arguments["--beta"])
    dataset = arguments["--dataset"]
    epoch = int(arguments["--epoch"])
    M = int(arguments["-M"])
    lr = float(arguments["--lr"])

    train(model, dataset, epoch, beta, M, lr)
