"""
Usage:
train.py  [--epoch=<epoch> --beta=<beta> -M=<M> --lr=<lr>] --strategy=<strategy> --dataset=<dataset> <model>

Options:
  -h --help                 Show this screen.
  --dataset=<dataset>       One from {mnist, fashion_mnist, cifar10} [default: mnist]
  --beta=<beta>             Value of β [default: 0.001]
  -M=<M>                    Value of M [default: 1]
  --lr=<lr>                 Learning rate [default: 0.001]
  --epoch=<epoch>           Number of epochs [default: 10]
  --strategy=<strategy>     Optimizaton strategy "oneshot" or "seq/d:1|e:10" [default: oneshot]
                            "seq/e:10|d:1" means "decoder get update every epoch while 10 for encoder".
"""

import time
import os

import yaml
import numpy as np
from docopt import docopt

import tensorflow as tf

# our core modules
import vdb
import datasets
import losses

# our helper modules
import plot_helper
import utils
import tfutils

ARTIFACT_DIR = "./artifacts"
BATCH_SIZE = 100 # todo: keep it fixed for now

def train(model, dataset, epochs, beta, M, lr, strategy):
    model_conf = model

    # todo: create data module for this
    train_set, test_set, small_set = datasets.get_dataset(dataset)

    TRAIN_BUF, TEST_BUF = datasets.dataset_size[dataset]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_set) \
    .shuffle(TRAIN_BUF).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_set) \
    .shuffle(TEST_BUF).batch(BATCH_SIZE)
    
    print(f"Training with {model} on {dataset} for {epochs} epochs (lr={lr})")
    print(f"Params: beta={beta} M={M} lr={lr} strategy={strategy}")

    optimizers, strategy_name, opt_params = losses.get_optimizer(strategy, lr)

    apply_gradient_func = getattr(losses, f"compute_apply_{strategy_name}_gradients")

    experiment_name = utils.get_experiment_name(f"vdb-{dataset}")

    print(f"Experiment name: {experiment_name}")
    artifact_dir = f"{ARTIFACT_DIR}/{experiment_name}"

    # Prepare experiment's directory and logging
    os.makedirs(f"{artifact_dir}/figures")

    train_log_dir = f"{artifact_dir}/logs/train"
    test_log_dir = f"{artifact_dir}/logs/test"

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Instantiate model
    _, architecture = model.split("/")
    architecture = utils.parse_arch(architecture)
    model = vdb.VDB(architecture, datasets.input_dims[dataset], beta=beta, M=M)

    metric_labels = ["loss", "I_YZ", "I_XZ", "accuracy"]

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        print(f"Epoch {epoch}")

        m = tf.keras.metrics.MeanTensor("train_metrics")
        for batch in train_dataset:
            x, y = batch
            q_zgx = model.encode(x)
    
            # shape: (M, batch_size, 10)
            z = q_zgx.sample(model.M)

            # shape: (M, batch_size, 10)
            logits = model.decode(z)

            # shape: (batch_size, 10)
            one_hot = tf.one_hot(y, depth=10)

            # shape: (M, batch_size, 10)
            sm = tf.nn.softmax(logits)
            sum_zero = tf.reduce_sum(tf.where(sm == 0, 1, 0))
            if sum_zero.numpy() > 0:
                print(sum_zero)
                raise SystemExit("found zeror")

            metrics = apply_gradient_func(
                model, batch, optimizers, epoch, opt_params
            )
            m.update_state(metrics)

        m = m.result().numpy()
        print(utils.format_metrics("Train", m))
        tfutils.log_metrics(train_summary_writer, metric_labels, m, epoch)
        train_metrics = m

        end_time = time.time()

        if model.latent_dim == 2:
            img_buff = plot_helper.plot_2d_representation(
                model,
                small_set,
                title="Epoch=%d Strategy=%s  Beta=%f" % (epoch, strategy, beta)
            )

            tfutils.summary_image(
                test_summary_writer,
                img_buff,
                "latent-representation",
                epochs
            )

        m = tf.keras.metrics.MeanTensor("test_metrics")
        for batch in test_dataset:
            metrics = losses.compute_loss(model, *batch)
            m.update_state(metrics)

        m = m.result().numpy()
        print(utils.format_metrics("Test", m))
        tfutils.log_metrics(test_summary_writer, metric_labels, m, epoch)
        test_metrics = m

        print(f"--- Time elapse for current epoch {end_time - start_time}")

    print(train_metrics.astype(float))
    print(dict(zip(metric_labels, list(train_metrics.astype(float)))))


    summary = dict(
        dataset=dataset,
        model=model_conf,
        strategy=strategy,
        beta=beta,
        epoch=epoch,
        M=M,
        lr=lr,
        metrics=dict(
            train=dict(zip(metric_labels, train_metrics.astype(float).tolist())),
            test=dict(zip(metric_labels, test_metrics.astype(float).tolist())),
        )
    )

    with open(f"{artifact_dir}/summary.yml", 'w') as f:
        print(summary)
        yaml.dump(summary, f, default_flow_style=False)

    print(f"Please see artifact at: {artifact_dir}")

if __name__ == "__main__":
    arguments = docopt(__doc__, version="VDB Experiment 1.0")

    # todo: write a function to do magic type cast
    model = arguments["<model>"]
    strategy = arguments["--strategy"]
    beta = float(arguments["--beta"])
    dataset = arguments["--dataset"]
    epoch = int(arguments["--epoch"])
    M = int(arguments["-M"])
    lr = float(arguments["--lr"])

    train(model, dataset, epoch, beta, M, lr, strategy)
