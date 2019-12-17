"""
Usage:
train.py  [--epoch=<epoch> --beta=<beta> -M=<M> --lr=<lr> --output-dir=<output-dir>] --strategy=<strategy> --dataset=<dataset> <model>

Options:
  -h --help                 Show this screen.
  --dataset=<dataset>       One from {mnist, fashion_mnist, cifar10} [default: mnist]
  --beta=<beta>             Value of β [default: 0.001]
  -M=<M>                    Value of M [default: 1]
  --lr=<lr>                 Learning rate [default: 0.0001]
  --epoch=<epoch>           Number of epochs [default: 200]
  --strategy=<strategy>     Optimizaton strategy "oneshot" or "seq/d:1|e:10" [default: oneshot]
                            "seq/e:10|d:1" means "decoder get update every epoch while 10 for encoder".
  --output-dir=<output-dir>   [default: ./artifacts]
"""

import time
import os

import json
import yaml
import numpy as np
from docopt import docopt

import tensorflow as tf
import tensorflow_probability as tfp

# our core modules
import nets
import datasets
import losses

# our helper modules
import plot_helper
import utils
import tfutils

BATCH_SIZE = 100 # todo: keep it fixed for now

metric_labels = ["loss", "I_YZ", "I_XZ"]
acc_labels = ["accuracy_L1", "accuracy_L12"]

def evaluate(model, test_dataset, tsb_writer, M):
    m = tf.keras.metrics.MeanTensor("test_metrics")
    am = tf.keras.metrics.MeanTensor("test_acc_metrics")
    for batch in test_dataset:
        metrics = losses.compute_loss(model, *batch, M)
        m.update_state(metrics)
        x, y = batch
        am.update_state(
            [
                losses.compute_acc(model, x, y, 1),
                losses.compute_acc(model, x, y, 12)
            ]
        )

    m = m.result().numpy()
    am = am.result().numpy()
    print(utils.format_metrics("Test", m, am))
    tfutils.log_metrics(tsb_writer, metric_labels, m, epoch)
    tfutils.log_metrics(tsb_writer, acc_labels, am, epoch)

    return m.astype(float).tolist() + am.astype(float).tolist()


def train_algo1(
        model, optimizers, train_dataset, tsb_writer, M,
        lr_labels, strategy_name, opt_params
    ):
    apply_gradient_func = getattr(losses, f"compute_apply_{strategy_name}_gradients")

    m = tf.keras.metrics.MeanTensor("train_metrics")
    am = tf.keras.metrics.MeanTensor("train_acc_metrics")
    for i, batch in enumerate(train_dataset):

        metrics = apply_gradient_func(
            model, batch, optimizers, epoch, opt_params, M
        )
        m.update_state(metrics)

        x, y = batch
        am.update_state(
            [
                losses.compute_acc(model, x, y, 1),
                losses.compute_acc(model, x, y, 12)
            ]
        )

    return m, am


def train_algo2(
        model, optimizers, train_dataset, tsb_writer, M,
        lr_labels, strategy_name, opt_params
    ):

    if "current_k" not in opt_params:
        opt_params["current_k"] = opt_params["e"]

    m = tf.keras.metrics.MeanTensor("train_metrics")
    am = tf.keras.metrics.MeanTensor("train_acc_metrics")
    for i, batch in enumerate(train_dataset):

        if opt_params["current_k"] > 0:
            metrics = losses.compute_apply_gradients_algo2_enc(
                model, model.encoder.trainable_variables,
                batch, optimizers[0], M
            )

            opt_params["current_k"] -= 1
        else:
            metrics = losses.compute_apply_gradients_algo2_dec(
                model, model.decoder.trainable_variables,
                batch, optimizers[1], M
            )

            opt_params["current_k"] -= opt_params['e']
        m.update_state(metrics)

        x, y = batch
        am.update_state(
            [
                losses.compute_acc(model, x, y, 1),
                losses.compute_acc(model, x, y, 12)
            ]
        )

    return m, am


def train(model, dataset, epochs, beta, M, initial_lr, strategy, output_dir):
    model_conf = model

    train_set, test_set, small_set = datasets.get_dataset(dataset)

    TRAIN_BUF, TEST_BUF = datasets.dataset_size[dataset]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_set) \
        .shuffle(TRAIN_BUF).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_set) \
        .shuffle(TEST_BUF).batch(BATCH_SIZE)
    
    print(f"Training with {model} on {dataset} for {epochs} epochs (lr={initial_lr})")
    print(f"Params: beta={beta} M={M} lr={initial_lr} strategy={strategy}")

    optimizers, strategy_name, opt_params = losses.get_optimizer(
        strategy,
        lr,
        dataset,
        BATCH_SIZE
    )

    network_name, architecture = model.split("/")
    experiment_name = utils.get_experiment_name(
        f"{network_name}-{dataset}"
    )

    print(f"Experiment name: {experiment_name}")
    artifact_dir = f"{output_dir}/{experiment_name}"
    print(f"Artifact directory: {artifact_dir}")

    train_log_dir = f"{artifact_dir}/logs/train"
    test_log_dir = f"{artifact_dir}/logs/test"

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Instantiate model
    architecture = utils.parse_arch(architecture)

    model = nets.get_network(network_name)(
        architecture, datasets.input_dims[dataset], beta=beta, M=M
    )

    model.build(input_shape=(BATCH_SIZE, *datasets.input_dims[dataset]))
    model.summary()

    lr_labels = list(map(lambda x: f"lr_{x}", range(len(optimizers))))

    train_step = train_algo2 if strategy[:3] == "alt"  else train_algo1

    print("Using trainstep: ", train_step)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        print(f"Epoch {epoch}")

        m, am = train_step(
            model,
            optimizers,
            train_dataset,
            train_summary_writer,
            M,
            lr_labels,
            strategy_name,
            opt_params
        )

        m = m.result().numpy()
        am = am.result().numpy()

        print(utils.format_metrics("Train", m, am))

        tfutils.log_metrics(train_summary_writer, metric_labels, m, epoch)
        tfutils.log_metrics(train_summary_writer, acc_labels, am, epoch)

        tfutils.log_metrics(
            train_summary_writer,
            lr_labels,
            map(lambda opt: opt._decayed_lr(tf.float32), optimizers),
            epoch
        )

        train_metrics = m.astype(float).tolist() + am.astype(float).tolist()
        end_time = time.time()

        test_metrics = evaluate(model, test_dataset, test_summary_writer, M)

        print(f"--- Time elapse for current epoch {end_time - start_time}")

    summary = dict(
        dataset=dataset,
        model=model_conf,
        strategy=strategy,
        beta=beta,
        epoch=epoch,
        M=M,
        lr=initial_lr,
        metrics=dict(
            train=dict(zip(metric_labels + acc_labels, train_metrics)),
            test=dict(zip(metric_labels + acc_labels, test_metrics)),
        )
    )

    if model.latent_dim == 2:
        plot_helper.plot_2d_representation(
            model,
            small_set,
            title="Epoch=%d Strategy=%s  Beta=%f M=%f" % (epoch, strategy, beta, M),
            path=f"{artifact_dir}/latent-representation.png"
        )

    with train_summary_writer.as_default():
        tf.summary.text(
            "setting",
            json.dumps(summary, sort_keys=True, indent=4),
            step=0
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
    output_dir = arguments['--output-dir']

    train(model, dataset, epoch, beta, M, lr, strategy, output_dir)
