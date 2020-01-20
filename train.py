"""
Usage:
train.py  [--batch-size=<batch-size> --epoch=<epoch> --beta=<beta> -M=<M> --lr=<lr>]
    [--output-dir=<output-dir> --class-loss=<class-loss> --cov-type=<cov-type>]
    [--lr-schedule=<schedule>]
    (--strategy=<strategy> --dataset=<dataset>) <model>

Options:
  -h --help                   Show this screen.
  --dataset=<dataset>         One from {mnist, fashion_mnist, cifar10} [default: mnist]
  --beta=<beta>               Value of β [default: 0.001]
  -M=<M>                      Value of M [default: 1]
  --lr=<lr>                   Learning rate [default: 0.0001]
  --epoch=<epoch>             Number of epochs [default: 200]
  --strategy=<strategy>       Optimizaton strategy "oneshot" or "algo1/d:1|e:10" [default: oneshot]
                              "algo1/e:10|d:1" means "decoder get update every epoch while 10 for encoder".
  --output-dir=<output-dir>   [default: ./artifacts]
  --class-loss=<class-loss>   Class loss {vdb, vib} [default: vdb]
  --cov-type=<cov-type>       Type of covariance {diag, full} [default: diag]
  --batch-size=<batch-size>   Batch size [default: 100]
  --lr-schedule=<schedule>    LR Schedule [default: constant]
"""

import time
import os

import json
import yaml
import numpy as np
import time

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

metric_labels = ["loss", "I_YZ", "I_XZ"]
acc_labels = ["accuracy_L1", "accuracy_L12"]

def evaluate(model, test_dataset, tsb_writer, M, epoch):
    m = tf.keras.metrics.MeanTensor("test_metrics")
    am = tf.keras.metrics.MeanTensor("test_acc_metrics")
    for batch in test_dataset:
        metrics = losses.compute_loss(model, *batch, M, training=False)
        m.update_state(metrics)
        x, y = batch
        am.update_state(
            [
                model.compute_acc(x, y, 1, training=False),
                model.compute_acc(x, y, 12, training=False)
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
        lr_labels, strategy_name, opt_params, epoch
    ):
    apply_gradient_func = getattr(losses, f"compute_apply_{strategy_name}_gradients")

    m = tf.keras.metrics.MeanTensor("train_metrics")
    am = tf.keras.metrics.MeanTensor("train_acc_metrics")
    for i, batch in enumerate(train_dataset):

        metrics = apply_gradient_func(
            model, batch, optimizers, epoch, opt_params, M, training=True
        )
        m.update_state(metrics)

        x, y = batch
        am.update_state(
            [
                model.compute_acc(x, y, 1, training=True),
                model.compute_acc(x, y, 12, training=True)
            ]
        )

    return m, am


def train_algo2(
        model, optimizers, train_dataset, tsb_writer, M,
        lr_labels, strategy_name, opt_params, epoch
    ):

    if "current_k" not in opt_params:
        opt_params["current_k"] = opt_params["k"]

    m = tf.keras.metrics.MeanTensor("train_metrics")
    am = tf.keras.metrics.MeanTensor("train_acc_metrics")
    for i, batch in enumerate(train_dataset):

        if opt_params["current_k"] > 0:
            metrics = losses.compute_apply_gradients_algo2_enc(
                model, model.encoder.trainable_variables,
                batch, optimizers[0], M, training=True
            )

            opt_params["current_k"] -= 1
        else:
            metrics = losses.compute_apply_gradients_algo2_dec(
                model, model.decoder.trainable_variables,
                batch, optimizers[1], M, training=True
            )

            opt_params["current_k"] = opt_params["k"]
        m.update_state(metrics)

        x, y = batch
        am.update_state(
            [
                model.compute_acc(x, y, 1, training=True),
                model.compute_acc(x, y, 12, training=True)
            ]
        )

    return m, am


def train(
        model, dataset, epochs, batch_size, beta, M,
        initial_lr, lr_schedule, strategy, output_dir, class_loss, cov_type
    ):

    model_conf = model

    train_set, test_set, small_set = datasets.get_dataset(dataset)

    TRAIN_BUF, TEST_BUF = datasets.dataset_size[dataset]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_set) \
        .shuffle(TRAIN_BUF).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_set) \
        .shuffle(TEST_BUF).batch(batch_size)
    
    print(f"Training with {model} on {dataset} for {epochs} epochs (lr={initial_lr}, schedule={lr_schedule})")
    print(f"Params: batch-size={batch_size} beta={beta} M={M} lr={initial_lr} strategy={strategy}")

    optimizers, strategy_name, opt_params = losses.get_optimizer(
        strategy,
        lr,
        lr_schedule,
        dataset,
        batch_size
    )

    network_name, architecture = model.split("/")
    experiment_name = utils.get_experiment_name(
        f"{network_name}-{class_loss}-{cov_type}-{dataset}"
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
        architecture,
        datasets.input_dims[dataset],
        datasets.num_classes[dataset],
        cov_type,
        beta=beta, M=M
    )

    model.build(input_shape=(batch_size, *datasets.input_dims[dataset]))
    model.summary()

    print(f"Class loss: {class_loss}")
    model.class_loss = getattr(losses, f"compute_{class_loss}_class_loss")

    lr_labels = list(map(lambda x: f"lr_{x}", range(len(optimizers))))

    train_step = train_algo2 if strategy.split("/")[0] == "algo2" else train_algo1

    print("Using trainstep: ", train_step)

    start_time = time.time()

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
            opt_params,
            epoch
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

        test_metrics = evaluate(model, test_dataset, test_summary_writer, M, epoch)

        print(f"--- Time elapse for current epoch {end_time - start_time}")
    
    end_time = time.time()
    elapsed_time = (end_time-start_time) / 60.

    summary = dict(
        dataset=dataset,
        model=model_conf,
        strategy=strategy,
        beta=beta,
        epoch=epoch,
        M=M,
        lr=initial_lr,
        lr_schedule=lr_labels,
        metrics=dict(
            train=dict(zip(metric_labels + acc_labels, train_metrics)),
            test=dict(zip(metric_labels + acc_labels, test_metrics)),
        ),
        class_loss=class_loss,
        cov_type=cov_type,
        batch_size=batch_size,
        elapsed_time=elapsed_time # in minutes
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

    model.save_weights(f"{artifact_dir}/model")

    print(f"Training took {elapsed_time:.4f} minutes")
    print(f"Please see artifact at: {artifact_dir}")

if __name__ == "__main__":
    arguments = docopt(__doc__, version="VDB Experiment 1.0")

    model = arguments["<model>"]
    strategy = arguments["--strategy"]
    beta = float(arguments["--beta"])
    dataset = arguments["--dataset"]
    epoch = int(arguments["--epoch"])
    M = int(arguments["-M"])
    lr = float(arguments["--lr"])
    lr_schedule = arguments['--lr-schedule']
    output_dir = arguments['--output-dir']
    class_loss = arguments['--class-loss']
    cov_type = arguments['--cov-type']
    batch_size = int(arguments['--batch-size'])

    train(
        model, dataset, epoch, batch_size, beta, M, lr, lr_schedule,
        strategy, output_dir,
        class_loss,
        cov_type
    )
