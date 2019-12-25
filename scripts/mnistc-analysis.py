"""
Usage:
mnistc-analysis.py [-L=<L>] <model-dir>

Options:
  -h --help     Show this screen.
  -L=<L>        Numbers of samples [default: 1,12]
"""

from docopt import docopt

import glob
import os
import yaml

import numpy as np

import tensorflow as tf

from nets import load_model
from datasets import get_dataset, normalization_mnist

from perturbation import salt_pepper_noise

# Remark: this should divide test_images.shape[0] without remainder
DATAPATH = "./datasets/mnist_c"
BATCH_SIZE = 1000

def load_mnistc(data_path=DATAPATH):
    print(glob.glob(f"{data_path}/*"))

    data = dict()
    print("loading MNIST-C")
    for directory in glob.glob(f"{data_path}/*"):
        name = os.path.basename(directory)

        test_images = np.load(f"{directory}/test_images.npy")
        test_labels = np.load(f"{directory}/test_labels.npy").astype("int32")

        print(f" {name:>15s}: {test_images.shape[0]} images | min, max: {np.min(test_images)} {np.max(test_images)}")

        data[name] = dict(
            test_images=normalization_mnist(test_images),
            test_labels=test_labels,
        )

    return data

if __name__ == "__main__":
    arguments = docopt(__doc__)

    model_dir = arguments["<model-dir>"]

    Ls = list(map(int, arguments["-L"].split(",")))

    print(f"MNIST-C Analysis: {model_dir}")
    print("Ls:", Ls)

    models = glob.glob(f"{model_dir}/summary.yml")

    print(f"We have {len(models)} models.")

    mnistc = load_mnistc()

    cat_keys = sorted(mnistc.keys())

    for i, model in enumerate(models):
        path = os.path.dirname(model)

        model, summary = load_model(path)
        print(f"{i+1}: {path}")

        assert summary["dataset"] == "mnist"

        metrics = dict(batch_size=BATCH_SIZE, categories=dict())

        for cat in cat_keys:

            ds = tf.data.Dataset \
                .from_tensor_slices(
                    (mnistc[cat]["test_images"], mnistc[cat]["test_labels"])
                ) \
                .batch(BATCH_SIZE)

            mean_acc = tf.keras.metrics.MeanTensor()
            for i, (x, y) in enumerate(ds):
                accs = list(
                    map(lambda l: model.compute_acc(x, y, l, training=False), Ls)
                )
                mean_acc.update_state(accs)

            mean_acc = mean_acc.result().numpy()

            metrics["categories"][cat] = []

            for i, L in enumerate(Ls):
                metrics["categories"][cat].append(
                    dict(L=L, accuracy=float(mean_acc[i]))
                )

        with open(f"{path}/mnistc-analysis.yml", 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)

        tf.keras.backend.clear_session()