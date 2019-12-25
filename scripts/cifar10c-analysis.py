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
from datasets import get_dataset, normalization_cifar10

from perturbation import salt_pepper_noise

DATAPATH = "./datasets/CIFAR-10-C"
SAMPLES_PER_SEVERITY = 10000

# Remark: this should divide test_images.shape[0] without remainder
BATCH_SIZE = 1000


@tf.function
def data_normalisation(x):
    return normalization_cifar10(x)

def load_cifar10c(data_path=DATAPATH):

    data = dict()
    print("loading CIFAR10-C")

    data["labels"] = np.load(f"{data_path}/labels.npy").astype("int32")
    print("labels shape:", data["labels"].shape)

    data["categories"] = []

    for file in glob.glob(f"{data_path}/*.npy"):
        name = os.path.basename(file).split(".")[0]

        if name == "labels":
            continue

        data["categories"].append(name)

    return data

if __name__ == "__main__":
    arguments = docopt(__doc__)

    model_dir = arguments["<model-dir>"]

    Ls = list(map(int, arguments["-L"].split(",")))

    print(f"CIFAR-10 Analysis: {model_dir}")
    print("Ls:", Ls)

    models = glob.glob(f"{model_dir}/summary.yml")

    print(f"We have {len(models)} models.")


    _, test_set, _ = get_dataset("cifar10")

    cifar10c = load_cifar10c()
    cat_keys = sorted(cifar10c["categories"])
    print(cat_keys, len(cat_keys))
    assert len(cat_keys) == 19

    for i, model in enumerate(models):
        path = os.path.dirname(model)

        model, summary = load_model(path)

        assert summary["dataset"] == "cifar10"
        print(f"{i+1}: {path}")

        metrics = dict(batch_size=BATCH_SIZE, categories=dict())

        for cat in cat_keys:
            metrics["categories"][cat] = []

            images = np.load(f"{DATAPATH}/{cat}.npy").astype(np.float32)

            # loop for each level of severity
            for s in range(5):

                # get start and end indices for each level
                six, eix = s*SAMPLES_PER_SEVERITY, (s+1)*SAMPLES_PER_SEVERITY

                # prepare dataset
                ds = tf.data.Dataset \
                    .from_tensor_slices(
                        (
                            images[six:eix],
                            cifar10c["labels"][six:eix]
                        )
                    ) \
                    .batch(BATCH_SIZE)

                mean_acc = tf.keras.metrics.MeanTensor()

                for i, (x, y) in enumerate(ds):
                    x = data_normalisation(x)
                    accs = list(
                        map(lambda l: model.compute_acc(x, y, l, training=False), Ls)
                    )
                    mean_acc.update_state(accs)

                mean_acc = mean_acc.result().numpy()

                level_metrics = []

                for i, L in enumerate(Ls):
                    level_metrics.append(
                        dict(L=L, accuracy=float(mean_acc[i]))
                    )
                
                metrics["categories"][cat].append(dict(
                    severity=s+1, # to make it match with the actual level
                    metrics=level_metrics
                ))

        # verify test_set accuracy again
        ds_test = tf.data.Dataset \
            .from_tensor_slices(test_set) \
            .batch(BATCH_SIZE)

        mean_acc = tf.keras.metrics.MeanTensor()
        for (x, y) in ds_test:
            accs = list(
                map(lambda l: model.compute_acc(x, y, l, training=False), Ls)
            )
            mean_acc.update_state(accs)

        mean_acc = mean_acc.result().numpy()

        metrics["clean_image"] = []
        for i, L in enumerate(Ls):
            metrics["clean_image"].append(
                dict(L=L, accuracy=float(mean_acc[i]))
            )

        with open(f"{path}/cifar10c-analysis.yml", 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)

        tf.keras.backend.clear_session()