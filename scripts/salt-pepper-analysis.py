"""
Usage:
salt-pepper-analysis.py  [-p=<p> -L=<L>] <model-dir>

Options:
  -h --help     Show this screen.
  -p=<p>        Probability of flipling, [default: 0.1,0.2,0.4]
  -L=<L>        Number of prediction samplings, [default: 1, 12]
"""

from docopt import docopt

import glob
import os
import yaml

import numpy as np

import tensorflow as tf

from nets import load_model
from datasets import get_dataset

from perturbation import salt_pepper_noise

# Remark: this should divide test_images.shape[0] without remainder
BATCH_SIZE = 1000
L = 1

if __name__ == "__main__":
    arguments = docopt(__doc__)

    probs = list(map(float, arguments["-p"].split(",")))
    Ls = list(map(int, arguments["-L"].split(",")))

    model_dir = arguments['<model-dir>']

    print(f"Salt-Pepper Analysis: {model_dir}")
    print("Probabilities: ", probs)
    print("L for samples: ", Ls)

    models = glob.glob(f"{model_dir}/summary.yml")

    print(f"We have {len(models)} models.")

    model, summary = load_model(os.path.dirname(models[0]))

    _, (test_images, test_labels), _ = get_dataset(summary["dataset"])

    assert test_images.shape[0] % BATCH_SIZE == 0, \
        f"BATCH_SIZE={BATCH_SIZE} doesn't match with {test_images.shape[0]}"

    extreme_values = np.min(test_images), np.max(test_images)

    del model

    perturbed_image_set = dict()

    # include p=0 as well for sanity checks
    probs = [0, *probs]

    for p in probs:
        # cache this pertubed images
        perturbed_image_set[p] = salt_pepper_noise(
            test_images,
            p=p,
            extreme_values=extreme_values
        )

    for i, model in enumerate(models):
        path = os.path.dirname(model)

        model, summary = load_model(path)
        print(f"{i+1}: {path}")

        metrics = dict(batch_size=BATCH_SIZE, settings=[])

        for p in probs:
            perturbed_images = perturbed_image_set[p]

            ds = tf.data.Dataset \
                .from_tensor_slices((perturbed_images, test_labels)) \
                .shuffle(test_images.shape[0]).batch(BATCH_SIZE)

            mean_acc = tf.keras.metrics.MeanTensor()
            for i, (x, y) in enumerate(ds):
                accs = list(
                    map(lambda l: model.compute_acc(x, y, l, training=False), Ls)
                )
                mean_acc.update_state(accs)

            mean_acc = mean_acc.result().numpy()

            for i, L in enumerate(Ls):
                metrics['settings'].append(
                    dict(p=p, L=L, accuracy=float(mean_acc[i]))
                )

        with open(f"{path}/salt-pepper-analysis.yml", 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)