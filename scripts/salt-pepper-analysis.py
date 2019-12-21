"""
Usage:
salt-pepper-analysis.py  [-p=<p>] <model-dir>

Options:
  -h --help     Show this screen.
  -p=<p>            Probability of flipling, [default: 0.1,0.2,0.4]
"""

from docopt import docopt

import glob
import yaml

import numpy as np

import tensorflow as tf

from nets import load_model
from datasets import get_dataset

from perturbation import salt_paper_noise

# Remark: this should divide test_images.shape[0] without remainder
BATCH_SIZE = 1000
L=12

def salt_pepper_analysis(model_path, probs):

    model, summary = load_model(model_path)

    _, (test_images, test_labels), _ = get_dataset(summary["dataset"])

    assert test_images.shape[0] % BATCH_SIZE == 0, \
        f"BATCH_SIZE={BATCH_SIZE} doesn't match with {test_images.shape[0]}"

    extreme_values = np.min(test_images), np.max(test_images)


    metrics = dict(L=L, batch_size=BATCH_SIZE, settings=[])

    for p in [0, *probs]:
        perturbed_images = salt_paper_noise(
            test_images,
            p=p,
            extreme_values=extreme_values
        )

        dataset = tf.data.Dataset.from_tensor_slices((perturbed_images, test_labels)) \
            .shuffle(test_images.shape[0]).batch(BATCH_SIZE)

        mean_acc = tf.keras.metrics.Mean()
        for i, (x, y) in enumerate(dataset):
            acc = model.compute_acc(x, y, L)
            mean_acc.update_state(acc)

        mean_acc = float(mean_acc.result().numpy())

        metrics['settings'].append(dict(p=p, accuracy=mean_acc))

    with open(f"{model_path}/salt-pepper-analysis.yml", 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    probs = list(map(float, arguments['-p'].split(",")))
    model_dir = arguments['<model-dir>']

    print(f"Salt-Pepper Analysis: {model_dir}")
    print("Probabilities: ", probs)

    models = glob.glob(f"{model_dir}/summary.yml")

    print(f"We have {len(models)} models.")

    for  model in models:
        path = "/".join(model.split("/")[:-1])
        salt_pepper_analysis(path, probs)
