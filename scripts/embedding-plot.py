"""
Usage:
embedding-plot.py <model-dir>

Options:
  -h --help     Show this screen.
"""

from docopt import docopt

import glob
import os
import yaml

import numpy as np

import tensorflow as tf

from tensorboard.plugins import projector

from nets import load_model
from datasets import get_dataset, normalization_cifar10

import umap

from matplotlib import pyplot as plt

BATCH_SIZE = 100

def main(model_dir):
    print(f"Getting models from {model_dir}")

    models = glob.glob(f"{model_dir}/summary.yml")

    print(f"We have {len(models)} models.")

    for i, model in enumerate(models):
        path = os.path.dirname(model)
        print(f"Model {i+1} | {path}")

        graph = tf.Graph()

        model, summary = load_model(os.path.dirname(model))
        _, test_set, small_set = get_dataset(summary["dataset"])

        ds = tf.data.Dataset \
            .from_tensor_slices(test_set)\
            .batch(BATCH_SIZE)

        log_dir = f"{path}/logs/test"

        embeddings = []
        labels = []
        for (x, y) in ds:

            mu, _ =  model.encode(x)

            embeddings.append(mu.numpy())
            labels.append(y.numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        print("Embedding's shape:", embeddings.shape)

        labels = np.array(labels).reshape(-1)
        print("Labels's shape:", labels.shape)

        with open(f"{log_dir}/embedding-meta.tsv", "w") as fhy, \
            open(f"{log_dir}/embedding-vec.tsv", "w") as fhx:
            for x, y in zip(embeddings,labels):
                fhy.write(f"{y}\n")
                fhx.write("\t".join([str(v) for v in x]) + "\n")

        reducer = umap.UMAP()

        umap_embedding = reducer.fit_transform(embeddings)

        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title("%s-%s" % (summary["class_loss"], summary["strategy"]))
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.savefig(f"{path}/umap-embedding.png")

        plt.close("all")

if __name__ == "__main__":

    arguments = docopt(__doc__)
    model_dir = arguments["<model-dir>"]

    main(model_dir)