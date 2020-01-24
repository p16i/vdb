"""
Usage:
prepare_cv_index.py [--seed=<seed> --cv=<cv>] <dataset>

Options:
  -h --help        Show this screen.
  --cv=<cv>        Number of fold for preparing cross validation [default: 5]
  --seed=<seed>    Seed [default: 99]
"""

import yaml

import numpy as np
from docopt import docopt

from sklearn.model_selection import StratifiedKFold

import datasets

OUTPUT_DIR = "./datasets/cv-index"

if __name__ == "__main__":
    arguments = docopt(__doc__, version="Preparing cross validation sets")

    dataset = arguments["<dataset>"]
    cv = int(arguments["--cv"])
    seed = int(arguments["--seed"])

    print(f"Preparing {cv}-cross-validation sets for {dataset} with seed={seed}")

    (X, y), _, _ = datasets.get_dataset(dataset)

    skf = StratifiedKFold(n_splits=cv)

    train_sets, val_sets = [], []
    for train_index, val_index in skf.split(X, y):
        train_sets.append(train_index)
        val_sets.append(val_index)

    train_sets = np.array(train_sets)
    val_sets = np.array(val_sets)
    assert train_sets.reshape(-1).shape[0] / (cv-1) == X.shape[0]
    assert val_sets.reshape(-1).shape[0] == X.shape[0]

    dest = f"{OUTPUT_DIR}/{dataset}-cv{cv}"
    print(train_sets.shape, val_sets.shape)

    np.save(f"{dest}-train", np.array(train_sets))
    np.save(f"{dest}-val", np.array(val_sets))

    print(f"Saved indices to {dest}")