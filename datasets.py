import numpy as np

import tensorflow as tf

input_dims = {
    "mnist": (28, 28, 1),
    "fashion_mnist": (28, 28, 1),
    "cifar10": (32, 32, 3)
}

dataset_size = {
    "mnist": (60000, 10000),
    "fashion_mnist": (60000, 10000),
    "cifar10": (50000, 10000),
}

def get_dataset(name):
    ds_method = getattr(tf.keras.datasets, name)
    (train_images, train_labels), (test_images, test_labels) = ds_method.load_data()

    dim  = input_dims[name]
    train_images = train_images.reshape(train_images.shape[0], *dim)\
        .astype("float32")
    test_images = test_images.reshape(test_images.shape[0], *dim)\
        .astype("float32")

    train_labels = train_labels.astype("int32").reshape(-1)
    test_labels = test_labels.astype("int32").reshape(-1)

    ## Normalizing the images to the range of [-1, 1]
    train_images = 2*(train_images / 255.) - 1
    test_images  = 2*(test_images / 255.) - 1

    ## Binarization
    # if not name == "cifar10":
    #     train_images[train_images >= .5] = 1.
    #     train_images[train_images < .5] = 0.
    #     test_images[test_images >= .5] = 1.
    #     test_images[test_images < .5] = 0.

    np.random.seed(101)
    indices = np.random.choice(test_labels.shape[0], 1000, replace=False)
    selected_labels = test_labels[indices]
    selected_images = test_images[indices, :]

    return (train_images, train_labels), \
        (test_images, test_labels), \
        (selected_images, selected_labels)
